from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchcodec.decoders import VideoDecoder

from vatix.network.wan21.wan.modules.vae import WanVAE


class ClipDataset(Dataset):
    def __init__(self, video_list, root, num_frames=9, stride=4, out_h=320, out_w=416):
        self.video_list = video_list
        self.root = root
        self.num_frames = num_frames
        self.stride = stride
        self.out_h = out_h
        self.out_w = out_w

    def _resize_decoded_frames(self, frames):
        if frames.ndim != 4:
            raise RuntimeError(f"Expected decoded frames with 4 dims, got shape={tuple(frames.shape)}")

        if frames.shape[1] == 3:
            nchw = frames
        elif frames.shape[-1] == 3:
            nchw = frames.permute(0, 3, 1, 2)
        else:
            raise RuntimeError(
                f"Unsupported decoded frame layout (no RGB channel of size 3): shape={tuple(frames.shape)}"
            )

        nchw = nchw.float().div(255.0)
        nchw = F.interpolate(nchw, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return nchw.to(torch.float16)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        path = self.root + self.video_list[idx]
        _idx = idx
        decoder = VideoDecoder(path, seek_mode="approximate")
        total_frames = decoder.metadata.num_frames

        total_sampled = (total_frames + self.stride - 1) // self.stride
        n_full = total_sampled // self.num_frames

        if n_full == 0:
            return None

        clips = []
        timesteps = []
        for clip_idx in range(n_full):
            start = clip_idx * self.num_frames * self.stride
            end = start + self.num_frames * self.stride
            chunk = decoder.get_frames_in_range(start, min(end, total_frames), self.stride).data
            chunk = self._resize_decoded_frames(chunk)

            if chunk.shape[0] < self.num_frames:
                break

            clips.append(chunk[: self.num_frames])
            timesteps.append(start)

        if len(clips) == 0:
            return None

        clips = torch.stack(clips, dim=0)
        timesteps = torch.tensor(timesteps, dtype=torch.long)
        del decoder
        return clips, self.video_list[_idx], timesteps, _idx


def collate_skip_invalid(batch):
    valid_batch = [item for item in batch if item is not None]
    if len(valid_batch) == 0:
        return None
    return default_collate(valid_batch)


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


@torch.no_grad()
def encode_wan_vae_clip(batch, encoder):
    b, k, c, t, h, w = batch.shape
    batch = batch.view(b * k, c, t, h, w)
    latents = torch.stack(encoder.encode(batch))
    latents = latents.view(b, k, *latents.shape[1:])
    return latents


@torch.no_grad()
def decode_wan_vae_clip(latents, encoder, device):
    latents = latents.to(device)
    decoded = torch.stack(encoder.decode(latents)).cpu()
    return decoded


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    # Same location as `vqgan_folder` in conf/experiment/base.yaml.
    default_vae = repo_root / "ckpt" / "wan21" / "wan_2.1_vae.pth"

    parser = argparse.ArgumentParser(description="Extract WAN VAE latents from mp4 videos.")
    parser.add_argument("--root-folder", type=str, required=True, help="Root folder containing source videos.")
    parser.add_argument("--video-list", type=str, default="path_to_mp4.txt", help="Path to txt list (relative to root-folder if not absolute).")
    parser.add_argument("--out-dir", type=str, required=True, help="Output folder for .pt latent files.")
    parser.add_argument("--wan-path", type=str, default=str(default_vae), help="Path to wan_2.1_vae.pth.")

    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--out-h", type=int, default=320)
    parser.add_argument("--out-w", type=int, default=416)

    parser.add_argument("--clip-chunk", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--fps", type=int, default=10)

    parser.add_argument("--dry-run", action="store_true", help="Reconstruct first video to mp4 for sanity check.")
    parser.add_argument("--recon-dir", type=str, default="./dry_run_recon")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    root_folder = args.root_folder
    if not root_folder.endswith(os.sep):
        root_folder = root_folder + os.sep

    video_list_path = Path(args.video_list)
    if not video_list_path.is_absolute():
        video_list_path = Path(root_folder) / video_list_path

    out_dir = args.out_dir
    wan_path = args.wan_path
    recon_dir = args.recon_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        compute_dtype = torch.float32

    is_distributed, rank, world_size, local_rank = init_distributed()
    print(is_distributed, rank, world_size, local_rank)

    os.makedirs(out_dir, exist_ok=True)

    video_list = video_list_path.read_text(encoding="utf-8").splitlines()

    if args.skip_existing:
        pending_video_list = []
        skipped_count = 0
        for rel_video_path in video_list:
            out_rel_path = os.path.splitext(rel_video_path)[0] + ".pt"
            out_path = os.path.join(out_dir, out_rel_path)
            if os.path.exists(out_path):
                skipped_count += 1
            else:
                pending_video_list.append(rel_video_path)
        video_list = pending_video_list
        if rank == 0:
            print(f"SKIP_EXISTING enabled: skipped={skipped_count}, remaining={len(video_list)}")

    if args.dry_run:
        os.makedirs(recon_dir, exist_ok=True)

    if len(video_list) == 0:
        if rank == 0:
            print("No videos to process after filtering. Exiting.")
        if is_distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    encoder = WanVAE(vae_pth=wan_path, dtype=compute_dtype)

    dataset = ClipDataset(
        video_list,
        root_folder,
        num_frames=args.num_frames,
        stride=args.stride,
        out_h=args.out_h,
        out_w=args.out_w,
    )

    sampler = None
    if is_distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False, drop_last=False)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_skip_invalid,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    loader = DataLoader(**loader_kwargs)

    did_recon = False
    with torch.inference_mode():
        for batch in tqdm(loader, disable=(rank != 0)):
            if batch is None:
                print("Warning: all samples in this batch were invalid (e.g. too short videos); skipping batch.")
                continue

            clips, paths, timesteps, indices = batch
            _, k, _, _, _, _ = clips.shape
            latents_chunks = []
            for k0 in range(0, k, args.clip_chunk):
                k1 = min(k0 + args.clip_chunk, k)
                clips_chunk = clips[:, k0:k1].permute(0, 1, 3, 2, 4, 5)
                clips_chunk = clips_chunk.to(device, dtype=compute_dtype, non_blocking=args.pin_memory)
                lat_chunk = encode_wan_vae_clip(clips_chunk, encoder).to(compute_dtype)
                latents_chunks.append(lat_chunk)
                del clips_chunk

            latents = torch.cat(latents_chunks, dim=1)
            latents_cpu = latents.cpu()

            for j in range(len(paths)):
                rel_video_path = paths[j]
                out_rel_path = os.path.splitext(rel_video_path)[0] + ".pt"
                out_path = os.path.join(out_dir, out_rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                to_save = {
                    "video": rel_video_path,
                    "timesteps": timesteps[j],
                    "latents": latents_cpu[j],
                }
                torch.save(to_save, out_path)

                if args.dry_run and (not did_recon):
                    rec = decode_wan_vae_clip(latents_cpu[j], encoder, device)
                    rec = rec.clamp(0, 1).permute(0, 2, 3, 4, 1).reshape(-1, args.out_h, args.out_w, 3)
                    rec_uint8 = (rec * 255.0).to(torch.uint8).numpy()

                    recon_rel_path = os.path.splitext(rel_video_path)[0] + ".mp4"
                    recon_path = os.path.join(recon_dir, recon_rel_path)
                    os.makedirs(os.path.dirname(recon_path), exist_ok=True)
                    imageio.mimwrite(recon_path, rec_uint8, fps=args.fps)
                    did_recon = True

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
