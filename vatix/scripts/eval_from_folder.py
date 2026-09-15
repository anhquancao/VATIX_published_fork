"""Evaluate generated videos against a ground-truth folder.

This script computes video, image, and pixel metrics from two folder-based
video datasets loaded through the project dataloader.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import torch
from einops import rearrange
from tqdm import tqdm

from vatix.dataset.dataloader import get_data
from vatix.metrics.inception_metrics import MultiInceptionMetrics
from vatix.metrics.pixel_metrics import PixelMetrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for folder-based evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate generated videos from folders")
    parser.add_argument("--gt-folder", type=str, required=True, help="Ground-truth folder")
    parser.add_argument("--generated-folder", type=str, required=True, help="Generated samples folder")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[320, 416],
        metavar=("H", "W"),
        help="Input image size as height width",
    )
    parser.add_argument("--tot-frames", type=int, default=25, help="Total number of frames to evaluate")
    parser.add_argument("--n-frames", type=int, default=25, help="Rollout frames setting")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--bsize", type=int, default=1, help="Batch size")
    parser.add_argument("--num-video", type=int, default=2000, help="Number of videos to evaluate")
    parser.add_argument("--nb-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--nb-cam", type=int, default=1, help="Number of cameras")
    parser.add_argument("--is-multi-gpus", action="store_true", help="Use multi-GPU dataloader setup")
    parser.add_argument("--compute-pr", action="store_true", help="Enable precision/recall manifold metrics")
    parser.add_argument("--is-master", dest="is_master", action="store_true", help="Master process for logging")
    parser.add_argument("--not-master", dest="is_master", action="store_false", help="Disable master logging")
    parser.set_defaults(is_master=True)
    parser.add_argument("--save-sample", action="store_true", help="Save one generated/real comparison video")
    return parser.parse_args()


def main() -> int:
    """Run folder-based evaluation and print computed metrics."""
    args = parse_args()

    video_metrics = MultiInceptionMetrics(
        device=args.device,
        compute_manifold=args.compute_pr,
        num_inception_chunks=10,
        manifold_k=3,
        model="i3d",
        use_precomputed_feat="",
    )

    images_metrics = MultiInceptionMetrics(
        device=args.device,
        compute_manifold=args.compute_pr,
        num_inception_chunks=10,
        manifold_k=3,
        model="inception",
        use_precomputed_feat="",
    )

    pixel_metrics = PixelMetrics(device=args.device)

    gt_data = get_data(
        data="mp4",
        img_size=args.img_size,
        n_frames=args.tot_frames,
        seed=args.seed,
        data_folder=args.gt_folder,
        bsize=args.bsize,
        num_workers=4,
        is_multi_gpus=args.is_multi_gpus,
        shuffle=False,
    )[0]

    generated_data = get_data(
        data="mp4",
        img_size=args.img_size,
        n_frames=args.tot_frames,
        seed=args.seed,
        data_folder=args.generated_folder,
        bsize=args.bsize,
        num_workers=4,
        is_multi_gpus=args.is_multi_gpus,
        shuffle=False,
    )[0]

    total_videos = args.num_video // (args.nb_gpus * args.nb_cam)
    bar = tqdm(total=total_videos, leave=False, desc="Metrics Evaluation") if args.is_master else None

    with torch.inference_mode():
        for i, (gt_batch, gen_batch) in enumerate(zip(gt_data, generated_data)):
            gen_sample = gen_batch["images"].to(args.device)
            real_sample = gt_batch["images"].to(args.device)

            b, c, t, h, w_total = real_sample.size()
            w = w_total // args.nb_cam

            gen_sample = gen_sample[:, :, : args.tot_frames, :, :]
            gen_sample = rearrange(
                gen_sample,
                "b c t h (w cam) -> (b cam) t c h w",
                b=b,
                c=c,
                t=args.tot_frames,
                h=h,
                w=w,
                cam=args.nb_cam,
            )
            gen_sample = torch.clamp(((gen_sample.float() / 255.0) * 2) - 1, -1, 1).contiguous().float()

            real_sample = rearrange(
                real_sample,
                "b c t h (w cam) -> (b cam) t c h w",
                b=b,
                c=c,
                t=args.tot_frames,
                h=h,
                w=w,
                cam=args.nb_cam,
            )
            real_sample = torch.clamp(((real_sample.float() / 255.0) * 2) - 1, -1, 1).contiguous().float()

            video_metrics.update(gen_sample, image_type="fake")
            video_metrics.update(real_sample, image_type="real")

            images_metrics.update(gen_sample, image_type="fake")
            images_metrics.update(real_sample, image_type="real")

            pixel_metrics.update(gen_sample, real_sample)

            if args.save_sample and i == 0 and args.is_master:
                video = torch.cat([gen_sample[0], real_sample[0]], dim=-1)
                video = video.permute(0, 2, 3, 1).cpu().numpy()
                video = ((video + 1) * 127.5).astype("uint8")
                out_path = Path("./saved_video/generated_video.mp4")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(str(out_path), video, fps=9)

            if bar is not None:
                bar.update(b)

            if video_metrics.count >= total_videos:
                break

    video_m = video_metrics.compute()
    images_m = images_metrics.compute()
    pixel_m = pixel_metrics.compute()

    metrics = {**video_m, **images_m, **pixel_m}
    metrics = {k: round(v, 4) for k, v in metrics.items()}

    if bar is not None:
        bar.close()
        print(metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
