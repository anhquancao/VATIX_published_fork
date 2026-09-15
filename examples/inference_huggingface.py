"""Generate future frames from a conditioning image using the HF checkpoint."""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from huggingface_hub import snapshot_download
from hydra import compose, initialize_config_dir

from main import cfg_to_args
from vatix.trainer import FM
from vatix.trajectory_utils import command_trajectories


REPO_ID = "llvictorll/Vatix"
CHECKPOINT_REVISION = "main"


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_image = repo_root / "real_videos" / "context_frames" / "sample2_canada.png"
    parser = argparse.ArgumentParser(
        description="Generate future frames from a conditioning image using the HF checkpoint."
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=str(default_image),
        help="Path to the conditioning image. Defaults to the sample2_canada.png context frame.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    image_path = Path(args.input_image)
    if not image_path.is_absolute():
        image_path = (repo_root / image_path).resolve()
    model_root = repo_root / "ckpt" / "huggingface_vatix"

    # Downloads about 9.5 GB: the transformer and its EMA weights.
    snapshot_download(
        repo_id=REPO_ID,
        revision=CHECKPOINT_REVISION,
        local_dir=model_root,
        allow_patterns=[
            "1B_traj/ckpt/200000/model_state_dict.pt",
            "1B_traj/ckpt/200000/ema_state_dict.pt",
            "1B_traj/ckpt/200000/meta.json",
        ],
    )
    snapshot_download(
        repo_id=REPO_ID,
        revision=CHECKPOINT_REVISION,
        local_dir=repo_root / "ckpt",
        allow_patterns=["wan21/wan_2.1_vae.pth"],
    )

    with initialize_config_dir(config_dir=str(repo_root / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=base",
                "test_only=true",
                "use_trajectory_cond=true",
                "vit_size=giant",
                "vit_folder=./ckpt/huggingface_vatix/1B_traj",
                "writer_log=",
                "global_bsize=1",
            ],
        )

    args = cfg_to_args(cfg)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.iter = 0
    args.global_epoch = 0
    args.global_rank = 0
    args.is_master = True
    args.is_multi_gpus = False
    args.nb_gpus = 1
    args.num_nodes = 1
    args.bsize = 1

    fm = FM(args)

    frame = iio.imread(image_path)
    frame = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float()
    frame = torch.nn.functional.interpolate(
        frame[None], size=tuple(args.img_size), mode="bilinear", align_corners=False
    )
    # WAN's wrapper accepts [0, 255] pixels. Repeat the context frame over the
    # requested clip length; only its first latent frame is clamped as context.
    x_ctx = frame.unsqueeze(2).repeat(1, 1, args.n_frames, 1, 1).to(args.device)

    trajectory = torch.from_numpy(
        command_trajectories(args.trajectory_length)["straight"]
    ).unsqueeze(0).to(args.device)

    with torch.inference_mode(), fm.ema_scope():
        sample = fm.generation_helper.generate_samples(
            x_ctx=x_ctx,
            latent_context=1,
            num_steps=args.step,
            alpha=0.0,
            trajectory=trajectory,
            cfg_w=3.0,
            shared_noise=True,
        )

    video = ((sample[0].clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    output_path = repo_root / "sample2_canada_future.mp4"
    iio.imwrite(output_path, video, fps=9)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()