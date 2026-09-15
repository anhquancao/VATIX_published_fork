# VATIX Technical Guide

This guide contains the operational details moved out of the main README.
For a first overview and the shortest installation path, start with the
[README](../README.md). For a complete small-subset workflow, see the
[full latent-data tutorial](full_tutorial.md).

## Configuration

The default Hydra entry point is `conf/config.yaml`. It selects an experiment
from `conf/experiment/`:

| Experiment | Purpose |
| --- | --- |
| `base` | Single-GPU training and evaluation |
| `multi_gpu_ddp` | Distributed data parallel training |
| `multi_gpu_fsdp` | Fully sharded distributed training |

Override configuration values on the command line. For example:

```bash
bash launch/base.sh base data=natix_feat global_bsize=16 lr=2e-4
```

The launcher accepts the same Hydra overrides as `main.py` and runs from the
repository root automatically.

## Training modes

### Local multi-GPU

Set `NPROC_PER_NODE` to the number of GPUs on the machine:

```bash
NPROC_PER_NODE=4 bash launch/base.sh multi_gpu_ddp
NPROC_PER_NODE=4 bash launch/base.sh multi_gpu_fsdp
```

### Slurm

Use the Slurm launcher for cluster jobs:

```bash
sbatch --export=EXPERIMENT=multi_gpu_ddp launch/slurm.sh
sbatch --nodes=2 --gres=gpu:4 \
  --export=EXPERIMENT=multi_gpu_fsdp launch/slurm.sh
```

Hydra overrides can be passed through `HYDRA_OVERRIDES`:

```bash
sbatch --export=EXPERIMENT=multi_gpu_fsdp,HYDRA_OVERRIDES='max_iter=10000 lr=5e-5' launch/slurm.sh
```

## Data preparation

The project supports two dataset modes:

- `data=natix`: read video data directly from MP4 files.
- `data=natix_feat`: read latent files produced by WAN VAE extraction.

Configure `data_folder`, `train_list`, and `val_list` in the selected Hydra
experiment or as command-line overrides. List files contain paths relative to
the configured data folder.

For NATIX, use paths relative to `--root-folder`. For multi-camera training,
start with FRONT camera paths so camera remapping remains consistent.

## Extract WAN latents

Create a list of videos first. For example, from a raw NATIX dataset folder:

```bash
cd /path/to/natix_raw_dataset
find . -type f -path "*/FRONT_FOLDER/*.mp4" | sed 's#^./##' | sort > path_to_mp4.txt
cd -
```

Extract latents with the packaged script:

```bash
python vatix/scripts/extract_vid_emb.py \
  --root-folder /path/to/natix_raw_dataset \
  --video-list path_to_mp4.txt \
  --out-dir /path/to/Natix_feat \
  --skip-existing
```

Useful options include `--wan-path`, `--num-frames`, `--stride`,
`--num-workers`, `--dry-run`, and `--recon-dir`. Run
`python vatix/scripts/extract_vid_emb.py --help` for the complete list.

Download the WAN 2.1 VAE before latent extraction, training, or inference:

```bash
hf download llvictorll/Vatix wan21/wan_2.1_vae.pth \
  --repo-type model --local-dir ./ckpt
```

This creates `ckpt/wan21/wan_2.1_vae.pth`, the default path used by the
configuration.

## Train on extracted features

After creating `train.txt` and `val.txt`, a minimal single-GPU command is:

```bash
python main.py experiment=base \
  data=natix_feat \
  data_folder=./data_feat_subset \
  train_list=./splits_subset/train.txt \
  val_list=./splits_subset/val.txt \
  vit_folder=./ckpt/ \
  writer_log=./runs/ \
  global_bsize=4 \
  max_iter=3000
```

The [full tutorial](full_tutorial.md) includes deterministic split creation,
an evaluation-focused configuration, a DDP example, and fine-tuning advice
for small subsets.

## Pretrained checkpoints and inference

Download a checkpoint repository with the Hugging Face CLI:

```bash
huggingface-cli download valeoai/VATIX \
  --repo-type model --local-dir ./ckpt/VATIX
```

If it has not already been downloaded, fetch the WAN 2.1 VAE as well:

```bash
hf download llvictorll/Vatix wan21/wan_2.1_vae.pth \
  --repo-type model --local-dir ./ckpt
```

The `vit_folder` directory is expected to contain checkpoint subdirectories:

```text
ckpt/VATIX/
  ckpt/
    10000/
      model_state_dict.pt
      optim_state_dict.pt
      ema_state_dict.pt
      meta.json
```

Resume or evaluate with:

```bash
bash launch/base.sh base test_only=true vit_folder=./ckpt/VATIX
```

For plain-Python generation, compose the Hydra config with
`experiment=base`, set `test_only=true`, point `vit_folder` to the checkpoint,
and instantiate `vatix.trainer.FM`. The generated sample tensor has shape
`(B, C, T, H, W)` and values in `[-1, 1]`; convert it to uint8 frames before
writing an MP4. The notebook in `examples/inference_video_model.ipynb` is a
more interactive starting point.

### Minimal plain-Python inference

The following script generates one unconditional video from a checkpoint. Save
it as a temporary script in the repository root, or adapt it for your own
pipeline.

```python
from pathlib import Path

import imageio
import torch
from hydra import compose, initialize_config_dir

from main import cfg_to_args
from vatix.trainer import FM


def main():
  repo_root = Path(__file__).resolve().parent

  with initialize_config_dir(config_dir=str(repo_root / "conf"), version_base=None):
    cfg = compose(
      config_name="config",
      overrides=[
        "experiment=base",
        "test_only=true",
        "vit_size=giant",
        "vit_folder=./ckpt/VATIX",
        "writer_log=",
        "exp_name=inference",
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
  args.bsize = args.global_bsize

  fm = FM(args)

  with fm.ema_scope():
    sample = fm.generation_helper.generate_samples(
      nb_video=1,
      latent_context=0,
      num_steps=args.step,
      rollout_steps=1,
      alpha=0.0,
    )

  video = sample[0].permute(1, 2, 3, 0).clamp(-1, 1)
  video = ((video + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
  out_path = repo_root / "sample_inference.mp4"
  imageio.mimsave(out_path, video, fps=9)
  print(f"Saved: {out_path}")


if __name__ == "__main__":
  main()
```

### Trajectory-conditioned inference

Use the trajectory checkpoint with `use_trajectory_cond=true` and
`vit_size=giant`. The example below generates one side-by-side video for each
context frame, using shared noise so the four commands are easy to compare.

```python
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from main import cfg_to_args
from vatix.trainer import FM
from vatix.trajectory_utils import command_trajectories


def main():
  repo_root = Path(__file__).resolve().parent
  ctx_dir = repo_root / "real_videos" / "context_frames"

  with initialize_config_dir(config_dir=str(repo_root / "conf"), version_base=None):
    cfg = compose(
      config_name="config",
      overrides=[
        "experiment=base",
        "test_only=true",
        "use_trajectory_cond=true",
        "vit_size=giant",
        "vit_folder=./ckpt/VATIX/traj_v2",
        "writer_log=",
        "exp_name=inference",
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
  args.bsize = args.global_bsize

  fm = FM(args)
  names = ["left", "right", "straight", "static"]
  commands = command_trajectories(args.trajectory_length)
  trajectory = torch.from_numpy(np.stack([commands[name] for name in names])).to(args.device)

  for ctx_path in sorted(ctx_dir.glob("*.png")):
    frame = iio.imread(ctx_path)
    ctx = torch.from_numpy(frame.copy()).permute(2, 0, 1)[None, :, None].float()
    x_ctx = ((ctx / 127.5) - 1.0).repeat(len(names), 1, args.n_frames, 1, 1).to(args.device)

    torch.manual_seed(0)
    with fm.ema_scope():
      sample = fm.generation_helper.generate_samples(
        x_ctx=x_ctx,
        latent_context=1,
        num_steps=50,
        alpha=0.0,
        trajectory=trajectory,
        cfg_w=3.0,
        shared_noise=True,
      )

    video = ((sample.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    video = video.permute(0, 2, 3, 4, 1).cpu().numpy()
    panel = np.concatenate(list(video), axis=2)
    out_path = repo_root / f"command_panel_{ctx_path.stem}.mp4"
    imageio.mimsave(out_path, panel, fps=4)
    print(f"Saved: {out_path}  columns = {names}")


if __name__ == "__main__":
  main()
```

The trajectory example requires the matching trajectory checkpoint and the
`real_videos/context_frames/` images. Keep `vit_size` and
`use_trajectory_cond` consistent with the checkpoint.

### Hugging Face image-to-video example

The published `llvictorll/Vatix` checkpoint can be used with the ready-made
script below. It downloads the `1B_traj` transformer and EMA weights, encodes
`real_videos/context_frames/sample2_canada.png` with the local WAN VAE, and
writes 25 generated frames to `sample2_canada_future.mp4`:

```bash
python examples/inference_huggingface.py
```

The complete snippet is in
`examples/inference_huggingface.py`. It uses the `straight` trajectory; pass
`trajectory=None` to `generate_samples` for unconditional generation.

## Evaluation

Evaluation utilities are available in `vatix/metrics/` and
`vatix/scripts/eval_from_folder.py`. They operate on generated video folders;
see the command help for the exact input and output arguments:

```bash
python vatix/scripts/eval_from_folder.py --help
```

## Dataset and license notes

The NATIX Multi-Camera Driving Dataset is access-controlled and governed by
its upstream [Data RAIL-NC license](https://huggingface.co/datasets/natix-network-org/natix-multi-camera-driving-dataset/blob/main/LICENSE.md).
It is non-commercial and must not be redistributed.

Repository license details are in [licenses/](../licenses/):

- Code: [MIT License](../licenses/LICENSE)
- VATIX checkpoints: [model license](../licenses/LICENSE_MODEL.md)
- Included nuScenes-derived data: [data license](../licenses/LICENSE_DATA.md)
- WAN notices: [NOTICE_WAN2.1.md](../licenses/NOTICE_WAN2.1.md)

Datasets obtained independently are governed by their own source terms.
