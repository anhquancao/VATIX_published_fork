# NuScenes Subset As Latent Data (Extract -> Split -> Train)

If you want to treat your NuScenes subset like the "basic" latent workflow used in this repo, the right flow is:

1. Create and activate a Python environment.
2. Build a list of MP4 files from your subset.
3. Extract WAN latents (`.pt`) from those videos.
4. Create `train.txt` and `val.txt` pointing to the latent files.
5. Train with `data=natix_feat`.

## 1) Environment setup

Use Python 3.10+ and run commands from repository root.

```bash
git clone https://github.com/valeoai/VATIX.git
cd VATIX
python3 -m venv vatix_env
source vatix_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Installs all runtime dependencies, excluding PyTorch
pip install -e .
# Installs all runtime dependencies, including PyTorch
# pip install -e ".[torch]"
```

Download the WAN 2.1 VAE required for latent extraction and training:

```bash
hf download llvictorll/Vatix wan21/wan_2.1_vae.pth --repo-type model --local-dir ./ckpt
```

Quick sanity checks:

```bash
python -c "import torch, vatix; print('torch:', torch.__version__)"
python main.py --help
```

## 2) Put your real MP4 data in one place

In this example: use the folder already created in this repository (random video from NuScenes).

```text
./real_videos/
  Nuscenes_random_samples/
    clip_00031.mp4
    ...
  Nuscenes_traj_samples/
    clip_00000.mp4
    clip_00000.npy
    ...
```

## 3) Create a video list file

From repository root:

```bash
cd real_videos
find . -type f -name "*.mp4" | sed 's#^./##' | sort > ../path_to_mp4.txt
cd ..
```

`path_to_mp4.txt` should contain paths relative to `real_videos`.

## 4) Extract latents from the subset

`--video-list` is resolved relative to `--root-folder`. `--stride 1` because the shipped clips are
only 25 frames long.

```bash
python vatix/scripts/extract_vid_emb.py \
  --root-folder ./real_videos \
  --video-list ../path_to_mp4.txt \
  --out-dir ./data_feat_subset \
  --stride 1 \
  --skip-existing
```


## 5) Build train/val split txt files from extracted latents

Create deterministic split files:

```python
from pathlib import Path
import random

root = Path("./data_feat_subset")
out_dir = Path("./splits_subset")
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.pt"))
random.Random(42).shuffle(files)

split_idx = max(1, int(len(files) * 0.9))
train_files = files[:split_idx]
val_files = files[split_idx:]

(out_dir / "train.txt").write_text("\n".join(train_files) + "\n", encoding="utf-8")
(out_dir / "val.txt").write_text("\n".join(val_files) + "\n", encoding="utf-8")

print(f"train={len(train_files)}, val={len(val_files)}")
```

## 6) Train on extracted features (`data=natix_feat`)

Single GPU example:

```bash
python main.py experiment=base \
  data=natix_feat \
  data_folder=./data_feat_subset \
  train_list=./splits_subset/train.txt \
  val_list=./splits_subset/val.txt \
  vit_folder=./ckpt/ \
  writer_log=./runs/ \
  global_bsize=4 \
  num_workers=4 \
  max_iter=3000 \
  virtual_epoch=300 \
  eval_max_iter=20 \
  log_iter=200 \
  save_iter=500 \
  metrics_eval=4
```

DDP example:

```bash
torchrun --nproc_per_node=4 main.py experiment=multi_gpu_ddp \
  data=natix_feat \
  data_folder=./data_feat_subset \
  train_list=./splits_subset/train.txt \
  val_list=./splits_subset/val.txt \
  vit_folder=./ckpt/ \
  writer_log=./runs/ \
  global_bsize=16 \
  num_workers=4 \
  max_iter=3000 \
  virtual_epoch=300 \
  eval_max_iter=20 \
  log_iter=200 \
  save_iter=500 \
  metrics_eval=0
```

## 7) Fine-tune from a previous checkpoint on this subset (optional)

```bash
python main.py experiment=base \
  data=natix_feat \
  data_folder=./data_feat_subset \
  train_list=./splits_subset/train.txt \
  val_list=./splits_subset/val.txt \
  vit_folder=./ckpt/pretrained_model/ \
  writer_log=./runs/pretrained_model/ \
  resume=true \
  fresh_optim_warmup=500 \
  lr=5e-5 \
  global_bsize=4
```

Practical notes for very small subsets:

- Keep `metrics_eval=0` for fast iterations, then enable metrics for final checkpoints.
- Use lower LR (`1e-5` to `5e-5`) when fine-tuning.
- Watch for overfitting quickly on tiny data (train loss down, eval quality not improving).

## 8) Trajectory conditioning

The model can be steered by an ego-trajectory: a sequence of `(x, y)` waypoints in cumulative
meters, in the camera frame (`x` = right, `y` = forward), relative to the clip origin so waypoint 0
is `(0, 0)`. The waypoints are grouped onto latent frames and added to the timestep modulation, so
the generated video follows the requested path.

### Data layout

`data=mp4_traj` expects each `clip.mp4` to have a sibling `clip.npy` holding a
`(trajectory_length, 2)` float32 array. `real_videos/Nuscenes_traj_samples/` ships 30 such clips
built from the nuScenes validation set, spread over seven driving behaviours (see `clips.json`):
`IDLE`, `FORWARD_SLOW`, `FORWARD_MED`, `FORWARD_FAST`, `TURN_LEFT`, `TURN_RIGHT`, `BRAKE`.

```text
real_videos/Nuscenes_traj_samples/
  clip_00205.mp4
  clip_00205.npy
  ...
  clips.json          # scenario label + source nuScenes frames per clip
```

`data=mp4_traj` splits the folder 90/10, so the 30 shipped clips give 27 train / 3 validation.

### Train

```bash
python main.py experiment=base \
  data=mp4_traj \
  data_folder=./real_videos/Nuscenes_traj_samples \
  use_trajectory_cond=true \
  trajectory_length=25 \
  trajectory_cond_prob=0.85 \
  vit_folder=./ckpt/traj/ \
  writer_log=./runs/traj/ \
  global_bsize=2 \
  num_workers=4 \
  max_iter=2000 \
  virtual_epoch=200 \
  metrics_eval=0 \
  save_iter=500
```

This is a demo-scale example sized for the 30 shipped clips. You need to adapt the paths, batch size and
schedule to your own data.

`trajectory_cond_prob` is the probability of *keeping* the trajectory for a given sample. The
remaining fraction is trained unconditionally, which is what makes classifier-free guidance
possible at sampling time. Note `(trajectory_length - 1)` must divide evenly by
`(input_size[1] - 1)`. With the default `input_size: [16, 7, 40, 52]`, 25 waypoints map onto 7
latent frames as `[0], [1-4], [5-8], ..., [21-24]`.

The trajectory embedder is zero-initialised, so at step 0 the model is exactly the unconditional
model. That makes it safe to switch conditioning on when fine-tuning an existing checkpoint.

### Sample

Sampling with a trajectory is a plain-Python script: see
[Minimal Inference (Plain Python)](../README.md#minimal-inference-plain-python) in the README. Its
trajectory-conditioned variant rolls each frame in `real_videos/context_frames/` forward under four
commands (`left, right, straight, static`).
