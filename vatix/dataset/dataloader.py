import os
import glob
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

from vatix.dataset.dataset import (
    CodeDataset,
    VideoDatasetV2,
    NuScenesDataset,
    RawVideoDataset,
    MP4WithTrajectoryDataset,
)


def _subset_indices(total, fraction, seed=-1):
    if total <= 0 or fraction <= 0:
        return []
    if fraction >= 1:
        return list(range(total))

    target = max(1, int(total * fraction))
    rng = np.random.default_rng(seed if seed >= 0 else 0)
    idx = rng.choice(total, size=target, replace=False)
    return sorted(idx.tolist())


def _subset_sequence(seq, fraction, seed=-1):
    if fraction >= 1:
        return seq
    idx = _subset_indices(len(seq), fraction, seed=seed)
    return [seq[i] for i in idx]


def _subset_dataset(ds, fraction, seed=-1):
    if ds is None or fraction >= 1:
        return ds
    idx = _subset_indices(len(ds), fraction, seed=seed)
    return Subset(ds, idx)


def _keep_front_cam_files(video_files):
    """Keep only FRONT camera videos from a list of paths."""
    kept = []
    for path in video_files:
        norm = os.path.normpath(path)
        parts_upper = [p.upper() for p in norm.split(os.sep)]
        name_upper = os.path.basename(norm).upper()

        is_front = (
            "FRONT_CAM" in name_upper
            or name_upper.startswith("FRONT_")
            or "FRONT_FOLDER" in parts_upper
            or "FRONT_CAM" in parts_upper
        )
        if is_front:
            kept.append(path)

    return kept


def _extract_path_from_list_line(line):
    """Extract the video path from a list line.

    Supports both:
    - plain path lines
    - CSV-like lines: rel_link,start,end,stride
    """
    raw = line.strip()
    if not raw:
        return ""

    # Keep only the first CSV field when metadata is present.
    path = raw.split(",", 1)[0].strip().strip('"').strip("'")
    return path


def _load_paths_from_list(list_path, data_root="", convert_to_pt=False):
    paths = []
    if not list_path:
        return paths

    with open(list_path, "r") as f:
        for line in f:
            path = _extract_path_from_list_line(line)
            if not path:
                continue

            if convert_to_pt:
                path = path.replace("splits_iid/train.txt/", "")
                path = path.replace("splits_iid/val.txt/", "")
                path = path.replace(".mp4", ".pt")

            if not os.path.isabs(path) and data_root:
                path = os.path.join(data_root, path)

            paths.append(path)

    return paths


def get_data(
    data,
    img_size,
    data_folder,
    bsize,
    shuffle=True,
    num_workers=0,
    is_multi_gpus=False,
    seed=-1,
    n_frames=1,
    cameras=None,
    data_fraction=1.0,
    train_list="",
    val_list="",
    trajectory_length=25,
    smooth_trajectory=False,
    horizontal_flip_aug=False,
):
    """ Class to load data """

    if data == "natix":
        transform = v2.Resize(size=(img_size[0], img_size[1]))
        if train_list != "" and val_list != "":
            train_files = _load_paths_from_list(train_list, data_root=data_folder, convert_to_pt=False)
            test_files = _load_paths_from_list(val_list, data_root=data_folder, convert_to_pt=False)

            train_files = _subset_sequence(train_files, data_fraction, seed=seed)

            data_train = VideoDatasetV2(train_files, num_frames=n_frames, transform=transform, cameras=cameras, sample_stride=4, shuffle=True)
            data_test = VideoDatasetV2(test_files, num_frames=n_frames, transform=transform, cameras=cameras, sample_stride=4, shuffle=False)
        
        elif data_folder.endswith(".txt"):

            video_files = _load_paths_from_list(data_folder, data_root="", convert_to_pt=False)
            nb_vid = len(video_files)
            split_idx = int(nb_vid * 0.95)
            train_files = video_files[:split_idx]
            test_files = video_files[split_idx:]

            train_files = _subset_sequence(train_files, data_fraction, seed=seed)

            data_train = VideoDatasetV2(train_files, num_frames=n_frames, transform=transform, cameras=cameras, sample_stride=4, shuffle=True)
            data_test = VideoDatasetV2(test_files, num_frames=n_frames, transform=transform, cameras=cameras, sample_stride=4, shuffle=False)
        
        else:
            video_files = glob.glob(os.path.join(data_folder, "**", "*.mp4"), recursive=True)
            video_files = _keep_front_cam_files(video_files)

            nb_vid = len(video_files)
            split_idx = int(nb_vid * 0.95)
            train_files = video_files[:split_idx]
            test_files = video_files[split_idx:]
            train_files = _subset_sequence(train_files, data_fraction, seed=seed)

            data_train = VideoDatasetV2(train_files, num_frames=n_frames, transform=transform, cameras=cameras, sample_stride=4, shuffle=True)
            data_test = VideoDatasetV2(test_files, num_frames=n_frames, transform=transform, cameras=cameras, sample_stride=4, shuffle=False) # always the first frames for test to avoid randomness in evaluation

    elif data == "natix_feat": # Not video but already extracted features.
        if train_list != "" and val_list != "":
            train_files = _load_paths_from_list(train_list, data_root=data_folder, convert_to_pt=True)
            test_files = _load_paths_from_list(val_list, data_root=data_folder, convert_to_pt=True)
    
        else:
            if isinstance(data_folder, str) and data_folder.endswith(".txt") and os.path.isfile(data_folder):
                video_files = _load_paths_from_list(data_folder, data_root=data_folder, convert_to_pt=True)
            else:
                video_files = glob.glob(os.path.join(data_folder, "*.pt"), recursive=True)

            nb_vid = len(video_files)
            split_idx = int(nb_vid * 0.95)
            train_files = video_files[:split_idx]
            test_files = video_files[split_idx:]
            train_files = _subset_sequence(train_files, data_fraction, seed=seed)

        nb_latents_frame = 1 + (n_frames - 1) // 4

        data_train = CodeDataset(train_files, nb_latents_frame=nb_latents_frame, samples_per_file=None, cameras=cameras)
        data_test = CodeDataset(test_files, nb_latents_frame=nb_latents_frame, samples_per_file=None, cameras=cameras)

    elif data == "nuscenes":
        transform = v2.Resize(size=(img_size[0], img_size[1]))

        data_train = NuScenesDataset(
            data_root=data_folder,
            num_frames=n_frames,
            transform=transform,
            sample_stride=1,
            split="train",
            version="v1.0-trainval",
            cameras=cameras,
            target_fps=10.0,
        )
        data_test = NuScenesDataset(
            data_root=data_folder,
            num_frames=n_frames,
            transform=transform,
            sample_stride=1,
            split="val",
            version="v1.0-trainval",
            cameras=cameras,
            target_fps=10.0,
        )

        print("NuScenes windows loaded:", "train=", len(data_train), "val=", len(data_test))

    elif data == "mp4":
        transform = v2.Resize(size=(img_size[0], img_size[1]))
        # Recursive: `real_videos/` keeps its clips in sub-folders.
        video_files = sorted(glob.glob(os.path.join(data_folder, "**", "*.mp4"), recursive=True))
        nb_vid = len(video_files)
        print("MP4 total number of video found:", nb_vid)
        if nb_vid < 2:
            raise ValueError(f"mp4 needs at least 2 clips under {data_folder} to build a train and a test split, found {nb_vid}")

        split_idx = min(max(1, int(nb_vid * 0.95)), nb_vid - 1)
        train_files = video_files[:split_idx]
        test_files = video_files[split_idx:]
        train_files = _subset_sequence(train_files, data_fraction, seed=seed)

        data_train = RawVideoDataset(
            train_files,
            num_frames=n_frames,
            transform=transform,
            sample_stride=1,
            shuffle=shuffle,
        )
        data_test = RawVideoDataset(
            test_files,
            num_frames=n_frames,
            transform=transform,
            sample_stride=1,
            shuffle=shuffle,
        )

        data_train = _subset_dataset(data_train, data_fraction, seed=seed)

    elif data == "mp4_traj":
        transform = v2.Resize(size=(img_size[0], img_size[1]))
        video_files = sorted(glob.glob(os.path.join(data_folder, "*.mp4")))

        nb_vid = len(video_files)
        print("MP4+trajectory total number of video found:", nb_vid)

        if nb_vid < 2:
            raise ValueError(
                f"mp4_traj needs at least 2 clips in {data_folder} to build both a train and a "
                f"test split, found {nb_vid}"
            )

        # nb_vid >= 2 here, so both splits get at least one clip.
        split_idx = min(max(1, int(nb_vid * 0.9)), nb_vid - 1)
        train_files = video_files[:split_idx]
        test_files = video_files[split_idx:]
        train_files = _subset_sequence(train_files, data_fraction, seed=seed)

        data_train = MP4WithTrajectoryDataset(
            train_files,
            num_frames=n_frames,
            transform=transform,
            sample_stride=1,
            trajectory_length=trajectory_length,
            smooth_trajectory=smooth_trajectory,
            horizontal_flip_aug=horizontal_flip_aug,
        )
        # Smoothing in val too, so train and val see the same trajectory statistics; no flip.
        data_test = MP4WithTrajectoryDataset(
            test_files,
            num_frames=n_frames,
            transform=transform,
            sample_stride=1,
            trajectory_length=trajectory_length,
            smooth_trajectory=smooth_trajectory,
        )

    else:
        data_train = None
        data_test = None

    

    train_sampler = DistributedSampler(data_train, shuffle=True, seed=seed) if is_multi_gpus else None

    train_loader = DataLoader(data_train, batch_size=bsize,
                              shuffle=False if is_multi_gpus else shuffle,
                              num_workers=num_workers, pin_memory=False,
                              drop_last=True, sampler=train_sampler)
    
    if data_test is None:
        return train_loader, None
    
    test_sampler = DistributedSampler(data_test, shuffle=False, seed=seed) if is_multi_gpus else None
    test_loader = DataLoader(data_test, batch_size=bsize,
                             shuffle=False,
                             num_workers=num_workers, pin_memory=False,
                             drop_last=True, sampler=test_sampler)

    return train_loader, test_loader

