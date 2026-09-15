"""Per-frame ego-trajectory conditioning for the latent video transformer.

Waypoints are ``(x, y)`` in cumulative meters, camera frame (x = right, y = forward), relative to
the clip origin (waypoint 0 is ``(0, 0)``). Each waypoint becomes a feature vector, each latent
frame's waypoints are combined into one conditioning vector, and the transformer adds it to the
timestep embedding.
"""

import math

import torch
from torch import nn


# Calibrated on real driving data: standstill jitter (~0.03 m/step) gets w ~ 0.2, reversing keeps
# most of its heading signal.
_TANH_SCALE = 0.15   # heading confidence weight: w = tanh(|d| / scale)
_D0 = 0.15           # normalization floor for u/v/curvature


def directional_waypoint_features(traj):
    """Direction-dominant per-waypoint features: (B, L, 2) meters -> (B, L, 200).

    Expects smoothed positions (``smooth_trajectory``), since heading and curvature come from
    consecutive displacements. Channel layout per waypoint, 200 total:
      - x:            64 odd features  sin(4*x*f_i)
      - y:            64 features      cat(cos, sin)(y*f_i)
      - heading:      32 odd features  w * sin(10*theta'*f_i),  theta' = atan2(dx, |dy|)
      - longitudinal: 16 odd features  sin(3*v*f_i),            v = dy / (|d| + d0)
      - curvature:    16 odd features  sin(5*kappa*f_i),        kappa = normalized cross product
      - raw:           8 features      [x/2, y/10, u, v] repeated 2x
    """
    traj = traj.float()
    x, y = traj[..., 0], traj[..., 1]
    dev = traj.device

    # Log-spaced frequencies, as in the timestep embedding.
    f64 = torch.exp(
        -math.log(10000.0) * torch.arange(64, device=dev, dtype=torch.float32) / 64
    )

    # x: sin-only (sign-carrying); x4 moves realistic |x| into the responsive band.
    ex = torch.sin(4.0 * x[..., None] * f64)                            # (B,L,64)

    # y: magnitude channel, cos+sin over 32 frequencies.
    ay = y[..., None] * f64[::2]                                        # (B,L,32)
    ey = torch.cat((torch.cos(ay), torch.sin(ay)), dim=-1)              # (B,L,64)

    # Per-step displacement; d[0] = 0 since waypoint 0 is the origin.
    d = torch.diff(traj, dim=1, prepend=traj[:, :1])                    # (B,L,2)
    d_norm = d.norm(dim=-1)                                             # (B,L)

    # Steering angle only: |dy| folds reversing onto theta' ~ 0 (drive direction lives in v below).
    # w down-weights tiny jitter steps whose angle is meaningless.
    theta_steer = torch.atan2(d[..., 0], d[..., 1].abs())               # (B,L)
    w = torch.tanh(d_norm / _TANH_SCALE)                                # (B,L)
    eth = w[..., None] * torch.sin(10.0 * theta_steer[..., None] * f64[:32])   # (B,L,32)

    # Longitudinal: ~+1 forward, ~-1 reversing, ~0 stopped.
    v = d[..., 1] / (d_norm + _D0)                                      # (B,L)
    ev = torch.sin(3.0 * v[..., None] * f64[:16])                       # (B,L,16)

    # Curvature: normalized cross product of consecutive displacements (signed, bounded, no wrap).
    d_prev = torch.cat([d[:, :1], d[:, :-1]], dim=1)                    # (B,L,2) one-step lag
    d_prev_norm = d_prev.norm(dim=-1)                                   # (B,L)
    cross = d_prev[..., 0] * d[..., 1] - d_prev[..., 1] * d[..., 0]     # (B,L)
    kappa = cross / ((d_prev_norm + _D0) * (d_norm + _D0))              # (B,L)
    ek = torch.sin(5.0 * kappa[..., None] * f64[:16])                   # (B,L,16)

    # Raw normalized coords: a linear path for every sign.
    u = d[..., 0] / (d_norm + _D0)                                      # (B,L)
    raw = torch.stack([x / 2.0, y / 10.0, u, v], dim=-1)
    raw = raw.repeat_interleave(2, dim=-1)                              # (B,L,8)

    return torch.cat([ex, ey, eth, ev, ek, raw], dim=-1)                # (B,L,200)


# Feature count produced by directional_waypoint_features for one waypoint.
FEATURES_PER_WAYPOINT = 200


def _build_waypoint_group_index(trajectory_length, latent_length):
    """Map waypoints onto latent frames 1..latent_length-1, following the WAN causal-VAE layout.

    E.g. 25 waypoints -> 7 frames as [0], [1-4], [5-8], ..., [21-24]. Frame 0 is the origin
    waypoint and is skipped. Returns a (latent_length - 1, stride) LongTensor: row k holds the
    waypoint indices of latent frame k+1.
    """
    if not 2 <= latent_length <= trajectory_length:
        raise ValueError(
            f"latent_length must be in [2, trajectory_length={trajectory_length}], got {latent_length}"
        )
    if (trajectory_length - 1) % (latent_length - 1) != 0:
        raise ValueError(
            f"Cannot map {trajectory_length} waypoints onto {latent_length} latent frames with the "
            f"WAN causal layout: (trajectory_length - 1) must be divisible by (latent_length - 1)."
        )
    stride = (trajectory_length - 1) // (latent_length - 1)
    # Row k holds waypoints [1 + k*stride, 1 + (k+1)*stride): the ones that produced frame k+1.
    return torch.arange(1, trajectory_length, dtype=torch.long).view(latent_length - 1, stride)


class PerFrameTrajectoryEmbedder(nn.Module):
    """One conditioning vector per latent frame, (B, latent_length, out_dim).

    Each frame's waypoint features are concatenated in order (order encodes direction) and mapped
    by a shared MLP. Latent frame 0 only holds the origin and gets a zero row.
    """

    def __init__(self, trajectory_length, hidden_dim, latent_length, out_dim=None):
        super().__init__()

        self.trajectory_length = trajectory_length
        self.latent_length = latent_length
        self.out_dim = hidden_dim if out_dim is None else out_dim

        # Fixed (latent_length - 1, group_size) waypoint->frame gather indices, origin frame excluded.
        group_index = _build_waypoint_group_index(trajectory_length, latent_length)
        self.group_size = group_index.shape[1]
        self.register_buffer("group_index", group_index, persistent=False)

        # Shared MLP over one frame's concatenated waypoint features.
        feat_dim = self.group_size * FEATURES_PER_WAYPOINT
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, self.out_dim),
        )

    def forward(self, trajectory):
        if trajectory is None or trajectory.shape[1:] != (self.trajectory_length, 2):
            got = None if trajectory is None else tuple(trajectory.shape)
            raise ValueError(f"trajectory_cond must have shape (B, {self.trajectory_length}, 2), got {got}")

        emb = directional_waypoint_features(trajectory)                       # (B, L, 200)

        grouped = emb[:, self.group_index].flatten(2)                         # (B, T-1, group_size * 200)
        frames = self.mlp(grouped)                                            # (B, T-1, out_dim)

        # Frame 0 holds only the origin waypoint -> zero modulation.
        frame0 = frames.new_zeros(frames.shape[0], 1, self.out_dim)
        return torch.cat([frame0, frames], dim=1)
