from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class GenerationHelper:
    """Sampling utilities used to generate videos from the trained model."""

    def __init__(self, trainer):
        """Store a reference to the trainer.

        Args:
            trainer: Trainer instance exposing model, autoencoder and runtime args.
        """
        self.fm = trainer

    @torch.no_grad()
    def generate_samples(
        self,
        x_ctx=None,
        nb_video=1,
        latent_context=1,
        num_steps=50,
        alpha=0.5,
        scheduler_mode="cosine",
        rollout_steps=1,
        trajectory=None,
        cfg_w=1.0,
        shared_noise=False,
    ):
        """Generate videos by iterative latent denoising and optional rollouts.

        Args:
            x_ctx (torch.Tensor | None): Optional context video in pixel space, i.e., the first cond frame
            nb_video (int): Number of videos to generate if ``x_ctx`` is None.
            latent_context (int): Number of latent frames clamped as context.
            num_steps (int): Number of denoising integration steps.
            alpha (float): Per-frame temporal offset factor for integration times.
            scheduler_mode (str): Time schedule in {"cosine", "square", "linear"}.
            rollout_steps (int): Number of autoregressive chunks to chain.
            trajectory (torch.Tensor | None): Ego waypoints `(B, trajectory_length, 2)` in
                cumulative meters, camera frame (x=right, y=forward), origin-relative.
            cfg_w (float): Classifier-free guidance weight. `<= 1` disables guidance, `> 1`
                guides toward the trajectory.
            shared_noise (bool): Draw one noise tensor and reuse it for the whole batch, so a
                batch that differs only in its trajectory isolates that difference.

        Returns:
            torch.Tensor: Generated video tensor in shape ``(B, C, T, H, W)``.

        Notes:
            - If ``x_ctx`` is provided, its batch size overrides ``nb_video``.
            - A trajectory-conditioned model with ``trajectory`` None samples unconditionally
              (zeros trajectory, all-False keep mask), as under training-time dropout.
            - Under DDP, ``self.fm.vit.module`` is used.
            - Under FSDP, keep ``self.fm.vit`` wrapper for forward.
            - Subsequent rollouts reuse the tail of the previous chunk as context.
        """

        if isinstance(self.fm.vit, DDP):
            model = self.fm.vit.module
        else:
            model = self.fm.vit
        model.eval()
        c, t_dim, h, w = self.fm.args.input_size

        all_chunks = []
        if x_ctx is not None:
            nb_video = x_ctx.shape[0]

        device = self.fm.args.device
        use_trajectory_cond = getattr(self.fm.args, "use_trajectory_cond", False)
        drop_mask = torch.zeros(nb_video, dtype=torch.bool, device=device)  # all-False keep mask
        keep_mask = None                                                    # None: keep the trajectory
        if trajectory is not None:
            if not use_trajectory_cond:
                raise ValueError(
                    "a trajectory was passed but trajectory conditioning is disabled for this model "
                    "(use_trajectory_cond=false)"
                )
            if rollout_steps > 1:
                raise ValueError(
                    "rollout_steps > 1 is not supported with a trajectory: waypoints are relative to "
                    "the clip origin, so every chunk after the first would wrongly reuse the same "
                    "origin-relative trajectory"
                )
        elif use_trajectory_cond:
            # Unconditional: zeros trajectory dropped through the all-False keep mask.
            trajectory = torch.zeros(nb_video, getattr(self.fm.args, "trajectory_length", 25), 2, device=device)
            keep_mask = drop_mask
        use_cfg = cfg_w > 1 and trajectory is not None and keep_mask is None

        if x_ctx is not None and latent_context > 0:
            with self.fm.autocast:
                c_ctx = self.fm.ae.encode(x_ctx.clone())[:, :, :latent_context]
        else:
            c_ctx = None
        nb_frames_in_context = max(1 + (4 * (latent_context - 1)), 0)
        for rollout in range(rollout_steps):
            z = torch.randn(1 if shared_noise else nb_video, c, t_dim, h, w, device=device)
            if shared_noise:
                # clone(): expand aliases one buffer and the context write below is in place.
                z = z.expand(nb_video, -1, -1, -1, -1).clone()

            if c_ctx is not None and latent_context > 0:
                z[:, :, :latent_context] = c_ctx

            # Integrate from noisy latents to clean latents according to the scheduler.
            for i in tqdm(range(num_steps), leave=False):
                if scheduler_mode == "cosine":
                    progress = 1 - torch.cos(torch.tensor(i / num_steps) * torch.pi / 2)
                    progress_next = 1 - torch.cos(torch.tensor((i + 1) / num_steps) * torch.pi / 2)
                elif scheduler_mode == "square":
                    progress = (i / num_steps) ** 0.5
                    progress_next = ((i + 1) / num_steps) ** 0.5
                else:
                    progress = (i / num_steps)
                    progress_next = ((i + 1) / num_steps)

                offsets = 1 + (torch.linspace(1, 0, t_dim, device=z.device)[None, :] * alpha)
                # clone() so the expanded view can be written in place below.
                t_curr = torch.clamp(progress * offsets, 0, 1).expand(nb_video, t_dim).clone()
                t_next = torch.clamp(progress_next * offsets, 0, 1)
                if latent_context > 0:
                    t_curr[:, :latent_context] = 1
                    t_next[:, :latent_context] = 1

                with self.fm.autocast:
                    pred = model(x=z, ada_cond=t_curr, trajectory_cond=trajectory, trajectory_keep_mask=keep_mask)
                    if use_cfg:
                        # Classifier-free guidance: the unconditional pass drops the trajectory.
                        pred_uncond = model(x=z, ada_cond=t_curr, trajectory_cond=trajectory, trajectory_keep_mask=drop_mask)
                        pred = pred_uncond + cfg_w * (pred - pred_uncond)

                if self.fm.args.pred_mode == "x":
                    v = (pred - z) / torch.clamp(1 - t_curr.view(nb_video, 1, t_dim, 1, 1), min=0.05)
                elif self.fm.args.pred_mode == "e":
                    v = (z - pred) / torch.clamp(1 - t_curr.view(nb_video, 1, t_dim, 1, 1), min=0.05)
                else:
                    v = pred

                # Keep context frames fixed throughout sampling.
                if latent_context > 0:
                    v[:, :, :latent_context] = 0.0

                z = z + (t_next - t_curr).view(nb_video, 1, t_dim, 1, 1) * v

            with self.fm.autocast:
                video_chunk = self.fm.ae.decode(z)

            if latent_context > 0 and rollout == 0:
                _video_chunk = video_chunk.clone()
            else:
                _video_chunk = video_chunk[:, :, nb_frames_in_context:].clone()

            all_chunks.append(_video_chunk)
            latent_context = max(latent_context, 1)

            # Re-encode the generated tail so the next rollout can continue temporally.
            if latent_context > 0 and rollout < rollout_steps - 1:
                _x_ctx = video_chunk[:, :, -nb_frames_in_context:]
                _x_ctx = (_x_ctx + 1) / 2.0
                _x_ctx = torch.clamp(_x_ctx, 0.0, 1.0)
                x_ctx_uint8 = (_x_ctx * 255)

                with self.fm.autocast:
                    c_ctx = self.fm.ae.encode(x_ctx_uint8.clone())

        video = torch.cat(all_chunks, dim=2)
        return video.float()

