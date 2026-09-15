import torch


class EMA():
    def __init__(self, model, decay=0.9995):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.collected = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @staticmethod
    def _normalize_name(name):
        return name.replace("module.", "").replace("_orig_mod.", "")

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)


    @torch.no_grad()
    def store(self, model):
        self.collected = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.collected[name] = p.detach().clone()

    @torch.no_grad()
    def copy_to(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])


    @torch.no_grad()
    def restore(self, model):
        if not self.collected:
            return

        for name, p in model.named_parameters():
            if name in self.collected:
                p.copy_(self.collected[name])

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    @staticmethod
    def _adapt_legacy_time_embed_tensor(name, value, target_shape, model):
        """Map legacy AdaLN tensors (8 chunks) to current 6-chunk layout.

        Legacy order:
            [spatial(2), temporal(2), cross(2), mlp(2)]
        Current order:
            [spatial(2), temporal(2), mlp(2)]
        """
        if not hasattr(model, "hidden_dim"):
            return value

        hidden_dim = int(model.hidden_dim)
        if name.endswith("time_embed.mlp.2.weight"):
            if value.ndim == 2 and value.shape[0] == 8 * hidden_dim and target_shape[0] == 6 * hidden_dim:
                keep = [0, 1, 2, 3, 6, 7]
                return value.view(8, hidden_dim, value.shape[1])[keep].reshape(6 * hidden_dim, value.shape[1])
        elif name.endswith("time_embed.mlp.2.bias"):
            if value.ndim == 1 and value.shape[0] == 8 * hidden_dim and target_shape[0] == 6 * hidden_dim:
                keep = [0, 1, 2, 3, 6, 7]
                return value.view(8, hidden_dim)[keep].reshape(6 * hidden_dim)

        return value

    @torch.no_grad()
    def load_state_dict(self, state, model):
        if not state:
            return

        self.decay = state.get("decay", self.decay)
        device = next(model.parameters()).device

        shadow_by_norm = {self._normalize_name(k): k for k in self.shadow.keys()}

        for name, value in state.get("shadow", {}).items():
            target_name = name if name in self.shadow else shadow_by_norm.get(self._normalize_name(name))
            if target_name is not None:
                target = self.shadow[target_name]
                src = value

                if tuple(src.shape) != tuple(target.shape):
                    src = self._adapt_legacy_time_embed_tensor(target_name, src, target.shape, model)

                if tuple(src.shape) != tuple(target.shape) and src.numel() == target.numel():
                    try:
                        src = src.reshape(target.shape)
                        print(
                            f"[EMA] Reshaped '{target_name}' from {tuple(value.shape)} "
                            f"to {tuple(target.shape)}"
                        )
                    except Exception:
                        pass

                if tuple(src.shape) != tuple(target.shape):
                    # Shape mismatch can happen when architecture changed; keep current EMA tensor.
                    print(
                        f"[EMA] Skip '{target_name}' due to shape mismatch: "
                        f"ckpt={tuple(value.shape)} current={tuple(target.shape)}"
                    )
                    continue

                self.shadow[target_name].copy_(src.to(device))


