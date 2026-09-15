from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric


def remap_image_torch(image: torch.Tensor) -> torch.Tensor:
    """image should be between -1 and 1, convert it to [0, 1]"""
    image = image.view(-1, 3, image.size(-2), image.size(-1))
    image = (image + 1) / 2
    return image


class PixelMetrics(Metric):
    def __init__(
        self,
        device: torch.device,
        metrics: List[str] = None,
        **kwargs: Dict[str, Union[str, torch.device]],
    ) -> None:

        metric_kwargs = {k: v for k, v in kwargs.items() if k not in ["fid_model"]}
        super().__init__(**metric_kwargs)

        if metrics is None:
            metrics = ["PSNR", "SSIM"]

        self.metrics: List[str] = metrics

        if "MSE" in metrics:
            self.add_state("mse_metrics", [], dist_reduce_fx=None)

        if "PSNR" in metrics:
            self.add_state("psnr_metrics", [], dist_reduce_fx=None)

        if "SSIM" in metrics:
            self.ssim: StructuralSimilarityIndexMeasure = StructuralSimilarityIndexMeasure(data_range=1, reduction="none").to(
                device
            )
            self.add_state("ssim_metrics", [], dist_reduce_fx=None)

    @staticmethod
    def compute_mse(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        return F.mse_loss(predictions, targets, reduction=reduction)

    def compute_psnr(self, predictions: torch.Tensor, targets: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        mse = self.compute_mse(predictions, targets).mean(dim=(1, 2, 3))
        psnr = 10 * torch.log10(data_range / torch.sqrt(mse)).to(mse.device)
        return psnr

    def compute_ssim(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ssim(predictions, targets)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = remap_image_torch(predictions)
        targets = remap_image_torch(targets)

        # Update reconstruction metrics
        if "MSE" in self.metrics:
            batch_mse = self.compute_mse(predictions, targets)
            batch_mse = batch_mse.mean(dim=(1, 2, 3))
            self.mse_metrics += batch_mse.cpu().tolist()

        if "PSNR" in self.metrics:
            batch_psnr = self.compute_psnr(predictions, targets)
            self.psnr_metrics += batch_psnr.cpu().tolist()

        if "SSIM" in self.metrics:
            batch_ssim = self.compute_ssim(predictions, targets)
            self.ssim_metrics += batch_ssim.cpu().tolist()


    def compute(self) -> Dict[str, float]:
        output_metrics: Dict[str, float] = {}

        # Compute reconstruction metrics
        if "MSE" in self.metrics:
            output_metrics["MSE"] = sum(self.mse_metrics) / len(self.mse_metrics)

        if "PSNR" in self.metrics:
            output_metrics["PSNR"] = sum(self.psnr_metrics) / len(self.psnr_metrics)

        if "SSIM" in self.metrics:
            output_metrics["SSIM"] = sum(self.ssim_metrics) / len(self.ssim_metrics)

        return output_metrics

