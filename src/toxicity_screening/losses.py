from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weights: torch.Tensor | None = None,
    task_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked BCE exactly excluding missing endpoint labels from the denominator."""
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets, and mask must have identical shapes")
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=positive_weights,
    )
    if task_weights is not None:
        if task_weights.ndim != 1 or task_weights.numel() != logits.shape[-1]:
            raise ValueError("task_weights must have one value per endpoint")
        loss = loss * task_weights.view(1, -1)
    weighted_mask = mask.to(loss.dtype)
    denominator = weighted_mask.sum()
    if denominator <= 0:
        raise ValueError("Masked loss received a batch with no observed labels")
    return (loss * weighted_mask).sum() / denominator


def positive_class_weights(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positives = (labels * mask).sum(dim=0)
    negatives = ((1.0 - labels) * mask).sum(dim=0)
    return negatives / positives.clamp_min(1.0)
