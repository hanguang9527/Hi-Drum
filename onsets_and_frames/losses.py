"""
losses module

Function:
- Hold multi-task loss definitions used by the transcription model.
- Keep training policy details out of the top-level module assembly path.

Key args:
- onset_logits / frame_logits: logits before sigmoid for BCE-with-logits
- velocity_pred: velocity prediction after sigmoid
- onset_label / frame_label / velocity_label: supervision tensors
"""

import torch
import torch.nn.functional as F


def compute_velocity_loss(velocity_pred, velocity_label, onset_label):
    """
    Weighted velocity regression loss evaluated only on onset positions.
    """
    denominator = onset_label.sum()
    if denominator.item() == 0:
        return torch.tensor(0.0, device=velocity_pred.device, requires_grad=True)
    return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


DEFAULT_LOSS_WEIGHTS = (2, 1, 1)


def compute_multi_task_losses(
    onset_logits,
    frame_logits,
    velocity_pred,
    onset_label,
    frame_label,
    velocity_label,
    onset_weight=DEFAULT_LOSS_WEIGHTS[0],
    frame_weight=DEFAULT_LOSS_WEIGHTS[1],
    velocity_weight=DEFAULT_LOSS_WEIGHTS[2],
):
    """
    Aggregate the current project's multi-task loss terms.

    Default weights follow the historical 2:1:1 ratio (onset:frame:velocity).
    """
    losses = {
        "loss/onset": F.binary_cross_entropy_with_logits(onset_logits, onset_label),
        "loss/frame": F.binary_cross_entropy_with_logits(frame_logits, frame_label),
        # Why: velocity is only meaningful on note onsets, so its task-specific loss stays separate
        # from the two BCE heads and is composed here instead of being hidden inside model assembly.
        "loss/velocity": compute_velocity_loss(velocity_pred, velocity_label, onset_label),
    }
    losses["loss/total"] = (
        onset_weight * losses["loss/onset"]
        + frame_weight * losses["loss/frame"]
        + velocity_weight * losses["loss/velocity"]
    )
    return losses
