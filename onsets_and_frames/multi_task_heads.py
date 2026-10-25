"""
multi_task_heads module

Function:
- Hold all implementations related to multi-task heads.
- Keep the public `MultiTaskHeads` concept aligned with the project's
  encoder -> adapter -> decoder -> multi-task heads decomposition.

Key args:
- decoder_output_dim: feature dimension received from the decoder
- hidden_dim: shared hidden dimension used inside each task head
- output_dim: task output dimension, usually the number of drum classes / bins
- head_mode: choose between fully independent heads and the paper-style
  hierarchical onset-conditioned heads
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def head_mode_to_logdir_tag(head_mode: str) -> str:
    """Map head_mode to the experiment directory token."""
    if head_mode == PostDecoderProcessor.HEAD_MODE_PAPER_HIERARCHICAL:
        return "HeadsPaper"
    return "HeadsInd"


class PostDecoderProcessor(nn.Module):
    """
    Compatibility base for the original post-decoder processor naming.
    """

    HEAD_MODE_INDEPENDENT = "independent"
    HEAD_MODE_PAPER_HIERARCHICAL = "paper_hierarchical"
    SUPPORTED_HEAD_MODES = {
        HEAD_MODE_INDEPENDENT,
        HEAD_MODE_PAPER_HIERARCHICAL,
    }

    def __init__(
        self,
        decoder_output_dim: int = 896,
        hidden_dim: int = 512,
        output_dim: int = 88,
        dropout: float = 0.25,
        head_mode: str = HEAD_MODE_INDEPENDENT,
    ):
        super(PostDecoderProcessor, self).__init__()

        if head_mode not in self.SUPPORTED_HEAD_MODES:
            raise ValueError(
                f"Unsupported head_mode '{head_mode}'. "
                f"Expected one of {sorted(self.SUPPORTED_HEAD_MODES)}."
            )

        self.decoder_output_dim = decoder_output_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.head_mode = head_mode
        self.use_global_conv = head_mode == self.HEAD_MODE_PAPER_HIERARCHICAL

        if self.use_global_conv:
            self.global_conv = nn.Sequential(
                nn.Conv1d(decoder_output_dim, decoder_output_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(decoder_output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.global_conv = None

        frame_input_dim = decoder_output_dim
        velocity_input_dim = decoder_output_dim
        if self.head_mode == self.HEAD_MODE_PAPER_HIERARCHICAL:
            # Why: 论文设定里 onset branch 要辅助 frame/velocity，
            # 这里通过拼接 onset 概率显式把这种层级关系落到 heads 结构里。
            frame_input_dim += output_dim
            velocity_input_dim += output_dim

        self.onset_branch = BranchNetwork(
            decoder_output_dim, hidden_dim, output_dim, dropout, use_conv=False
        )
        self.frame_branch = BranchNetwork(
            frame_input_dim, hidden_dim, output_dim, dropout, use_conv=False
        )
        self.velocity_branch = BranchNetwork(
            velocity_input_dim, hidden_dim, output_dim, dropout, use_conv=False
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, decoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = decoder_output

        if self.use_global_conv and self.global_conv is not None:
            x = x.transpose(1, 2)
            x = self.global_conv(x)
            x = x.transpose(1, 2)

        onset_pred = self.onset_branch(x)

        frame_input = x
        velocity_input = x
        if self.head_mode == self.HEAD_MODE_PAPER_HIERARCHICAL:
            # Why: 使用 onset 概率作为下游条件信号，能在保留端到端训练的同时，
            # 明确区分“论文层级绑定”与“完全独立三头”两种实验模式。
            onset_condition = torch.sigmoid(onset_pred)
            frame_input = torch.cat([frame_input, onset_condition], dim=-1)
            velocity_input = torch.cat([velocity_input, onset_condition], dim=-1)

        frame_pred = self.frame_branch(frame_input)
        velocity_pred = self.velocity_branch(velocity_input)

        return onset_pred, frame_pred, velocity_pred


class MultiTaskHeads(PostDecoderProcessor):
    """
    Project-specific name for the multi-task output heads.
    """


class BranchNetwork(nn.Module):
    """
    Single task head: optional temporal conv + MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.25,
        use_conv: bool = True,
    ):
        super(BranchNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_conv = use_conv

        if self.use_conv:
            self.branch_conv = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.branch_conv = None

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if self.use_conv and self.branch_conv is not None:
            x_conv = x.transpose(1, 2)
            x_conv = self.branch_conv(x_conv)
            x = x_conv.transpose(1, 2)

        x_reshaped = x.reshape(-1, x.shape[-1])
        output = self.mlp(x_reshaped)
        output = output.view(batch_size, seq_len, -1)
        return output

    def get_parameter_count(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        layer_params = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                params = sum(p.numel() for p in module.parameters())
                layer_params[name] = params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layer_parameters": layer_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }

    def __repr__(self):
        return (
            f"MLPBranch(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim})"
        )


class MLPBranchFactory:
    """
    Compatibility factory kept for the current training / evaluation code path.
    """

    @staticmethod
    def create_onset_branch(input_dim: int = 896, output_dim: int = 88) -> BranchNetwork:
        return BranchNetwork(
            input_dim=input_dim, hidden_dim=512, output_dim=output_dim, dropout=0.25, use_conv=False
        )

    @staticmethod
    def create_frame_branch(input_dim: int = 896, output_dim: int = 88) -> BranchNetwork:
        return BranchNetwork(
            input_dim=input_dim, hidden_dim=512, output_dim=output_dim, dropout=0.25, use_conv=False
        )

    @staticmethod
    def create_combined_branch(input_dim: int = 176, output_dim: int = 88) -> BranchNetwork:
        return BranchNetwork(
            input_dim=input_dim, hidden_dim=512, output_dim=output_dim, dropout=0.25, use_conv=False
        )

    @staticmethod
    def create_velocity_branch(input_dim: int = 896, output_dim: int = 88) -> BranchNetwork:
        return BranchNetwork(
            input_dim=input_dim, hidden_dim=512, output_dim=output_dim, dropout=0.25, use_conv=False
        )

    @staticmethod
    def create_post_decoder_processor(
        decoder_output_dim: int = 896,
        output_dim: int = 88,
        head_mode: str = PostDecoderProcessor.HEAD_MODE_INDEPENDENT,
    ) -> PostDecoderProcessor:
        return MultiTaskHeads(
            decoder_output_dim=decoder_output_dim,
            hidden_dim=512,
            output_dim=output_dim,
            dropout=0.25,
            head_mode=head_mode,
        )
