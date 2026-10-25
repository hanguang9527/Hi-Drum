"""
Compatibility layer for the old mlp_branch module path.

Why:
- Multi-task heads now live in `multi_task_heads.py`
- Training scripts / old checkpoints may still import `mlp_branch.py`
- Keep this module as a stable re-export shim
"""

from .multi_task_heads import (
    BranchNetwork,
    MLPBranchFactory,
    MultiTaskHeads,
    PostDecoderProcessor,
)

__all__ = [
    "BranchNetwork",
    "MLPBranchFactory",
    "MultiTaskHeads",
    "PostDecoderProcessor",
]
