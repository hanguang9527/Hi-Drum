"""
Compatibility layer for the old mlp_branch module path.

Why:
- 第二阶段把 multi-task heads 物理拆到 `multi_task_heads.py`
- 但当前训练脚本、旧 checkpoint 和外部导入路径仍可能依赖 `mlp_branch.py`
- 所以这里先保留旧模块名，只做稳定转发，避免一次重构破坏现有入口
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
