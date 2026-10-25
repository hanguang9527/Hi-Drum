"""Training checkpoint save/load helpers for interruptible, reconfigurable resume."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

# Weights shape / architecture — changing these invalidates a weight load.
STRUCTURE_KEYS = (
    'encoder_model',
    'decoder_model',
    'decoder_backend',
    'encoder_layer_selection',
    'decoder_layer_selection',
    'enable_branch1',
    'enable_branch2',
    'enable_branch3',
    'head_mode',
    'model_complexity',
)

# Parallelism / effective batch — mismatch → rebuild Adam.
ADAM_SENSITIVE_KEYS = (
    'world_size',
    'batch_size',
    'matched_samples_per_batch',
    'unmatched_samples_per_batch',
    'slakh_samples_per_batch',
)

# LR schedule — mismatch → rebuild StepLR from the new config at current step.
SCHEDULER_SENSITIVE_KEYS = (
    'learning_rate',
    'learning_rate_decay_steps',
    'learning_rate_decay_rate',
)

META_FILENAME = 'checkpoint-meta.json'
MODEL_LATEST = 'model-latest.pt'
OPTIMIZER_LATEST = 'last-optimizer-state.pt'
SCHEDULER_LATEST = 'last-scheduler-state.pt'
MODEL_BEST_NOTE = 'model-best-note.pt'


def atomic_torch_save(obj: Any, path: str) -> None:
    """Write via temp file + rename so a crash mid-save cannot leave a half file."""
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    tmp_path = f'{path}.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def atomic_json_save(obj: Mapping[str, Any], path: str) -> None:
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def build_train_meta(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Snapshot of knobs used to decide what can be restored on resume."""
    meta = {key: config.get(key) for key in STRUCTURE_KEYS}
    meta.update({key: config.get(key) for key in ADAM_SENSITIVE_KEYS})
    meta.update({key: config.get(key) for key in SCHEDULER_SENSITIVE_KEYS})
    meta['encoder_freeze'] = config.get('encoder_freeze')
    meta['decoder_freeze_model'] = config.get('decoder_freeze_model')
    meta['clip_gradient_norm'] = config.get('clip_gradient_norm')
    meta['loss_onset_weight'] = config.get('loss_onset_weight')
    meta['loss_frame_weight'] = config.get('loss_frame_weight')
    meta['loss_velocity_weight'] = config.get('loss_velocity_weight')
    return meta


def resolve_checkpoint_dir(resume_from: Optional[str], logdir: str) -> str:
    if resume_from:
        path = resume_from
        if os.path.isfile(path):
            path = os.path.dirname(path)
        return path
    return logdir


def load_checkpoint_meta(ckpt_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(ckpt_dir, META_FILENAME)
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def resolve_resume_iteration(resume_iteration: Any, ckpt_dir: str) -> int:
    """Map None/'auto'/int onto the step to continue from (last completed step)."""
    if resume_iteration is None:
        raise ValueError('resolve_resume_iteration called without a resume request')

    if isinstance(resume_iteration, str) and resume_iteration.lower() == 'auto':
        meta = load_checkpoint_meta(ckpt_dir)
        if 'iteration' in meta:
            return int(meta['iteration'])
        # Legacy runs: infer from best_note_f1.txt is wrong for "latest";
        # require model-latest.pt and fall back to 0 only if missing entirely.
        if os.path.exists(os.path.join(ckpt_dir, MODEL_LATEST)):
            raise ValueError(
                f'resume_iteration=auto needs {META_FILENAME} in {ckpt_dir}. '
                f'Pass an explicit resume_iteration for legacy checkpoints.'
            )
        raise FileNotFoundError(f'No resumable checkpoint found in {ckpt_dir}')

    return int(resume_iteration)


def find_model_weights_path(ckpt_dir: str, resume_iteration: int) -> str:
    latest = os.path.join(ckpt_dir, MODEL_LATEST)
    if os.path.exists(latest):
        return latest
    stepped = os.path.join(ckpt_dir, f'model-{resume_iteration}.pt')
    if os.path.exists(stepped):
        return stepped
    raise FileNotFoundError(
        f'No model weights at {latest} or {stepped} (resume_iteration={resume_iteration})'
    )


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-12
        except (TypeError, ValueError):
            return a == b
    return a == b


def compare_meta_keys(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    keys: Tuple[str, ...],
) -> Tuple[bool, list]:
    mismatches = []
    for key in keys:
        if key not in saved:
            continue
        if not _values_equal(saved.get(key), current.get(key)):
            mismatches.append((key, saved.get(key), current.get(key)))
    return len(mismatches) == 0, mismatches


def decide_resume_restore(
    saved_train_meta: Mapping[str, Any],
    current_train_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return restore flags and human-readable reasons."""
    struct_ok, struct_mismatches = compare_meta_keys(
        saved_train_meta, current_train_meta, STRUCTURE_KEYS
    )
    adam_ok, adam_mismatches = compare_meta_keys(
        saved_train_meta, current_train_meta, ADAM_SENSITIVE_KEYS
    )
    sched_ok, sched_mismatches = compare_meta_keys(
        saved_train_meta, current_train_meta, SCHEDULER_SENSITIVE_KEYS
    )

    if not struct_ok:
        details = ', '.join(f'{k}: {old} -> {new}' for k, old, new in struct_mismatches)
        raise ValueError(
            'Refusing to resume: structure/hparams that affect weight shapes changed '
            f'({details}). Start a fresh run or hot-start weights explicitly.'
        )

    restore_optimizer = adam_ok
    # Scheduler state only makes sense on top of a restored optimizer + unchanged LR knobs.
    restore_scheduler = adam_ok and sched_ok

    reasons = []
    if not adam_ok:
        details = ', '.join(f'{k}: {old} -> {new}' for k, old, new in adam_mismatches)
        reasons.append(f'Rebuilding Adam ({details})')
    if adam_ok and not sched_ok:
        details = ', '.join(f'{k}: {old} -> {new}' for k, old, new in sched_mismatches)
        reasons.append(f'Rebuilding LR schedule from new config ({details})')
    if restore_optimizer and restore_scheduler:
        reasons.append('Restoring Adam + scheduler state')
    elif restore_optimizer:
        reasons.append('Restoring Adam; scheduler rebuilt from current LR config')

    return {
        'restore_optimizer': restore_optimizer,
        'restore_scheduler': restore_scheduler,
        'reasons': reasons,
        'adam_mismatches': adam_mismatches,
        'scheduler_mismatches': sched_mismatches,
    }


def load_best_metrics_from_txt(logdir: str) -> Tuple[float, float, int]:
    best_f1_path = os.path.join(logdir, 'best_note_f1.txt')
    if not os.path.exists(best_f1_path):
        return 0.0, 0.0, 0
    try:
        with open(best_f1_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        for line in content.split('\n'):
            if line.startswith('Best Note F1:'):
                parts = line.split(' | ')
                return (
                    float(parts[0].split(': ')[1]),
                    float(parts[1].split(': ')[1]),
                    int(parts[2].split(': ')[1]),
                )
        return float(content), 0.0, 0
    except (ValueError, IndexError, AttributeError):
        return 0.0, 0.0, 0


def save_training_checkpoint(
    logdir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    iteration: int,
    best_note_f1: float,
    best_note_f1_frame_f1: float,
    best_note_f1_step: int,
    train_meta: Mapping[str, Any],
    *,
    is_best: bool = False,
) -> None:
    """Overwrite latest (and optionally best) weights + optimizer/scheduler + meta."""
    atomic_torch_save(model.state_dict(), os.path.join(logdir, MODEL_LATEST))
    atomic_torch_save(optimizer.state_dict(), os.path.join(logdir, OPTIMIZER_LATEST))
    if scheduler is not None:
        atomic_torch_save(scheduler.state_dict(), os.path.join(logdir, SCHEDULER_LATEST))

    meta = {
        'iteration': int(iteration),
        'best_note_f1': float(best_note_f1),
        'best_note_f1_frame_f1': float(best_note_f1_frame_f1),
        'best_note_f1_step': int(best_note_f1_step),
        'train_meta': dict(train_meta),
    }
    atomic_json_save(meta, os.path.join(logdir, META_FILENAME))

    if is_best:
        atomic_torch_save(model.state_dict(), os.path.join(logdir, MODEL_BEST_NOTE))
