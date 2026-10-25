"""Batch-size helpers for grouped training and GPU memory tiers."""


def compute_grouped_quotas(batch_size):
    """Scale matched/unmatched/slakh quotas while preserving the 5:2:1 ratio."""
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')

    matched = max(1, (batch_size * 5) // 8)
    unmatched = max(0, (batch_size * 2) // 8)
    slakh = batch_size - matched - unmatched

    if slakh < 0:
        slakh = 0
        unmatched = max(0, batch_size - matched)
        if matched + unmatched > batch_size:
            matched = max(1, batch_size - unmatched)

    remainder = batch_size - matched - unmatched - slakh
    if remainder > 0:
        matched += remainder
    elif remainder < 0:
        matched = max(1, matched + remainder)

    if matched + unmatched + slakh != batch_size:
        raise ValueError(
            f'Unable to derive grouped quotas for batch_size={batch_size}: '
            f'{matched}/{unmatched}/{slakh}'
        )
    return matched, unmatched, slakh


def compute_groove_egmd_quotas(batch_size):
    """Scale matched/unmatched quotas while preserving the 5:3 ratio (no Slakh)."""
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    matched = max(1, (batch_size * 5) // 8)
    unmatched = batch_size - matched
    return matched, unmatched, 0


def is_default_heavy_grouped_setup(
    train_on,
    encoder_model,
    decoder_model,
    matched_samples_per_batch,
    unmatched_samples_per_batch,
    slakh_samples_per_batch,
    batch_size,
):
    return (
        train_on == 'GroupedGrooveEgmdSlakh'
        and encoder_model == 'MERT_330m'
        and decoder_model == 'Qwen3-0.6B'
        and batch_size == 8
        and (
            (matched_samples_per_batch == 5 and unmatched_samples_per_batch == 2 and slakh_samples_per_batch == 1)
            or (matched_samples_per_batch == 5 and unmatched_samples_per_batch == 3 and slakh_samples_per_batch == 0)
        )
    )


def apply_gpu_memory_batch_defaults(
    total_gpu_memory,
    train_on,
    encoder_model,
    decoder_model,
    batch_size,
    matched_samples_per_batch,
    unmatched_samples_per_batch,
    slakh_samples_per_batch,
    sequence_length,
    distributed=False,
):
    """
    Adjust batch/quotas for known-safe memory tiers on the default heavy setup.

    Returns:
        (batch_size, matched, unmatched, slakh, sequence_length, message)
    """
    message = None

    if total_gpu_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        matched_samples_per_batch, unmatched_samples_per_batch, slakh_samples_per_batch = (
            compute_grouped_quotas(batch_size)
        )
        message = (
            f'Reducing batch size to {batch_size} and sequence_length to '
            f'{sequence_length} to save memory'
        )
    elif total_gpu_memory < 26e9 and is_default_heavy_grouped_setup(
        train_on,
        encoder_model,
        decoder_model,
        matched_samples_per_batch,
        unmatched_samples_per_batch,
        slakh_samples_per_batch,
        batch_size,
    ):
        batch_size = 4
        matched_samples_per_batch = 2
        unmatched_samples_per_batch = 1
        slakh_samples_per_batch = 1
        message = (
            'Detected 24GB-class GPU with MERT_330m + Qwen3-0.6B grouped training; '
            'using memory-safe batch/quotas: batch_size=4 (matched=2, unmatched=1, slakh=1)'
        )
    elif total_gpu_memory >= 40e9 and is_default_heavy_grouped_setup(
        train_on,
        encoder_model,
        decoder_model,
        matched_samples_per_batch,
        unmatched_samples_per_batch,
        slakh_samples_per_batch,
        batch_size,
    ):
        # Do not keep the old 46GB probe (batch 16). Current runs use batch 32
        # so step counts match the B2-none recipe. 80GB A800 has headroom;
        # larger batches were measured and did not improve samples/sec.
        batch_size = 32
        if slakh_samples_per_batch == 0:
            matched_samples_per_batch, unmatched_samples_per_batch, slakh_samples_per_batch = (
                compute_groove_egmd_quotas(batch_size)
            )
        else:
            matched_samples_per_batch, unmatched_samples_per_batch, slakh_samples_per_batch = (
                compute_grouped_quotas(batch_size)
            )
        mode = 'distributed' if distributed else 'single-GPU'
        message = (
            f'Detected >=40GB GPU with MERT_330m + Qwen3-0.6B grouped training ({mode}); '
            f'using batch_size={batch_size} '
            f'(matched={matched_samples_per_batch}, unmatched={unmatched_samples_per_batch}, '
            f'slakh={slakh_samples_per_batch})'
        )

    return (
        batch_size,
        matched_samples_per_batch,
        unmatched_samples_per_batch,
        slakh_samples_per_batch,
        sequence_length,
        message,
    )
