#!/usr/bin/env python3
"""Evaluate transcribed MIDI files against reference MIDI labels."""

import argparse
import json
import os
from collections import defaultdict

import mir_eval.onset
import numpy as np
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
from tqdm import tqdm

from onsets_and_frames.midi import parse_midi


def midi_notes_to_eval_arrays(notes):
    """Convert parse_midi output to mir_eval interval/pitch arrays."""
    if len(notes) == 0:
        return np.zeros((0, 2)), np.array([])

    intervals = np.column_stack([notes[:, 0], notes[:, 1]])
    pitches = np.array([midi_to_hz(note) for note in notes[:, 2]])
    return intervals, pitches


def evaluate_pair(ref_path, est_path):
    ref_notes = parse_midi(ref_path)
    est_notes = parse_midi(est_path)

    ref_intervals, ref_pitches = midi_notes_to_eval_arrays(ref_notes)
    est_intervals, est_pitches = midi_notes_to_eval_arrays(est_notes)

    ref_onsets = ref_notes[:, 0] if len(ref_notes) else np.array([])
    est_onsets = est_notes[:, 0] if len(est_notes) else np.array([])

    if len(ref_onsets) == 0 and len(est_onsets) == 0:
        onset_f1, onset_p, onset_r = 1.0, 1.0, 1.0
    else:
        onset_f1, onset_p, onset_r = mir_eval.onset.f_measure(
            ref_onsets, est_onsets, window=0.05
        )

    note_p, note_r, note_f1, note_overlap = evaluate_notes(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        offset_ratio=None,
    )

    return {
        'ref_notes': int(len(ref_notes)),
        'est_notes': int(len(est_notes)),
        'onset_f1': float(onset_f1),
        'onset_precision': float(onset_p),
        'onset_recall': float(onset_r),
        'note_f1': float(note_f1),
        'note_precision': float(note_p),
        'note_recall': float(note_r),
        'note_overlap': float(note_overlap),
    }


def collect_pairs(pred_root, ref_root):
    pairs = []
    skipped = []

    for dirpath, _, filenames in os.walk(pred_root):
        for filename in filenames:
            if not filename.endswith('.pred.mid'):
                continue
            pred_path = os.path.join(dirpath, filename)
            rel_dir = os.path.relpath(dirpath, pred_root)
            if rel_dir == '.':
                skipped.append((pred_path, 'no_song_subfolder'))
                continue

            ref_path = os.path.join(ref_root, rel_dir, 'drums.mid')
            if not os.path.isfile(ref_path):
                skipped.append((pred_path, f'missing_ref:{ref_path}'))
                continue
            pairs.append((rel_dir, ref_path, pred_path))

    return sorted(pairs), skipped


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate transcription MIDI against reference MIDI')
    parser.add_argument(
        '--pred-root',
        default='/data1/weiyangding/test500_hidrum_drum_midi',
    )
    parser.add_argument(
        '--ref-root',
        default='/data1/weiyangding/test500_drum_ref_audio_midi',
    )
    parser.add_argument(
        '--output-json',
        default='/data1/weiyangding/test500_hidrum_drum_midi_evaluation.json',
    )
    args = parser.parse_args()

    pairs, skipped = collect_pairs(args.pred_root, args.ref_root)
    if not pairs:
        raise RuntimeError('No valid prediction/reference pairs found')

    per_song = {}
    metrics = defaultdict(list)

    for song, ref_path, pred_path in tqdm(pairs, desc='Evaluating'):
        result = evaluate_pair(ref_path, pred_path)
        result['ref_path'] = ref_path
        result['pred_path'] = pred_path
        per_song[song] = result
        for key in ('onset_f1', 'note_f1', 'onset_precision', 'onset_recall', 'note_precision', 'note_recall'):
            metrics[key].append(result[key])

    summary = {
        'num_pairs': len(pairs),
        'num_skipped': len(skipped),
        'onset_f1': summarize(metrics['onset_f1']),
        'note_f1': summarize(metrics['note_f1']),
        'onset_precision': summarize(metrics['onset_precision']),
        'onset_recall': summarize(metrics['onset_recall']),
        'note_precision': summarize(metrics['note_precision']),
        'note_recall': summarize(metrics['note_recall']),
    }

    output = {
        'pred_root': args.pred_root,
        'ref_root': args.ref_root,
        'summary': summary,
        'per_song': per_song,
        'skipped': [{'pred_path': p, 'reason': r} for p, r in skipped],
    }

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print('\n=== Transcription Evaluation ===')
    print(f'Pairs evaluated: {summary["num_pairs"]}')
    print(f'Skipped: {summary["num_skipped"]}')
    print(f'Onset F1:  {summary["onset_f1"]["mean"]:.4f} ± {summary["onset_f1"]["std"]:.4f}')
    print(f'Note F1:   {summary["note_f1"]["mean"]:.4f} ± {summary["note_f1"]["std"]:.4f}')
    print(f'Onset P/R: {summary["onset_precision"]["mean"]:.4f} / {summary["onset_recall"]["mean"]:.4f}')
    print(f'Note P/R:  {summary["note_precision"]["mean"]:.4f} / {summary["note_recall"]["mean"]:.4f}')
    print(f'Results saved to: {args.output_json}')


if __name__ == '__main__':
    main()
