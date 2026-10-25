"""
Slakh drum extraction utility.

This script scans a Slakh-style directory tree, finds stems whose metadata marks
them as drums, mixes those drum stems into one drum-only audio file per track,
and converts the aligned drum MIDI into a TSV label file that matches the
project's `PianoRollAudioDataset.load()` input format.

Key arguments:
- --input-root: Slakh root directory, e.g. a `train/validation/test` tree.
- --output-root: Destination root for `{split}_data`, `{split}_label`, and manifests.
- --target-sr: Output sample rate. Default is 24000 to match the current project constants.
- --limit: Optional cap for quick smoke tests before processing the full dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi
import soundfile as sf
import yaml
from scipy.signal import resample_poly


SUPPORTED_AUDIO_SUFFIXES = (".wav", ".flac")
SUPPORTED_MIDI_SUFFIXES = (".mid", ".midi")
DEFAULT_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract drum-only audio and TSV labels from a Slakh dataset."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory of Slakh data that contains train/validation/test.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for extracted drum dataset.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=24000,
        help="Target sample rate for output audio.",
    )
    parser.add_argument(
        "--audio-format",
        choices=("flac", "wav"),
        default="flac",
        help="Output audio container.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on processed tracks per split for debugging.",
    )
    parser.add_argument(
        "--min-midi",
        type=int,
        default=21,
        help="Minimum MIDI note kept in the TSV labels.",
    )
    parser.add_argument(
        "--max-midi",
        type=int,
        default=108,
        help="Maximum MIDI note kept in the TSV labels.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_existing_file(base_path: Path, suffixes: Iterable[str]) -> Path | None:
    for suffix in suffixes:
        candidate = base_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    # Why: polyphase resampling is fast and keeps this conversion script aligned
    # with the project's fixed-rate training pipeline without adding ffmpeg steps.
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def read_audio_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # Why: Slakh stems can arrive with channel axes depending on prior
        # preprocessing; forcing mono keeps label/audio alignment deterministic.
        audio = audio.mean(axis=1)
    return resample_audio(audio, sr, target_sr), target_sr


def mix_audios(audio_paths: list[Path], target_sr: int) -> tuple[np.ndarray, int]:
    mixed = None
    for audio_path in audio_paths:
        audio, _ = read_audio_mono(audio_path, target_sr)
        if mixed is None:
            mixed = audio
            continue
        if len(audio) > len(mixed):
            mixed = np.pad(mixed, (0, len(audio) - len(mixed)))
        elif len(audio) < len(mixed):
            audio = np.pad(audio, (0, len(mixed) - len(audio)))
        mixed = mixed + audio

    if mixed is None:
        raise ValueError("No drum audio found to mix.")

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.999:
        # Why: summed drum stems can exceed full scale; gentle peak normalization
        # avoids clipped supervision audio while preserving relative dynamics.
        mixed = mixed / peak * 0.999
    return mixed.astype(np.float32, copy=False), target_sr


def midi_to_events(
    midi_paths: list[Path], min_midi: int, max_midi: int
) -> tuple[list[tuple[float, float, int, int]], int]:
    events: list[tuple[float, float, int, int]] = []
    skipped = 0

    for midi_path in midi_paths:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        for instrument in midi.instruments:
            for note in instrument.notes:
                if note.pitch < min_midi or note.pitch > max_midi:
                    skipped += 1
                    continue
                onset = float(note.start)
                offset = float(note.end)
                if offset <= onset:
                    # Why: zero-length notes break downstream frame labels, so we
                    # enforce a tiny positive duration before writing the TSV.
                    offset = onset + 1e-3
                events.append((onset, offset, int(note.pitch), int(note.velocity)))

    events.sort(key=lambda x: (x[0], x[2], x[1], x[3]))
    return events, skipped


def write_tsv(path: Path, events: list[tuple[float, float, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("onset", "offset", "note", "velocity"))
        for onset, offset, note, velocity in events:
            writer.writerow(
                (
                    f"{onset:.6f}",
                    f"{offset:.6f}",
                    note,
                    velocity,
                )
            )


def iter_track_dirs(split_root: Path) -> list[Path]:
    return sorted(
        path for path in split_root.iterdir() if path.is_dir() and path.name.startswith("Track")
    )


def collect_drum_sources(track_dir: Path) -> tuple[list[Path], list[Path], list[str]]:
    metadata_path = track_dir / "metadata.yaml"
    metadata = load_yaml(metadata_path)
    audio_paths: list[Path] = []
    midi_paths: list[Path] = []
    stem_ids: list[str] = []

    for stem_id, stem_info in metadata.get("stems", {}).items():
        if not stem_info.get("is_drum", False):
            continue

        audio_path = None
        if stem_info.get("audio_rendered", False):
            audio_path = find_existing_file(track_dir / "stems" / stem_id, SUPPORTED_AUDIO_SUFFIXES)
        midi_path = None
        if stem_info.get("midi_saved", False):
            midi_path = find_existing_file(track_dir / "MIDI" / stem_id, SUPPORTED_MIDI_SUFFIXES)

        if audio_path is not None:
            audio_paths.append(audio_path)
        if midi_path is not None:
            midi_paths.append(midi_path)
        if audio_path is not None or midi_path is not None:
            stem_ids.append(stem_id)

    return audio_paths, midi_paths, stem_ids


def process_split(args: argparse.Namespace, split: str) -> dict:
    split_root = args.input_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_root}")

    output_audio_dir = args.output_root / f"{split}_data"
    output_label_dir = args.output_root / f"{split}_label"
    manifest_path = args.output_root / f"{split}_manifest.csv"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "example_id",
        "audio_path",
        "label_path",
        "track_dir",
        "mix_path",
        "drum_stem_ids",
        "num_drum_audio_stems",
        "num_drum_midi_stems",
        "num_events",
        "skipped_events_out_of_range",
    ]

    processed = 0
    skipped_tracks = 0
    total_events = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for track_dir in iter_track_dirs(split_root):
            if args.limit is not None and processed >= args.limit:
                break

            audio_paths, midi_paths, stem_ids = collect_drum_sources(track_dir)
            if not audio_paths or not midi_paths:
                skipped_tracks += 1
                continue

            example_id = f"{track_dir.name}_drums"
            output_audio_path = output_audio_dir / f"{example_id}.{args.audio_format}"
            output_label_path = output_label_dir / f"{example_id}.tsv"
            mix_path = find_existing_file(track_dir / "mix", SUPPORTED_AUDIO_SUFFIXES)

            mixed_audio, _ = mix_audios(audio_paths, args.target_sr)
            sf.write(output_audio_path, mixed_audio, args.target_sr)

            events, skipped = midi_to_events(midi_paths, args.min_midi, args.max_midi)
            write_tsv(output_label_path, events)

            writer.writerow(
                {
                    "split": split,
                    "example_id": example_id,
                    "audio_path": str(output_audio_path.resolve()),
                    "label_path": str(output_label_path.resolve()),
                    "track_dir": str(track_dir.resolve()),
                    "mix_path": str(mix_path.resolve()) if mix_path is not None else "",
                    "drum_stem_ids": json.dumps(stem_ids, ensure_ascii=True),
                    "num_drum_audio_stems": len(audio_paths),
                    "num_drum_midi_stems": len(midi_paths),
                    "num_events": len(events),
                    "skipped_events_out_of_range": skipped,
                }
            )

            processed += 1
            total_events += len(events)

    return {
        "split": split,
        "processed_tracks": processed,
        "skipped_tracks": skipped_tracks,
        "total_events": total_events,
        "manifest_path": str(manifest_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "target_sr": args.target_sr,
        "audio_format": args.audio_format,
        "splits": list(args.splits),
        "min_midi": args.min_midi,
        "max_midi": args.max_midi,
        "per_split": [],
    }

    for split in args.splits:
        summary["per_split"].append(process_split(args, split))

    summary_path = args.output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
