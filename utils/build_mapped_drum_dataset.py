"""
Build a mapped drum dataset view from an existing drum-only dataset.

Key arguments:
- --source-root: Existing drum dataset root with `{split}_data` and `{split}_label`.
- --output-root: New dataset root whose name can encode label semantics, e.g. `7class`.
- --mapping-mode: Label mapping mode, matching the project's drum standard (`gm`, `3`, `7`).
- --merge-tambourine-to-hh: Whether note 54 should collapse into the hi-hat class.

Why this file exists:
- Reusing already-rendered 24 kHz audio avoids another expensive Slakh audio pass.
- A separate output root makes raw-note and class-mapped datasets easy to distinguish later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import map_tsv_notes


DEFAULT_SPLITS = ("train", "validation", "test")

# Why: Slakh includes auxiliary percussion notes beyond the project's original
# Roland-centric 7-class table. These additions collapse those extra events into
# the same seven target classes so the resulting labels are truly 7-class.
SLAKH_AUX_TO_7_CLASS_MAPPING = {
    27: 53,  # Laser / custom percussion -> Bell/Percussion bucket
    58: 53,  # Vibraslap -> Bell/Percussion bucket
    60: 50,  # Hi Bongo -> Toms
    61: 50,  # Low Bongo -> Toms
    62: 50,  # Mute Hi Conga -> Toms
    63: 50,  # Open Hi Conga -> Toms
    64: 50,  # Low Conga -> Toms
    65: 50,  # High Timbale -> Toms
    66: 50,  # Low Timbale -> Toms
    67: 53,  # High Agogo -> Bell/Percussion bucket
    68: 53,  # Low Agogo -> Bell/Percussion bucket
    69: 42,  # Cabasa -> Hi-Hat family
    70: 42,  # Maracas -> Hi-Hat family
    71: 53,  # Short Whistle -> Bell/Percussion bucket
    72: 53,  # Long Whistle -> Bell/Percussion bucket
    73: 42,  # Short Guiro -> Hi-Hat family
    74: 42,  # Long Guiro -> Hi-Hat family
    75: 53,  # Claves -> Bell/Percussion bucket
    76: 53,  # Hi Wood Block -> Bell/Percussion bucket
    77: 53,  # Low Wood Block -> Bell/Percussion bucket
    78: 50,  # Mute Cuica -> Toms
    79: 50,  # Open Cuica -> Toms
    80: 53,  # Mute Triangle -> Bell/Percussion bucket
    81: 53,  # Open Triangle -> Bell/Percussion bucket
    82: 42,  # Shaker -> Hi-Hat family
    83: 53,  # Sleigh Bell -> Bell/Percussion bucket
    84: 53,  # Bell Tree -> Bell/Percussion bucket
    85: 53,  # Castanets -> Bell/Percussion bucket
    86: 36,  # Surdu dead stroke -> Kick
    87: 36,  # Surdu -> Kick
    89: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    90: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    91: 38,  # Snare drum rod -> Snare
    92: 53,  # Ocean drum / custom auxiliary percussion -> Bell/Percussion bucket
    93: 38,  # Snare drum brush -> Snare
    94: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    95: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    96: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    98: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    99: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    100: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    104: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
    105: 53,  # Custom auxiliary percussion -> Bell/Percussion bucket
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a mapped drum dataset view from an extracted drum dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Existing drum dataset root with {split}_data and {split}_label.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New dataset root for the mapped labels and linked audio directories.",
    )
    parser.add_argument(
        "--mapping-mode",
        choices=("gm", "3", "7"),
        required=True,
        help="Target drum mapping mode.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to process.",
    )
    parser.add_argument(
        "--merge-tambourine-to-hh",
        action="store_true",
        help="Map note 54 to hi-hat to match the project's legacy 7-class behavior.",
    )
    return parser.parse_args()


def load_mapping(mode: str, merge_tambourine_to_hh: bool) -> dict[int, int]:
    # Why: the project's canonical drum-class rules already live in map_tsv_notes.py,
    # so this wrapper reuses the same source of truth instead of re-defining classes.
    map_tsv_notes.MERGE_TAMBOURINE_TO_HH = merge_tambourine_to_hh
    mapping = map_tsv_notes.get_mapping(mode)
    if mode == "7":
        # Why: Slakh uses extra percussion notes that are outside the original
        # project mapping table, so we extend that table here to keep 7-class
        # labels closed under all observed Slakh percussion notes.
        mapping.update(SLAKH_AUX_TO_7_CLASS_MAPPING)
    return mapping


def ensure_clean_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {output_root}. Please remove it or choose a new path."
        )
    output_root.mkdir(parents=True, exist_ok=False)


def map_label_file(source_path: Path, target_path: Path, pitch_mapping: dict[int, int]) -> int:
    num_events = 0
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8", newline="") as src_handle:
        reader = csv.DictReader(src_handle, delimiter="\t")
        with target_path.open("w", encoding="utf-8", newline="") as dst_handle:
            writer = csv.DictWriter(
                dst_handle,
                fieldnames=("onset", "offset", "note", "velocity"),
                delimiter="\t",
            )
            writer.writeheader()
            for row in reader:
                note = int(row["note"])
                row["note"] = str(pitch_mapping.get(note, note))
                writer.writerow(row)
                num_events += 1
    return num_events


def rewrite_manifest(
    source_path: Path,
    target_path: Path,
    output_root: Path,
    split: str,
) -> int:
    row_count = 0
    with source_path.open("r", encoding="utf-8", newline="") as src_handle:
        reader = csv.DictReader(src_handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Manifest has no header: {source_path}")
        with target_path.open("w", encoding="utf-8", newline="") as dst_handle:
            writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                audio_name = Path(row["audio_path"]).name
                label_name = Path(row["label_path"]).name
                # Why: manifests should point at the new dataset root so downstream
                # jobs can distinguish raw-note and mapped datasets by path alone.
                row["audio_path"] = str(output_root / f"{split}_data" / audio_name)
                row["label_path"] = str(output_root / f"{split}_label" / label_name)
                writer.writerow(row)
                row_count += 1
    return row_count


def process_split(
    source_root: Path,
    output_root: Path,
    split: str,
    pitch_mapping: dict[int, int],
) -> dict:
    source_audio_dir = source_root / f"{split}_data"
    source_label_dir = source_root / f"{split}_label"
    source_manifest_path = source_root / f"{split}_manifest.csv"

    output_audio_dir = output_root / f"{split}_data"
    output_label_dir = output_root / f"{split}_label"
    output_manifest_path = output_root / f"{split}_manifest.csv"

    if not source_audio_dir.exists() or not source_label_dir.exists():
        raise FileNotFoundError(f"Missing source split directories for {split}: {source_root}")

    # Why: the mapped dataset only changes labels; symlinking audio avoids duplicating
    # a large 24 kHz corpus while still giving the mapped dataset its own root path.
    output_audio_dir.symlink_to(source_audio_dir.resolve(), target_is_directory=True)

    total_events = 0
    processed_tracks = 0
    for source_label_path in sorted(source_label_dir.glob("*.tsv")):
        target_label_path = output_label_dir / source_label_path.name
        total_events += map_label_file(source_label_path, target_label_path, pitch_mapping)
        processed_tracks += 1

    manifest_rows = rewrite_manifest(source_manifest_path, output_manifest_path, output_root, split)
    if manifest_rows != processed_tracks:
        raise ValueError(
            f"Manifest/data mismatch for {split}: {manifest_rows} manifest rows vs {processed_tracks} labels."
        )

    return {
        "split": split,
        "processed_tracks": processed_tracks,
        "total_events": total_events,
        "manifest_path": str(output_manifest_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    ensure_clean_output_root(args.output_root)

    pitch_mapping = load_mapping(args.mapping_mode, args.merge_tambourine_to_hh)
    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "mapping_mode": args.mapping_mode,
        "merge_tambourine_to_hh": args.merge_tambourine_to_hh,
        "splits": list(args.splits),
        "per_split": [],
        "pitch_mapping": {str(k): v for k, v in sorted(pitch_mapping.items())},
    }

    for split in args.splits:
        summary["per_split"].append(process_split(args.source_root, args.output_root, split, pitch_mapping))

    summary_path = args.output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
