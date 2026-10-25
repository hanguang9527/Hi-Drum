#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function:
    Convert ENST, MDB Drums, and IDMT-SMT-Drums into test-only datasets that
    match the current Hi_Drum reader contract.

Key parameters:
    --input-root: Root directory that stores the downloaded raw datasets.
    --output-root: Root directory to save processed test-only datasets.
    --datasets: Which datasets to convert. Supports: enst, mdb, idmt, all.
    --overwrite: Whether to overwrite existing converted files.

Notes:
    1. The current project reader expects `test_data/*.flac` and
       `test_label/*.tsv`.
    2. The current reader still consumes four TSV columns
       (onset, offset, note, velocity), so this script writes a pseudo offset
       for drum hits even though drum transcription does not use a real offset
       label.
    3. The project constants assume 24 kHz audio and a hop size derived from
       24 kHz, so conversion here always forces mono + 24 kHz to keep train and
       evaluation time grids consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
DEFAULT_VELOCITY = 100
HOP_LENGTH = TARGET_SAMPLE_RATE // 75
# Why: keep a small but non-zero frame span for the current dataset reader,
# while still treating drum hits as onset-centric events.
PSEUDO_OFFSET_SECONDS = 3 * HOP_LENGTH / TARGET_SAMPLE_RATE


ENST_LABEL_MAP = {
    "bd": 36,
    "sd": 38,
    "sd-": 38,
    "cs": 38,
    "lt": 50,
    "mt": 50,
    "lmt": 50,
    "lft": 50,
    # Why: the ENST paper's label table explicitly defines `mtr` as
    # "Mid tom, hit on the rim", so it still belongs to the tom family and
    # should map into the project's TT class instead of being discarded.
    "mtr": 50,
    "chh": 42,
    "ohh": 42,
    "cr1": 49,
    "cr2": 49,
    "cr5": 49,
    "c1": 49,
    "c4": 49,
    "ch1": 49,
    "ch5": 49,
    "spl2": 49,
    "rc2": 59,
    "rc3": 59,
    "rc4": 59,
    "cb": 53,
}

ENST_IGNORED_LABELS: set[str] = set()

MDB_SUBCLASS_MAP = {
    "KD": 36,
    "SD": 38,
    "SDB": 38,
    "SDD": 38,
    "SDF": 38,
    "SDG": 38,
    "SDNS": 38,
    "SPC": 38,
    "SST": 38,
    "CHH": 42,
    "OHH": 42,
    "PHH": 42,
    "CHC": 42,
    "LFT": 50,
    "HFT": 50,
    "MHT": 50,
    "HIT": 50,
    "CRC": 49,
    "RDC": 59,
    "RDB": 53,
    # Why: align with the existing project mapping where tambourine is merged
    # into hi-hat for the 7-class setting.
    "TMB": 42,
}

IDMT_LABEL_MAP = {
    "KD": 36,
    "SD": 38,
    "HH": 42,
}


@dataclass
class DatasetRecord:
    output_dataset: str
    split: str
    source_audio: str
    source_label: str
    output_audio: str
    output_label: str
    event_count: int
    ignored_count: int


class ConversionStats:
    def __init__(self) -> None:
        self.dataset_event_counts: Counter[str] = Counter()
        self.dataset_file_counts: Counter[str] = Counter()
        self.ignored_label_counts: Counter[str] = Counter()
        self.records: list[DatasetRecord] = []

    def add_record(self, record: DatasetRecord) -> None:
        self.records.append(record)
        self.dataset_file_counts[record.output_dataset] += 1
        self.dataset_event_counts[record.output_dataset] += record.event_count


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_output_dirs(dataset_root: Path, overwrite: bool) -> tuple[Path, Path]:
    if overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)
    audio_dir = dataset_root / "test_data"
    label_dir = dataset_root / "test_label"
    ensure_dir(audio_dir)
    ensure_dir(label_dir)
    return audio_dir, label_dir


def convert_audio_to_project_flac(src_audio: Path, dst_flac: Path) -> None:
    audio, sample_rate = sf.read(str(src_audio), dtype="float32", always_2d=True)

    # Why: accompaniment and full-mix files can be stereo, but the project
    # reader and existing processed data assume mono waveforms.
    if audio.shape[1] != TARGET_CHANNELS:
        audio = np.mean(audio, axis=1, keepdims=True)

    audio = audio[:, 0]

    if sample_rate != TARGET_SAMPLE_RATE:
        up = TARGET_SAMPLE_RATE
        down = sample_rate
        gcd = math.gcd(up, down)
        audio = resample_poly(audio, up // gcd, down // gcd).astype(np.float32)

    sf.write(
        str(dst_flac),
        audio,
        TARGET_SAMPLE_RATE,
        format="FLAC",
        subtype="PCM_16",
    )


def write_tsv(events: list[tuple[float, int, int]], dst_tsv: Path) -> None:
    with dst_tsv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["onset", "offset", "note", "velocity"])
        for onset, note, velocity in sorted(events, key=lambda item: (item[0], item[1])):
            offset = onset + PSEUDO_OFFSET_SECONDS
            writer.writerow(
                [
                    f"{onset:.6f}",
                    f"{offset:.6f}",
                    int(note),
                    int(velocity),
                ]
            )


def validate_output_audio(dst_flac: Path) -> None:
    info = sf.info(str(dst_flac))
    if info.samplerate != TARGET_SAMPLE_RATE:
        raise ValueError(f"{dst_flac} has samplerate={info.samplerate}, expected {TARGET_SAMPLE_RATE}")
    if info.channels != TARGET_CHANNELS:
        raise ValueError(f"{dst_flac} has channels={info.channels}, expected {TARGET_CHANNELS}")


def parse_enst_events(label_path: Path, stats: ConversionStats) -> tuple[list[tuple[float, int, int]], int]:
    events: list[tuple[float, int, int]] = []
    ignored_count = 0
    with label_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for raw_line in fp:
            parts = raw_line.strip().split()
            if len(parts) < 2:
                continue
            onset = float(parts[0])
            label = parts[1].lower()
            if label in ENST_IGNORED_LABELS:
                stats.ignored_label_counts[f"ENST:{label}"] += 1
                ignored_count += 1
                continue
            if label not in ENST_LABEL_MAP:
                stats.ignored_label_counts[f"ENST:UNMAPPED:{label}"] += 1
                ignored_count += 1
                continue
            events.append((onset, ENST_LABEL_MAP[label], DEFAULT_VELOCITY))
    return events, ignored_count


def convert_enst(input_root: Path, output_root: Path, overwrite: bool, stats: ConversionStats) -> None:
    source_dir = input_root / "ENST-drums-dataset" / "accompaniment_all_drummers_test"
    label_files = sorted(source_dir.glob("*.txt"))

    datasets = {
        "enst_eval_drum_only": prepare_output_dirs(output_root / "enst_eval_drum_only", overwrite),
        "enst_eval_accompaniment": prepare_output_dirs(output_root / "enst_eval_accompaniment", overwrite),
    }

    for label_path in label_files:
        basename = label_path.stem
        events, ignored_count = parse_enst_events(label_path, stats)
        if not events:
            continue

        variants = {
            "enst_eval_drum_only": source_dir / f"{basename}.wav",
            "enst_eval_accompaniment": source_dir / f"{basename}_acc.wav",
        }

        for dataset_name, src_audio in variants.items():
            if not src_audio.exists():
                continue

            audio_dir, label_dir = datasets[dataset_name]
            dst_audio = audio_dir / f"{basename}.flac"
            dst_label = label_dir / f"{basename}.tsv"
            convert_audio_to_project_flac(src_audio, dst_audio)
            write_tsv(events, dst_label)
            validate_output_audio(dst_audio)
            stats.add_record(
                DatasetRecord(
                    output_dataset=dataset_name,
                    split="test",
                    source_audio=str(src_audio),
                    source_label=str(label_path),
                    output_audio=str(dst_audio),
                    output_label=str(dst_label),
                    event_count=len(events),
                    ignored_count=ignored_count,
                )
            )


def parse_mdb_events(label_path: Path, stats: ConversionStats) -> tuple[list[tuple[float, int, int]], int]:
    events: list[tuple[float, int, int]] = []
    ignored_count = 0
    with label_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for raw_line in fp:
            parts = raw_line.strip().split()
            if len(parts) < 2:
                continue
            onset = float(parts[0])
            label = parts[1].upper()
            if label not in MDB_SUBCLASS_MAP:
                stats.ignored_label_counts[f"MDB:UNMAPPED:{label}"] += 1
                ignored_count += 1
                continue
            events.append((onset, MDB_SUBCLASS_MAP[label], DEFAULT_VELOCITY))
    return events, ignored_count


def convert_mdb(input_root: Path, output_root: Path, overwrite: bool, stats: ConversionStats) -> None:
    source_root = input_root / "MDBDrums" / "MDB Drums"
    subclass_dir = source_root / "annotations" / "subclass"

    datasets = {
        "mdb_eval_drum_only": prepare_output_dirs(output_root / "mdb_eval_drum_only", overwrite),
        "mdb_eval_full_mix": prepare_output_dirs(output_root / "mdb_eval_full_mix", overwrite),
    }

    for label_path in sorted(subclass_dir.glob("*_subclass.txt")):
        song_name = label_path.name.replace("_subclass.txt", "")
        events, ignored_count = parse_mdb_events(label_path, stats)
        if not events:
            continue

        variants = {
            "mdb_eval_drum_only": source_root / "audio" / "drum_only" / f"{song_name}_Drum.wav",
            "mdb_eval_full_mix": source_root / "audio" / "full_mix" / f"{song_name}_MIX.wav",
        }

        for dataset_name, src_audio in variants.items():
            if not src_audio.exists():
                continue

            audio_dir, label_dir = datasets[dataset_name]
            dst_audio = audio_dir / f"{song_name}.flac"
            dst_label = label_dir / f"{song_name}.tsv"
            convert_audio_to_project_flac(src_audio, dst_audio)
            write_tsv(events, dst_label)
            validate_output_audio(dst_audio)
            stats.add_record(
                DatasetRecord(
                    output_dataset=dataset_name,
                    split="test",
                    source_audio=str(src_audio),
                    source_label=str(label_path),
                    output_audio=str(dst_audio),
                    output_label=str(dst_label),
                    event_count=len(events),
                    ignored_count=ignored_count,
                )
            )


def parse_idmt_events(label_path: Path, stats: ConversionStats) -> tuple[list[tuple[float, int, int]], int]:
    events: list[tuple[float, int, int]] = []
    ignored_count = 0
    tree = ET.parse(label_path)
    for event in tree.findall(".//event"):
        onset_text = event.findtext("onsetSec")
        instrument = event.findtext("instrument")
        if onset_text is None or instrument is None:
            continue
        label = instrument.upper()
        if label not in IDMT_LABEL_MAP:
            stats.ignored_label_counts[f"IDMT:UNMAPPED:{label}"] += 1
            ignored_count += 1
            continue
        events.append((float(onset_text), IDMT_LABEL_MAP[label], DEFAULT_VELOCITY))
    return events, ignored_count


def convert_idmt(input_root: Path, output_root: Path, overwrite: bool, stats: ConversionStats) -> None:
    source_root = input_root / "IDMT-SMT-Drums"
    xml_dir = source_root / "annotation_xml"
    audio_dir = source_root / "audio"
    out_audio_dir, out_label_dir = prepare_output_dirs(output_root / "idmt_eval_mix", overwrite)

    for label_path in sorted(xml_dir.glob("*#MIX.xml")):
        basename = label_path.stem
        src_audio = audio_dir / f"{basename}.wav"
        if not src_audio.exists():
            continue

        events, ignored_count = parse_idmt_events(label_path, stats)
        if not events:
            continue

        dst_audio = out_audio_dir / f"{basename}.flac"
        dst_label = out_label_dir / f"{basename}.tsv"
        convert_audio_to_project_flac(src_audio, dst_audio)
        write_tsv(events, dst_label)
        validate_output_audio(dst_audio)
        stats.add_record(
            DatasetRecord(
                output_dataset="idmt_eval_mix",
                split="test",
                source_audio=str(src_audio),
                source_label=str(label_path),
                output_audio=str(dst_audio),
                output_label=str(dst_label),
                event_count=len(events),
                ignored_count=ignored_count,
            )
        )


def write_summary(output_root: Path, stats: ConversionStats) -> None:
    ensure_dir(output_root)

    summary = {
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "target_channels": TARGET_CHANNELS,
        "default_velocity": DEFAULT_VELOCITY,
        "pseudo_offset_seconds": PSEUDO_OFFSET_SECONDS,
        "dataset_file_counts": dict(stats.dataset_file_counts),
        "dataset_event_counts": dict(stats.dataset_event_counts),
        "ignored_label_counts": dict(stats.ignored_label_counts),
        "datasets": [
            {
                "output_dataset": record.output_dataset,
                "split": record.split,
                "source_audio": record.source_audio,
                "source_label": record.source_label,
                "output_audio": record.output_audio,
                "output_label": record.output_label,
                "event_count": record.event_count,
                "ignored_count": record.ignored_count,
            }
            for record in stats.records
        ],
    }

    with (output_root / "conversion_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    with (output_root / "conversion_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "output_dataset",
                "split",
                "source_audio",
                "source_label",
                "output_audio",
                "output_label",
                "event_count",
                "ignored_count",
            ]
        )
        for record in stats.records:
            writer.writerow(
                [
                    record.output_dataset,
                    record.split,
                    record.source_audio,
                    record.source_label,
                    record.output_audio,
                    record.output_label,
                    record.event_count,
                    record.ignored_count,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert external drum datasets into test-only project format.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/dingweiyang/data/drum_dataset"),
        help="Root directory of downloaded raw datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/dingweiyang/data/drum_dataset_processed"),
        help="Root directory to save processed evaluation datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "enst", "mdb", "idmt"],
        help="Datasets to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite processed datasets if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = ConversionStats()

    datasets = set(args.datasets)
    if "all" in datasets:
        datasets = {"enst", "mdb", "idmt"}

    if "enst" in datasets:
        convert_enst(args.input_root, args.output_root, args.overwrite, stats)
    if "mdb" in datasets:
        convert_mdb(args.input_root, args.output_root, args.overwrite, stats)
    if "idmt" in datasets:
        convert_idmt(args.input_root, args.output_root, args.overwrite, stats)

    write_summary(args.output_root, stats)

    print("Conversion finished.")
    print(f"Target audio format: {TARGET_CHANNELS} channel, {TARGET_SAMPLE_RATE} Hz, FLAC")
    for dataset_name in sorted(stats.dataset_file_counts):
        print(
            f"{dataset_name}: files={stats.dataset_file_counts[dataset_name]}, "
            f"events={stats.dataset_event_counts[dataset_name]}"
        )
    if stats.ignored_label_counts:
        print("Ignored/unmapped labels:")
        for label, count in sorted(stats.ignored_label_counts.items()):
            print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
