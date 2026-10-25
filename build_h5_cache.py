#!/usr/bin/env python3
"""Build missing h5_hop cache files for Groove-format datasets."""

import argparse
import os
import time

from onsets_and_frames.constants import HOP_LENGTH
from onsets_and_frames.dataset import Groove


def count_h5(root):
    h5_dir = os.path.join(root, f'h5_hop{HOP_LENGTH}')
    if not os.path.isdir(h5_dir):
        return 0
    total = 0
    for _, _, files in os.walk(h5_dir):
        total += sum(1 for name in files if name.endswith('.h5'))
    return total


def count_flac(root):
    total = 0
    for group in Groove.available_groups():
        audio_dir = os.path.join(root, f'{group}_data')
        if os.path.isdir(audio_dir):
            total += sum(1 for name in os.listdir(audio_dir) if name.endswith('.flac'))
    return total


def build_h5(root, groups):
    print(f'\n=== {root} ===')
    before_h5 = count_h5(root)
    before_flac = count_flac(root)
    print(f'Before: {before_h5}/{before_flac} h5 files')
    start = time.time()
    Groove(path=root, groups=groups)
    after_h5 = count_h5(root)
    elapsed = time.time() - start
    print(f'After:  {after_h5}/{before_flac} h5 files ({elapsed:.1f}s)')
    if after_h5 < before_flac:
        print(f'Warning: still missing {before_flac - after_h5} h5 files')
    return after_h5, before_flac


def main():
    parser = argparse.ArgumentParser(description='Build h5_hop cache for Groove datasets')
    parser.add_argument(
        '--groove-path',
        default='/data1/weiyangding/data/drum/processed/groove_processed_5_10_24khz_7class',
    )
    parser.add_argument(
        '--egmd-path',
        default='/data1/weiyangding/data/drum/processed/egmd_processed_5_10_24khz_7class',
    )
    parser.add_argument('--groups', default='train,validation,test')
    parser.add_argument('--only', choices=['groove', 'egmd', 'both'], default='both')
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(',') if g.strip()]
    paths = []
    if args.only in ('groove', 'both'):
        paths.append(args.groove_path)
    if args.only in ('egmd', 'both'):
        paths.append(args.egmd_path)

    summary = []
    for path in paths:
        summary.append((path, *build_h5(path, groups)))

    print('\n=== Summary ===')
    for path, h5_count, flac_count in summary:
        status = 'OK' if h5_count >= flac_count else f'missing {flac_count - h5_count}'
        print(f'{path}: {h5_count}/{flac_count} ({status})')


if __name__ == '__main__':
    main()
