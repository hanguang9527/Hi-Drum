"""
Dataset Partitioner

Function description:
This script is used to divide the original data set containing audio (WAV) and MIDI files into
Directory structure of training set (train), validation set (validation) and test set (test).

Workflow:
1. Read the configured source folder (SOURCE_DIR).
2. Automatically search for CSV record files in the folder.
3. Parse the CSV file, automatically identify the file name column and divide the label column (train/test/validation).
4. Scan the folder for all .wav and .mid/.midi files.
5. Keep only files that appear in pairs (i.e. wav and midi with the same name).
6. Based on the division information in the CSV, move the paired files to the corresponding subdirectory under the target folder (OUTPUT_DIR):
   - train_data (wav) / train_label (midi)
   - test_data (wav) / test_label (midi)
   - validation_data (wav) / validation_label (midi)

Instructions for use:
1. Modify the [Configuration Parameters] area below:
   - SOURCE_DIR: Path to the folder where the original data is located
   - OUTPUT_DIR: output result folder path (set to None to automatically create formatted_dataset in the source directory)
2. Run this script directly: python utils/dataset_partitioner.py
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ===================== Configuration parameters (Configuration) =====================

# [Required] Original data folder path
# Please change this path to the actual folder path where you store wav, mid and csv files
SOURCE_DIR = r"data/groove" 

# [Optional] Output folder path
# If set to None, the "formatted_dataset" folder will be generated at the same level or inside SOURCE_DIR by default
OUTPUT_DIR = None 

# ===================================================================

def partition_dataset():
    """
    The main function that performs the partitioning of the dataset, using the global configuration parameters defined above.
    """
    source_path = Path(SOURCE_DIR)
    
    # Check if the source directory exists
    if not source_path.exists() or str(SOURCE_DIR) == "/path/to/your/dataset":
        print(f"❌ Error: Please set the correct SOURCE_DIR. The current path does not exist: {SOURCE_DIR}")
        return

    # Determine the output directory
    if OUTPUT_DIR is None:
        output_path = source_path / "formatted_dataset"
    else:
        output_path = Path(OUTPUT_DIR)
    
    print(f"📂 source directory: {source_path}")
    print(f"📂 target directory: {output_path}")

    # 1. Find CSV files
    csv_files = list(source_path.glob("*.csv"))
    if not csv_files:
        print(f"❌ Error: in {source_path} Not found inCSVdocument")
        return
    if len(csv_files) > 1:
        print(f"⚠️ Warning: Multiple foundCSVFile, the first one is used by default:{csv_files[0].name}")
    csv_file = csv_files[0]
    
    print(f"📄 Read the divided record file:{csv_file.name}")
    try:
        # Read CSV and remove leading and trailing spaces from column names
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"❌ readCSVfail:{e}")
        return
    
    # 2. Identify key columns (file name column and partition column)
    split_col = None
    id_col = None
    
    # Guess the partition column: including keywords such as train, test, val etc.
    # Prefer exact matches to 'split'
    if 'split' in df.columns:
        split_col = 'split'
    else:
        for col in df.columns:
            # Take the first few rows of samples or all unique values ​​to judge
            try:
                unique_vals = df[col].dropna().astype(str).str.lower().unique()
                if any('train' in v for v in unique_vals) and \
                   (any('test' in v for v in unique_vals) or any('val' in v for v in unique_vals)):
                    split_col = col
                    break
            except:
                continue
            
    # If not guessed, try common column names
    if not split_col:
        for name in ['subset', 'partition', 'dataset', 'type', 'group']:
            if name in df.columns:
                split_col = name
                break
    
    if not split_col:
        print("❌ Error: Unable to automatically identify dividing columns in CSV (need to contain values ​​such as train, test, validation, etc.)")
        print(f"   CSVColumn name:{list(df.columns)}")
        return

    # Guess the ID column: use 'audio_filename' or 'midi_filename' first, followed by 'filename' etc.
    potential_id_cols = ['audio_filename', 'midi_filename', 'filename', 'file', 'id', 'name', 'stem', 'song']
    for name in potential_id_cols:
        if name in df.columns:
            id_col = name
            break
    if not id_col:
        id_col = df.columns[0] #Default first column
        
    print(f"ℹ️  use '{id_col}' As a file name column,'{split_col}' as dividing column")
    
    # 3. Scan source files and match paired files
    # wav -> data, midi/mid -> label
    wav_files = {}
    midi_files = {}
    
    supported_audio = {'.wav'}
    supported_midi = {'.mid', '.midi'}
    
    print("🔍 Recursively scan a folder and its subdirectories...")
    # Use rglob to find all files recursively
    for f in source_path.rglob('*'):
        if f.is_file():
            if f.suffix.lower() in supported_audio:
                wav_files[f.stem] = f
            elif f.suffix.lower() in supported_midi:
                midi_files[f.stem] = f
            
    # Find pairs of files (the file names are the same as stem)
    common_stems = set(wav_files.keys()) & set(midi_files.keys())
    print(f"✅ turn up {len(common_stems)} pair matching files (wav + midi)")
    
    if len(common_stems) == 0:
        print("⚠️ No paired wav and midi files found, program ended.")
        return

    # 4. Create CSV mapping table
    # Standardize the file names in CSV to stem and establish stem -> split mapping
    stem_to_split = {}
    for idx, row in df.iterrows():
        fname = str(row[id_col])
        # Remove the extension to get stem in case the CSV contains .wav or .mid
        stem = Path(fname).stem
        
        raw_split = str(row[split_col]).lower().strip()
        
        # Normalized partition name
        if 'train' in raw_split:
            target_split = 'train'
        elif 'test' in raw_split:
            target_split = 'test'
        elif 'val' in raw_split:
            target_split = 'validation'
        else:
            target_split = None
            
        if target_split:
            stem_to_split[stem] = target_split
            
    print(f"ℹ️  CSVContains {len(stem_to_split)} valid division records")

    # 5. Perform partitioning (moving files)
    #Create target directory structure
    splits = ['train', 'test', 'validation']
    subdirs = ['data', 'label'] # put wav in data, put midi in label
    
    for sp in splits:
        for sub in subdirs:
            (output_path / f"{sp}_{sub}").mkdir(parents=True, exist_ok=True)
            
    print(f"🚀 Start processing...")
    
    processed_count = 0
    skipped_count = 0
    
    for stem in tqdm(common_stems):
        if stem not in stem_to_split:
            # This file is not recorded in CSV, skip it
            skipped_count += 1
            continue
            
        split_name = stem_to_split[stem]
        
        wav_src = wav_files[stem]
        midi_src = midi_files[stem]
        
        # target path
        # train_data / xxx.wav
        wav_dst = output_path / f"{split_name}_data" / wav_src.name
        # train_label / xxx.mid
        midi_dst = output_path / f"{split_name}_label" / midi_src.name
        
        try:
            shutil.move(wav_src, wav_dst)
            shutil.move(midi_src, midi_dst)
            processed_count += 1
        except Exception as e:
            print(f"❌ Move failed {stem}: {e}")
            
    print(f"\n🎉 Processing completed!")
    print(f"   successfully processed: {processed_count} right")
    print(f"   jump over(Not hereCSVmiddle): {skipped_count} right")
    print(f"   The results are saved in: {output_path}")

if __name__ == "__main__":
    partition_dataset()
