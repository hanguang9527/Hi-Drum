import os
import shutil
import time
import numpy as np
from datetime import timedelta
import warnings
from pydub import AudioSegment
import csv
import pandas as pd
import pretty_midi
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import io

# Suppress pydub ffmpeg warnings (since we use soundfile to save FLAC, no ffmpeg dependency)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

# ===================== Configuration Parameters (Adjust as needed) =====================
# ====== Audio Processing Parameters ======
TARGET_SAMPLE_RATE = 16000  # Target sample rate for downsampling (Hz)
MIN_ENHANCE_DURATION = 10   # Minimum duration for short audio enhancement (seconds)
SHORT_AUDIO_THRESHOLD = 5   # Threshold to identify short audio (seconds)
ENABLE_SHORT_AUDIO_ENHANCEMENT = True  # Whether to enable short audio enhancement
SUPPORTED_AUDIO_EXT = ('.wav', '.WAV', '.flac', '.FLAC')
SUPPORTED_MIDI_EXT = ('.mid', '.midi', '.MID', '.MIDI')

# ====== Input/Output Paths ======
SOURCE_DIR = "./data/groove"       # Source dataset root directory
OUTPUT_PARENT_DIR = "./data"       # Output parent directory
DATASET_NAME = "groove_processed_5_10_16khz"      # Output dataset name

# ====== File Format Parameters ======
FLAC_COMPRESSION_LEVEL = 5  # FLAC compression level (0-8, higher means more compression)
CSV_ENCODING = 'utf-8'      # CSV file encoding
TSV_ENCODING = 'utf-8'      # TSV file encoding

# ====== Processing Options ======
ENABLE_LONG_AUDIO_PROCESSING = True    # Whether to enable long audio processing
ENABLE_MIDI_TSV_GENERATION = True      # Whether to generate TSV annotation files
PRESERVE_DIRECTORY_STRUCTURE = True  # Whether to preserve source directory structure

# ===================== Utility Functions =====================
def _label_dir_from_data_dir(data_dir: str) -> str:
    """Map .../train_data -> .../train_label without touching other path segments."""
    data_path = Path(data_dir)
    if data_path.name.endswith('_data'):
        return str(data_path.with_name(data_path.name.replace('_data', '_label')))
    return data_dir.replace('_data', '_label')

def get_audio_duration(audio_path: str) -> float:
    """Get actual duration of audio file (seconds), compatible with multiple formats"""
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception as e:
        print(f"⚠️ Failed to read duration for {audio_path}: {str(e)}")
        return 0.0

def process_midi(midi_path: str, num_repeats: int, duration_per_repeat: float) -> pretty_midi.PrettyMIDI | None:
    """
    Expand MIDI file: repeat note events according to num_repeats, adjust timestamps
    :param midi_path: Original MIDI path
    :param num_repeats: Number of repeats
    :param duration_per_repeat: Duration per repeat (seconds)
    :return: Expanded MIDI object, None if failed
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        new_midi = pretty_midi.PrettyMIDI()
        
        # Create empty instrument tracks first (keep program and drum attributes consistent)
        for instr in midi.instruments:
            new_instr = pretty_midi.Instrument(
                program=instr.program, 
                is_drum=instr.is_drum,
                name=instr.name
            )
            new_midi.instruments.append(new_instr)
        
        # Repeatedly add notes (match by track)
        for repeat_idx in range(num_repeats):
            offset = repeat_idx * duration_per_repeat
            for instr_idx, instr in enumerate(midi.instruments):
                for note in instr.notes:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + offset,
                        end=note.end + offset
                    )
                    new_midi.instruments[instr_idx].notes.append(new_note)
        
        return new_midi
    except Exception as e:
        print(f"❌ Failed to process MIDI {midi_path}: {e}")
        return None

def preprocess_audio(audio_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> AudioSegment | None:
    """
    Audio preprocessing: convert to mono and downsample
    :param audio_path: Input audio path
    :param target_sr: Target sample rate
    :return: Processed AudioSegment object, None if failed
    """
    try:
        # Read audio and force mono (fix multi-channel issues)
        audio = AudioSegment.from_file(audio_path).set_channels(1)
        # Downsample
        audio = audio.set_frame_rate(target_sr)
        return audio
    except Exception as e:
        print(f"❌ Failed to preprocess audio {audio_path}: {e}")
        return None

def save_audio_as_flac(audio: AudioSegment, flac_path: str) -> bool:
    """
    Save AudioSegment object as FLAC format (no ffmpeg dependency)
    :param audio: AudioSegment object
    :param flac_path: Output FLAC path
    :return: True if successful
    """
    try:
        # Convert AudioSegment to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Reshape if stereo
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        # Convert to float32 and normalize to [-1, 1]
        samples = samples.astype(np.float32)
        if samples.max() > 0:
            samples = samples / np.abs(samples).max()
        
        # Use soundfile to save as FLAC format
        sf.write(flac_path, samples, audio.frame_rate, format='FLAC')
        return True
    except Exception as e:
        print(f"❌ Failed to save audio as FLAC {flac_path}: {e}")
        return False

def wav_to_flac_downsample(wav_path: str, flac_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> bool:
    """
    Convert audio to mono FLAC and downsample (maintain backward compatibility)
    :param wav_path: Input audio path
    :param flac_path: Output FLAC path
    :param target_sr: Target sample rate
    :return: True if successful
    """
    audio = preprocess_audio(wav_path, target_sr)
    if audio is None:
        return False
    return save_audio_as_flac(audio, flac_path)

def midi_to_tsv(midi_obj: pretty_midi.PrettyMIDI, tsv_path: str) -> bool:
    """
    Generate TSV annotation file from MIDI object (onset, offset, note, velocity)
    :param midi_obj: PrettyMIDI object
    :param tsv_path: Output TSV path
    :return: True if successful
    """
    if not ENABLE_MIDI_TSV_GENERATION:
        return True  # Return success directly if TSV generation is disabled
        
    try:
        tsv_rows = []
        for instr in midi_obj.instruments:
            # Skip empty tracks
            if not instr.notes:
                continue
            # Organize note data (sort by onset)
            for note in sorted(instr.notes, key=lambda x: x.start):
                tsv_rows.append({
                    "onset": round(note.start, 6),
                    "offset": round(note.end, 6),
                    "note": note.pitch,
                    "velocity": note.velocity
                })
        # Write to TSV (no index, tab-separated)
        pd.DataFrame(tsv_rows).to_csv(tsv_path, sep='\t', index=False, encoding=TSV_ENCODING)
        return True
    except Exception as e:
        print(f"❌ Failed to convert MIDI to TSV: {e}")
        return False

# ===================== Core Processing Functions =====================
def process_short_audio(wav_path: str, midi_path: str, output_audio_dir: str, output_label_dir: str) -> tuple:
    """
    Process short audio: preprocess (mono + downsample), then concatenate to enhance to >= MIN_ENHANCE_DURATION seconds, synchronously process MIDI/TSV/FLAC
    :param wav_path: Original WAV path
    :param midi_path: Original MIDI path
    :param output_audio_dir: Output audio directory
    :param output_label_dir: Output label directory
    :return: (Enhanced WAV path, FLAC path, TSV path, Enhanced MIDI path)
    """
    base_name = Path(wav_path).stem
    original_duration = get_audio_duration(wav_path)
    
    # Calculate required repetitions (ensure total duration >= MIN_ENHANCE_DURATION)
    num_repeats = int(np.ceil(MIN_ENHANCE_DURATION / original_duration))
    ext_suffix = f"_ext{original_duration*num_repeats:.1f}s"
    enhanced_base = f"{base_name}{ext_suffix}"

    # 1. Preprocess audio (mono + downsample)
    processed_audio = preprocess_audio(wav_path, TARGET_SAMPLE_RATE)
    if processed_audio is None:
        print(f"❌ Audio preprocessing failed, skipping {wav_path}")
        return None, None, None, None

    # 2. Concatenate and enhance audio
    concatenated_audio = processed_audio * num_repeats  # Simple concatenation (can be replaced with crossfade)
    enhanced_wav_path = os.path.join(output_audio_dir, f"{enhanced_base}.wav")
    concatenated_audio.export(enhanced_wav_path, format="wav")

    # 3. Process MIDI (synchronous expansion)
    enhanced_midi_path = None
    processed_midi = None
    if os.path.exists(midi_path) if midi_path else False:
        processed_midi = process_midi(midi_path, num_repeats, original_duration)
        if processed_midi:
            enhanced_midi_path = os.path.join(output_label_dir, f"{enhanced_base}.midi")
            processed_midi.write(enhanced_midi_path)
        else:
            print(f"⚠️ MIDI processing failed, skipping {midi_path}")
    else:
        print(f"⚠️ MIDI file does not exist {midi_path}")

    # 4. Save as FLAC (preprocessed audio is already in correct format)
    enhanced_flac_path = os.path.join(output_audio_dir, f"{enhanced_base}.flac")
    flac_success = save_audio_as_flac(concatenated_audio, enhanced_flac_path)
    if not flac_success:
        enhanced_flac_path = None

    # 5. Generate TSV annotation
    enhanced_tsv_path = None
    if processed_midi:
        enhanced_tsv_path = os.path.join(output_label_dir, f"{enhanced_base}.tsv")
        if not midi_to_tsv(processed_midi, enhanced_tsv_path):
            enhanced_tsv_path = None

    print(f"✅ Enhancement complete: {enhanced_base} (duration {original_duration*num_repeats:.1f}s)")
    return enhanced_wav_path, enhanced_flac_path, enhanced_tsv_path, enhanced_midi_path

def process_long_audio(wav_path: str, midi_path: str, output_audio_dir: str, output_label_dir: str) -> tuple:
    """
    Process long audio: preprocess (mono + downsample), then save as FLAC, synchronously process MIDI/TSV
    :param wav_path: Original WAV path
    :param midi_path: Original MIDI path
    :param output_audio_dir: Output audio directory
    :param output_label_dir: Output label directory
    :return: (Original WAV path, FLAC path, TSV path, Copied MIDI path)
    """
    base_name = Path(wav_path).stem

    # 1. Preprocess audio (mono + downsample)
    processed_audio = preprocess_audio(wav_path, TARGET_SAMPLE_RATE)
    if processed_audio is None:
        print(f"❌ Audio preprocessing failed, skipping {wav_path}")
        return None, None, None, None

    # 2. Save as FLAC
    flac_path = os.path.join(output_audio_dir, f"{base_name}.flac")
    flac_success = save_audio_as_flac(processed_audio, flac_path)
    if not flac_success:
        flac_path = None

    # 3. Process MIDI and TSV
    tsv_path = None
    midi_output_path = None
    if os.path.exists(midi_path) if midi_path else False:
        # Copy MIDI file
        midi_output_path = os.path.join(output_label_dir, Path(midi_path).name)
        shutil.copy2(midi_path, midi_output_path)
        # Generate TSV
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            tsv_path = os.path.join(output_label_dir, f"{base_name}.tsv")
            midi_to_tsv(midi, tsv_path)
        except Exception as e:
            print(f"❌ Failed to process long audio MIDI {midi_path}: {e}")
            tsv_path = None
    else:
        print(f"⚠️ MIDI file does not exist {midi_path}")

    print(f"✅ Processing complete: {base_name} (original duration {get_audio_duration(wav_path):.1f}s)")
    return wav_path, flac_path, tsv_path, midi_output_path

# ===================== Main Processing Flow =====================
def process_dataset(source_dir: str, output_parent_dir: str, dataset_name: str) -> None:
    """
    Batch process dataset: supports CSV file and auto-scan mode without CSV files
    :param source_dir: Source dataset root directory
    :param output_parent_dir: Output parent directory
    :param dataset_name: Output dataset name (subdirectory will be created)
    """
    # Initialize output directory
    output_dir = Path(output_parent_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📌 Processing configuration:")
    print(f"   Source directory: {source_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Short audio threshold: {SHORT_AUDIO_THRESHOLD}s, Min enhanced duration: {MIN_ENHANCE_DURATION}s")
    print(f"   Target sample rate: {TARGET_SAMPLE_RATE}Hz\n")

    # 1. Try to auto-detect CSV files first
    csv_files = {}
    found_csvs = [f for f in os.listdir(source_dir) if f.endswith(".csv")]
    
    for file in found_csvs:
        lower_file = file.lower()
        if "test" in lower_file:
            csv_files["test"] = file
        elif "train" in lower_file:
            csv_files["train"] = file
        elif "validation" in lower_file or "val" in lower_file:
            csv_files["validation"] = file
            
    # Fallback: If no standard splits found but CSVs exist, process all
    if not csv_files and found_csvs:
        print(f"⚠️ Standard splits (train/test/val) not found, will process all discovered CSV files")
        for f in found_csvs:
            csv_files[Path(f).stem] = f

    # If CSV files found, use CSV mode
    if csv_files:
        print(f"📄 CSV files found, using CSV mode")
        _process_dataset_csv_mode(source_dir, output_dir, csv_files)
    else:
        # If no CSV files, use auto-scan mode
        print(f"🔍 No CSV files found, using auto-scan mode")
        _process_dataset_auto_scan_mode(source_dir, output_dir)

def _process_dataset_csv_mode(source_dir: str, output_dir: Path, csv_files: dict) -> None:
    """Process dataset using CSV mode"""
    # Count total entries to process
    total_entries = 0
    for csv_name in csv_files.values():
        csv_path = os.path.join(source_dir, csv_name)
        try:
            with open(csv_path, 'r', encoding=CSV_ENCODING) as f:
                total_entries += sum(1 for _ in csv.DictReader(f))
        except Exception as e:
            print(f"⚠️ Failed to read {csv_name}: {e}, skipping file")
            del csv_files[[k for k, v in csv_files.items() if v == csv_name][0]]

    if total_entries == 0:
        print("❌ Error: No valid data entries to process")
        return

    # Initialize progress bar and timer
    overall_progress = tqdm(total=total_entries, unit="items", desc="📊 Overall Progress")
    start_time = time.time()

    # Process each subset
    for split, csv_name in csv_files.items():
        csv_path = os.path.join(source_dir, csv_name)
        output_csv_path = output_dir / csv_name

        # Read CSV and validate fields
        with open(csv_path, 'r', encoding=CSV_ENCODING) as infile:
            reader = csv.DictReader(infile)
            required_fields = ['audio_filename', 'midi_filename']
            if not all(f in reader.fieldnames for f in required_fields):
                print(f"❌ Error: {csv_name} missing required fields {required_fields}, skipping")
                overall_progress.update(sum(1 for _ in reader))
                continue

            # Write new CSV
            with open(output_csv_path, 'w', encoding=CSV_ENCODING) as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()

                # Process data row by row
                for row in tqdm(reader, desc=f"🔧 Processing {split} set", unit="items"):
                    _process_single_row(row, source_dir, output_dir, overall_progress)

        print(f"✅ {split} set processing complete, output CSV: {output_csv_path}")

    # Output processing summary
    overall_elapsed = time.time() - start_time
    overall_progress.close()
    print(f"\n🎉 All processing complete!")
    print(f"   Total time: {str(timedelta(seconds=int(overall_elapsed)))}")
    print(f"   Total items processed: {total_entries}")
    print(f"   Output directory: {output_dir}")

def _process_dataset_auto_scan_mode(source_dir: str, output_dir: Path) -> None:
    """Process dataset using auto-scan mode"""
    # Scan all supported audio files
    audio_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(SUPPORTED_AUDIO_EXT):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print("❌ Error: No audio files found")
        return
    
    print(f"🎵 Found {len(audio_files)} audio files")
    
    # Initialize progress bar and timer
    overall_progress = tqdm(total=len(audio_files), unit="files", desc="📊 Overall Progress")
    start_time = time.time()
    processed_count = 0
    
    # Process each audio file
    for audio_path in audio_files:
        try:
            # Find corresponding MIDI file
            base_name = Path(audio_path).stem
            midi_path = None
            
            # Search for corresponding MIDI file in the same directory and subdirectories
            audio_dir = os.path.dirname(audio_path)
            possible_midi_names = [f"{base_name}.mid", f"{base_name}.midi", 
                                 f"{base_name}.MID", f"{base_name}.MIDI"]
            
            for midi_name in possible_midi_names:
                test_path = os.path.join(audio_dir, midi_name)
                if os.path.exists(test_path):
                    midi_path = test_path
                    break
                
                if Path(audio_dir).name.endswith('_data'):
                    label_dir = _label_dir_from_data_dir(audio_dir)
                    test_path = os.path.join(label_dir, midi_name)
                    if os.path.exists(test_path):
                        midi_path = test_path
                        break
            
            if midi_path is None:
                print(f"⚠️ No MIDI found for {audio_path}, skipping")
                overall_progress.update(1)
                continue

            # Create output directory structure
            rel_path = os.path.relpath(audio_path, source_dir)
            output_audio_dir = output_dir / os.path.dirname(rel_path)
            output_audio_dir.mkdir(parents=True, exist_ok=True)
            output_label_dir = output_dir / os.path.dirname(rel_path).replace('_data', '_label')
            output_label_dir.mkdir(parents=True, exist_ok=True)
            
            # Branch processing by audio duration
            audio_duration = get_audio_duration(audio_path)
            if audio_duration < SHORT_AUDIO_THRESHOLD and ENABLE_SHORT_AUDIO_ENHANCEMENT:
                # Short audio: enhancement processing
                _, flac_path, _, midi_output_path = process_short_audio(
                    wav_path=audio_path,
                    midi_path=midi_path,
                    output_audio_dir=str(output_audio_dir),
                    output_label_dir=str(output_label_dir)
                )
            else:
                # Long audio: direct conversion
                _, flac_path, _, midi_output_path = process_long_audio(
                    wav_path=audio_path,
                    midi_path=midi_path,
                    output_audio_dir=str(output_audio_dir),
                    output_label_dir=str(output_label_dir)
                )
            
            processed_count += 1
            overall_progress.update(1)
            
        except Exception as e:
            print(f"⚠️ Failed to process {audio_path}: {e}")
            overall_progress.update(1)
    
    # Output processing summary
    overall_elapsed = time.time() - start_time
    overall_progress.close()
    print(f"\n🎉 All processing complete!")
    print(f"   Total time: {str(timedelta(seconds=int(overall_elapsed)))}")
    print(f"   Total files processed: {processed_count}")
    print(f"   Output directory: {output_dir}")

def _process_single_row(row: dict, source_dir: str, output_dir: Path, progress_bar) -> None:
    """Process a single row of CSV data"""
    # Get absolute paths for audio and MIDI
    audio_rel_path = row['audio_filename'].replace('\\', '/')
    audio_abs_path = os.path.join(source_dir, audio_rel_path)
    midi_rel_path = row['midi_filename'].replace('\\', '/')
    midi_abs_path = os.path.join(source_dir, midi_rel_path)

    # Automatically fix paths: adapt to partitioned dataset structure (egmd/train_data/...)
    if not os.path.exists(audio_abs_path) and row.get('split'):
        dataset_split = row['split']
        # Try to find in egmd/{split}_data/
        alt_audio_path = os.path.join(source_dir, 'egmd', f"{dataset_split}_data", os.path.basename(audio_rel_path))
        if os.path.exists(alt_audio_path):
            audio_abs_path = alt_audio_path
        
        # Try to find in egmd/{split}_label/
        alt_midi_path = os.path.join(source_dir, 'egmd', f"{dataset_split}_label", os.path.basename(midi_rel_path))
        if os.path.exists(alt_midi_path):
            midi_abs_path = alt_midi_path

    # Validate files
    if not os.path.exists(audio_abs_path) or not audio_abs_path.endswith(SUPPORTED_AUDIO_EXT):
        print(f"⚠️ Invalid WAV file: {audio_abs_path}, skipping")
        progress_bar.update(1)
        return
    if not os.path.exists(midi_abs_path) or not midi_abs_path.endswith(SUPPORTED_MIDI_EXT):
        print(f"⚠️ Invalid MIDI file: {midi_abs_path}, skipping")
        progress_bar.update(1)
        return

    # Create output subdirectories (preserve source directory structure)
    audio_dir = Path(audio_rel_path).parent
    midi_dir = Path(midi_rel_path).parent
    output_audio_dir = output_dir / audio_dir
    output_midi_dir = output_dir / midi_dir
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_midi_dir.mkdir(parents=True, exist_ok=True)

    # Branch processing by audio duration
    audio_duration = get_audio_duration(audio_abs_path)
    if audio_duration < SHORT_AUDIO_THRESHOLD:
        # Short audio: enhancement processing
        _, flac_path, _, midi_output_path = process_short_audio(
            wav_path=audio_abs_path,
            midi_path=midi_abs_path,
            output_audio_dir=str(output_audio_dir),
            output_label_dir=str(output_midi_dir)
        )
    else:
        # Long audio: direct conversion
        _, flac_path, _, midi_output_path = process_long_audio(
            wav_path=audio_abs_path,
            midi_path=midi_abs_path,
            output_audio_dir=str(output_audio_dir),
            output_label_dir=str(output_midi_dir)
        )

    # Update paths in CSV (relative to output directory)
    if flac_path:
        row['audio_filename'] = os.path.relpath(flac_path, output_dir)
    if midi_output_path:
        row['midi_filename'] = os.path.relpath(midi_output_path, output_dir)
    progress_bar.update(1)

def cleanup_wav_files(output_dir: str) -> None:
    """
    Clean up all WAV files in the output directory, keep only FLAC files
    :param output_dir: Output directory path
    """
    wav_count = 0
    output_path = Path(output_dir)
    
    print(f"🧹 Starting to clean up WAV files in output directory...")
    
    # Traverse all files in the output directory
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_path = os.path.join(root, file)
                try:
                    os.remove(wav_path)
                    wav_count += 1
                    if wav_count <= 5:  # Only show first 5 deleted files
                        print(f"   🗑️ Deleted: {os.path.relpath(wav_path, output_dir)}")
                except Exception as e:
                    print(f"   ⚠️ Failed to delete {wav_path}: {e}")
    
    if wav_count > 5:
        print(f"   ... and {wav_count - 5} more WAV files deleted")
    
    print(f"✅ WAV file cleanup complete, deleted {wav_count} files in total")

# ===================== Execution Entry Point =====================
if __name__ == "__main__":
    # Show current configuration
    print("📋 Current Processing Configuration:")
    print(f"   Source directory: {SOURCE_DIR}")
    print(f"   Output directory: {OUTPUT_PARENT_DIR}")
    print(f"   Dataset name: {DATASET_NAME}")
    print(f"   Target sample rate: {TARGET_SAMPLE_RATE}Hz")
    print(f"   Short audio threshold: {SHORT_AUDIO_THRESHOLD}s")
    print(f"   Min enhanced duration: {MIN_ENHANCE_DURATION}s")
    print(f"   FLAC compression level: {FLAC_COMPRESSION_LEVEL}")
    print(f"   Short audio enhancement: {'Enabled' if ENABLE_SHORT_AUDIO_ENHANCEMENT else 'Disabled'}")
    print(f"   Long audio processing: {'Enabled' if ENABLE_LONG_AUDIO_PROCESSING else 'Disabled'}")
    print(f"   TSV generation: {'Enabled' if ENABLE_MIDI_TSV_GENERATION else 'Disabled'}")
    print(f"   Preserve directory structure: {'Yes' if PRESERVE_DIRECTORY_STRUCTURE else 'No'}")
    print()

    # Validate source directory
    if not os.path.isdir(SOURCE_DIR):
        print(f"❌ Error: Source directory does not exist {SOURCE_DIR}")
        print("💡 Please change SOURCE_DIR parameter to the correct source dataset path")
    else:
        output_dir = Path(OUTPUT_PARENT_DIR) / DATASET_NAME
        process_dataset(SOURCE_DIR, OUTPUT_PARENT_DIR, DATASET_NAME)
        
        # Clean up WAV files after processing
        if output_dir.exists():
            cleanup_wav_files(str(output_dir))
        else:
            print("⚠️ Output directory does not exist, skipping WAV file cleanup")
