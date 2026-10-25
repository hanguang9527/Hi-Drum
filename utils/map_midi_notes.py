import os
import sys
import shutil
import mido

# ================================================================================
# MIDI Note Mapper (Copy & Process)
# ================================================================================
# This script copies the dataset to a new location and then modifies the MIDI files.
# ================================================================================

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------

#Input folder path (original data)
SOURCE_DIRECTORY = "data/validation_label"

# Output the parent directory (the processed folder will be placed here)
# Defaults to the current directory "."
OUTPUT_PARENT_DIRECTORY = "." 

# Output folder suffix
# For example: If the source folder is "demo_dataset" and the suffix is ​​"_processed",
#The generated new folder is named "demo_dataset_processed"
OUTPUT_SUFFIX = "_processed"

# Set mapping mode
# 'gm': Roland -> GM (standard GM mapping)
# '3' : 3 categories (KD, SD, HH) - corresponding to E-GMD Table 2 coarse-grained classification
# '7' : 7 categories (KD, SD, TT, HH, CY, RD, BE) - corresponding to E-GMD Table 2 fine-grained classification
MAPPING_MODE = '7' 

# [switch] Whether to classify Tambourine (54) as Hi-Hat (42)
MERGE_TAMBOURINE_TO_HH = True 

# --------------------------------------------------------------------------------
# Metadata & Definitions
# --------------------------------------------------------------------------------

DRUM_METADATA = {
    # Kick
    36: ("Bass Drum 1", "bass drum", "KD"),
    59: ("Acoustic Bass Drum", "Bass Drum (Original)", "KD"), 
    35: ("Acoustic Bass Drum", "Bass Drum (Original)", "KD"),
    
    # Snare
    38: ("Acoustic Snare", "military drum", "SD"),
    40: ("Electric Snare", "Electric Snare Drum", "SD"),
    37: ("Side Stick", "Drum Edge", "SD"),
    39: ("Hand Clap", "Clap your hands", "SD"),
    
    # Toms
    50: ("High Tom", "High Tom Drum", "TT"),
    48: ("Hi-Mid Tom", "Medium High Tom", "TT"),
    47: ("Low-Mid Tom", "Medium Low Tom", "TT"),
    45: ("Low Tom", "Low Tom Drum", "TT"),
    72: ("High Floor Tom", "High floor tom", "TT"), 
    43: ("High Floor Tom", "High floor tom", "TT"),
    41: ("Low Floor Tom", "Low tom", "TT"),
    73: ("Low Floor Tom", "Low tom", "TT"), 
    
    # Hi-Hat
    42: ("Closed Hi-Hat", "Closed cymbal", "HH"),
    46: ("Open Hi-Hat", "open cymbal", "HH"),
    44: ("Pedal Hi-Hat", "hi-hat", "HH"),
    
    # Cymbals
    49: ("Crash Cymbal 1", "Crash Cymbal 1", "CY"),
    57: ("Crash Cymbal 2", "Crash Cymbal 2", "CY"),
    52: ("Chinese Cymbal", "Chinese Cymbals", "CY"),
    55: ("Splash Cymbal", "Splash Cymbal", "CY"),
    
    # Ride
    51: ("Ride Cymbal 1", "Ring Cymbal 1", "RD"),
    59: ("Ride Cymbal 2", "Ring Cymbal 2", "RD"), 
    
    # Bell / Percussion
    53: ("Ride Bell", "cymbal hat", "BE"),
    56: ("Cowbell", "Cowbell", "BE"),
    54: ("Tambourine", "tambourine", "HH_Aux"),
}

# --------------------------------------------------------------------------------
# Mappings
# --------------------------------------------------------------------------------

ROLAND_TO_GM_MAPPING = {
    37: 38, 40: 38, 45: 47, 48: 50, 22: 42, 26: 46, 44: 42, 
    52: 49, 55: 49, 57: 49, 58: 43, 
    59: 59, 53: 53, 56: 53, 54: 54, 39: 39,
}

GM_TO_7_CLASS_MAPPING = {
    47: 50, 43: 50, 48: 50, 45: 50, 41: 50,
    46: 42, 44: 42, 26: 42, 22: 42,
    57: 49, 52: 49, 55: 49,
    51: 59,
    56: 53,
    39: 38,
}

GM_TO_3_CLASS_MAPPING = {
    35: 36,
    40: 38, 37: 38, 39: 38, 50: 38, 48: 38, 47: 38, 45: 38, 43: 38, 41: 38,
    46: 42, 44: 42, 49: 42, 57: 42, 52: 42, 55: 42, 51: 42, 59: 42, 53: 42, 56: 42, 54: 42,
}

# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------

def get_mapping(mode):
    mapping = ROLAND_TO_GM_MAPPING.copy()
    
    if mode == 'gm':
        return mapping
    elif mode == '7':
        combined = mapping.copy()
        for k, v in combined.items():
            if v in GM_TO_7_CLASS_MAPPING:
                combined[k] = GM_TO_7_CLASS_MAPPING[v]
        for k, v in GM_TO_7_CLASS_MAPPING.items():
            if k not in combined:
                combined[k] = v
        if MERGE_TAMBOURINE_TO_HH:
            combined[54] = 42
            print("Configuration: Tambourine (54) mapped to Hi-Hat (42).")
        else:
            print("Configuration: Tambourine (54) preserved as-is.")
        return combined
    elif mode == '3':
        combined = mapping.copy()
        for k, v in combined.items():
            if v in GM_TO_3_CLASS_MAPPING:
                combined[k] = GM_TO_3_CLASS_MAPPING[v]
        for k, v in GM_TO_3_CLASS_MAPPING.items():
            if k not in combined:
                combined[k] = v
        return combined
    else:
        raise ValueError(f"Unknown mode: {mode}")

def map_notes_in_midi(file_path, pitch_mapping):
    try:
        mid = mido.MidiFile(file_path)
        modified = False
        for track in mid.tracks:
            for msg in track:
                if msg.type in ['note_on', 'note_off', 'polytouch']:
                    original_note = msg.note
                    if original_note in pitch_mapping:
                        new_note = pitch_mapping[original_note]
                        if new_note != original_note:
                            msg.note = new_note
                            modified = True
        if modified:
            mid.save(file_path)
            # print(f"Processed: {file_path}") 
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def prepare_output_directory(source_dir, parent_dir, suffix):
    """
    Copies source_dir to parent_dir/source_basename + suffix.
    Returns the new directory path.
    """
    source_abs = os.path.abspath(source_dir)
    source_basename = os.path.basename(source_abs)
    if not source_basename: # Handle ending slash
        source_basename = os.path.basename(os.path.dirname(source_abs))
        
    new_dirname = f"{source_basename}{suffix}"
    output_path = os.path.join(parent_dir, new_dirname)
    output_abs = os.path.abspath(output_path)
    
    if os.path.exists(output_abs):
        print(f"Warning: Output directory already exists: {output_abs}")
        print("Cleaning up existing directory...")
        shutil.rmtree(output_abs)
    
    print(f"Copying {source_abs} -> {output_abs} ...")
    shutil.copytree(source_abs, output_abs)
    return output_abs

def process_directory(directory, mode):
    print(f"Configuration Mode: {mode}")
    mapping = get_mapping(mode)
    
    print("\nMapping Rules Preview (Input -> Output):")
    examples = [36, 38, 48, 46, 49, 51, 53, 54] 
    for ex in examples:
        target = mapping.get(ex, ex)
        meta_target = DRUM_METADATA.get(target, ("Unknown", "unknown", "UNK"))
        meta_source = DRUM_METADATA.get(ex, ("Unknown", "unknown", "UNK"))
        print(f"  Note {ex} ({meta_source[1]}) -> {target} ({meta_target[2]}: {meta_target[1]})")
        
    print(f"\nScanning directory: {directory} ...")
    
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".mid", ".midi")):
                file_path = os.path.join(root, file)
                map_notes_in_midi(file_path, mapping)
                count += 1
                
    print(f"Complete. Processed {count} MIDI files.")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    print("----------------------------------------------------------------")
    print("Drum MIDI Note Mapper (Copy & Process)")
    print(f"Source Directory: {SOURCE_DIRECTORY}")
    print(f"Output Parent: {OUTPUT_PARENT_DIRECTORY}")
    print(f"Output Suffix: {OUTPUT_SUFFIX}")
    print(f"Mapping Mode: {MAPPING_MODE}")
    print("----------------------------------------------------------------")
    
    if not os.path.exists(SOURCE_DIRECTORY):
        print(f"Error: Source directory not found: {SOURCE_DIRECTORY}")
        sys.exit(1)

    # 1. Copy Directory
    target_directory = prepare_output_directory(SOURCE_DIRECTORY, OUTPUT_PARENT_DIRECTORY, OUTPUT_SUFFIX)
    
    # 2. Process the copied directory
    process_directory(target_directory, MAPPING_MODE)
    
    print("----------------------------------------------------------------")
    print(f"All operations completed.")
    print(f"Result is located at: {target_directory}")
