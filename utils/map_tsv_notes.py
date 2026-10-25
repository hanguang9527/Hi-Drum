import os
import pandas as pd
import sys

# ================================================================================
# Difference description (Unified vs 7-Class Script)
# ================================================================================
# About the only logical difference between map_tsv_notes_7_class.py and this file:
#
# 1. Tambourine (Tambourine, Note 54):
# - Old (7_class.py): Force mapping of tambourine to Hi-Hat (Note 42).
# - This file (Unified): Leaves the tambourine as Note 54 by default (or as is according to Roland mapping).
#
# To resolve this difference, use the MERGE_TAMBOURINE_TO_HH switch below:
# - Set to True: emulate legacy script behavior, change 54 -> 42.
# - Set to False: Keep the default behavior of this file, leaving 54.
# ================================================================================

# (3-class)
# - Kick (KD) -> You originally represented it with 36. -> New code retains 36.
# - Snare (SD) -> You originally represented it with 38. -> New code retains 38.
# - Hi-Hat (HH) -> You originally represented it with 42. -> New code retains 42.
# 7-class
# - Kick (KD) -> You originally represented it with 36. -> New code retains 36.
# - Snare (SD) -> You originally represented it with 38. -> New code retains 38.
# - Toms (TT) -> You originally represented 50. -> New code retains 50 .
# - Hi-Hat (HH) -> You originally represented it with 42. -> New code retains 42.
# - Cymbal (CY) -> You originally represented it with 49. -> New code retains 49.
# - Ride (RD) -> You originally represented it with 59. -> New code retains 59.
# - Bell (BE) -> You were originally represented by 53. -> New code retains 53.

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------

#Set the target folder path
TARGET_DIRECTORY = "data/groove_processed_10s_24kh7class"

# Set mapping mode
# 'gm': Roland -> GM (standard GM mapping)
# '3' : 3 categories (KD, SD, HH) - corresponding to E-GMD Table 2 coarse-grained classification
# '7' : 7 categories (KD, SD, TT, HH, CY, RD, BE) - corresponding to E-GMD Table 2 fine-grained classification
MAPPING_MODE = '7' 

# [New switch] Whether to classify Tambourine (54) as Hi-Hat (42)
# True = emulates the behavior of map_tsv_notes_7_class.py
# False = keep tambourine independent (Unified default behavior)
MERGE_TAMBOURINE_TO_HH = True 

# --------------------------------------------------------------------------------
# Metadata & Definitions
# --------------------------------------------------------------------------------

# Drum instrument metadata (Note Metadata)
# Format: Note: (English Name, Chinese Name, Abbreviation)
DRUM_METADATA = {
    # Kick
    36: ("Bass Drum 1", "bass drum", "KD"),
    35: ("Acoustic Bass Drum", "Bass Drum (Original)", "KD"),
    
    # Snare
    38: ("Acoustic Snare", "military drum", "SD"),
    40: ("Electric Snare", "Electric Snare Drum", "SD"),
    37: ("Side Stick", "Drum Edge", "SD"),
    39: ("Hand Clap", "Clap your hands", "SD"),
    
    # Toms (Standard GM)
    50: ("High Tom", "High Tom Drum", "TT"),
    48: ("Hi-Mid Tom", "Medium High Tom", "TT"),
    47: ("Low-Mid Tom", "Medium Low Tom", "TT"),
    45: ("Low Tom", "Low Tom Drum", "TT"),
    43: ("High Floor Tom", "High floor tom", "TT"),
    41: ("Low Floor Tom", "Low tom", "TT"),
    
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
    54: ("Tambourine", "tambourine", "HH_Aux"), # In 3-class maps to HH
}

# --------------------------------------------------------------------------------
# Mappings
# --------------------------------------------------------------------------------

# 1. Roland TD-11 to Standard GM (Corrected & Preserved)
ROLAND_TO_GM_MAPPING = {
    37: 38,  # Snare X-Stick -> Snare
    40: 38,  # Snare Rim -> Snare
    45: 47,  # Tom 2 -> Low-Mid Tom
    48: 50,  # Tom 1 -> High Tom
    22: 42,  # HH Closed Edge -> Closed HH
    26: 46,  # HH Open Edge -> Open HH
    44: 42,  # HH Pedal -> Closed HH (User preference from original file, usually 44)
    52: 49,  # Crash 1 Edge -> Crash 1
    55: 49,  # Crash 2 Bow -> Crash 1 (Simplified as per original)
    57: 49,  # Crash 2 Edge -> Crash 1 (Simplified as per original)
    58: 43,  # Tom 3 Rim -> High Floor Tom
    
    # Correction: The original file maps both 53 (Bell) and 59 (Ride Edge) to 51, resulting in the 7 category being unable to distinguish between BE and RD/Edge
    59: 59, 
    53: 53,
    
    # Extra mappings from 7-class script
    56: 53, # Cowbell -> Ride Bell (BE class)
    54: 54, # Tambourine -> Tambourine (Keep as is for now, controlled by switch later)
    39: 39, # Clap -> Clap (Keep as is)
}

# 2. GM to 7-Class (KD, SD, TT, HH, CY, RD, BE)
GM_TO_7_CLASS_MAPPING = {
    # Toms -> TT (50)
    47: 50, 43: 50, 48: 50, 45: 50, 41: 50,
    
    # Hi-Hats -> HH (42)
    46: 42, 44: 42, 26: 42, 22: 42,
    
    # Cymbals -> CY (49)
    57: 49, 52: 49, 55: 49,
    
    # Rides -> RD (59)
    51: 59,
    
    # Bells -> BE (53)
    56: 53,
    
    # Others
    39: 38, # Clap -> SD (Table 2: Snare group includes Clap)
}

# 3. GM to 3-Class (KD, SD, HH)
GM_TO_3_CLASS_MAPPING = {
    # Kick Group (KD) -> 36
    35: 36,
    
    # Snare Group (SD) -> 38
    40: 38, 37: 38, 39: 38,
    50: 38, 48: 38, 47: 38, 45: 38, 43: 38, 41: 38,
    
    # Hi-Hat Group (HH) -> 42
    46: 42, 44: 42,
    49: 42, 57: 42, 52: 42, 55: 42,
    51: 42, 59: 42,
    53: 42, 56: 42,
    54: 42, # Tambourine -> HH
}

# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------

def get_mapping(mode):
    """
    Constructs the final mapping dictionary based on the selected mode.
    """
    # Start with Roland -> GM base
    mapping = ROLAND_TO_GM_MAPPING.copy()
    
    if mode == 'gm':
        return mapping
        
    elif mode == '7':
        combined = mapping.copy()
        
        # 1. Update existing targets in mapping
        for k, v in combined.items():
            if v in GM_TO_7_CLASS_MAPPING:
                combined[k] = GM_TO_7_CLASS_MAPPING[v]
        
        # 2. Add GM_TO_7 mappings for keys not in Roland map
        for k, v in GM_TO_7_CLASS_MAPPING.items():
            if k not in combined:
                combined[k] = v
        
        # [Logical difference processing]
        if MERGE_TAMBOURINE_TO_HH:
            # Force mapping of 54 (Tambourine) to 42 (Hi-Hat)
            # To match the behavior of map_tsv_notes_7_class.py
            combined[54] = 42
            print("Configuration: Tambourine (54) mapped to Hi-Hat (42).")
        else:
            print("Configuration: Tambourine (54) preserved as-is.")
                
        return combined
        
    elif mode == '3':
        # Apply GM -> 3 mapping on top
        combined = mapping.copy()
        
        # 1. Update existing targets
        for k, v in combined.items():
            if v in GM_TO_3_CLASS_MAPPING:
                combined[k] = GM_TO_3_CLASS_MAPPING[v]
                
        # 2. Add GM_TO_3 mappings
        for k, v in GM_TO_3_CLASS_MAPPING.items():
            if k not in combined:
                combined[k] = v
        
        return combined
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def map_notes_in_tsv(file_path, pitch_mapping):
    """
    Reads a TSV file, maps the 'note' column, and saves back.
    """
    try:
        df = pd.read_csv(file_path, delimiter='\t')
        
        if 'note' not in df.columns:
            print(f"Warning: 'note' column not found in {file_path}. Skipping.")
            return

        # Perform mapping
        df['note'] = df['note'].replace(pitch_mapping)
        
        df.to_csv(file_path, sep='\t', index=False)
        # print(f"Processed: {file_path}") # Optional: reduce spam
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory, mode):
    """
    Walks directory and processes .tsv files with the selected mode.
    """
    print(f"Configuration Mode: {mode}")
    mapping = get_mapping(mode)
    
    print("\nMapping Rules Preview (Input -> Output):")
    # Show a few examples including the critical ones
    examples = [36, 38, 48, 46, 49, 51, 53, 54] # Added 54 for verification
    for ex in examples:
        target = mapping.get(ex, ex) # If not in map, stays same
        meta_target = DRUM_METADATA.get(target, ("Unknown", "unknown", "UNK"))
        meta_source = DRUM_METADATA.get(ex, ("Unknown", "unknown", "UNK"))
        print(f"  Note {ex} ({meta_source[1]}) -> {target} ({meta_target[2]}: {meta_target[1]})")
        
    print(f"\nScanning directory: {directory} ...")
    
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tsv"):
                file_path = os.path.join(root, file)
                map_notes_in_tsv(file_path, mapping)
                count += 1
                
    print(f"Complete. Processed {count} files.")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # Verification Check Information
    print("----------------------------------------------------------------")
    print("Drum Note Mapper Unified (with Compatibility Switch)")
    print(f"Target Directory: {TARGET_DIRECTORY}")
    print(f"Mapping Mode: {MAPPING_MODE}")
    print(f"Merge Tambourine to HH: {MERGE_TAMBOURINE_TO_HH}")
    print("----------------------------------------------------------------")
    
    if not os.path.exists(TARGET_DIRECTORY):
        print(f"Error: Directory not found: {TARGET_DIRECTORY}")
        sys.exit(1)
        
    process_directory(TARGET_DIRECTORY, MAPPING_MODE)
