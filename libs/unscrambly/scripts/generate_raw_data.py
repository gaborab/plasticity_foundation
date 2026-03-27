"""
Generate MNE Raw objects from EEG data.

This script:
1. Preprocesses EEG data from multiple subjects/conditions
2. Returns a dictionary of Raw objects organized by subject and condition
3. Assigns the result to a variable for later use

Output structure: {subject_name: {condition_name: mne.io.Raw}}
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from preprocessing.preprocessing import DataPreprocessor

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data path (root directory containing subject folders)
DATA_PATH = None  # NOTE: configure this path

# Preprocessing parameters
CHANNELS = ['EEG 20', 'EEG 21', 'EEG 41', 'EEG 50']  # Set to None or [] for all channels
FILTER_PARAMS = {'lowpass': 30, 'highpass': 0.1, 'notch': 50}  # Set to None to disable filtering
BASELINE_WIN = (-0.2, 0)  # Baseline window in seconds (used for epoch time extension)
TIME_WINDOW = (-0.1, 1)  # Epoch time window (needed for preprocessing config)

# Re-referencing: 'average', 'laplacian', or None
RE_REFERENCE = 'average'

# Bad channel handling: 'interpolate', 'exclude', or None
BAD_CHANNELS = 'interpolate'
# If set to 'interpolate', a manual selection will be used which I previously assigned to each recording 

# ICA settings
ICA_ENABLED = False
ICA_N_COMPONENTS = 15  # Selections were made for 15; if other value chosen, bad ICA components will need to be re-specified

# Epoch split mode: 'sides', 'eccentricity', or 'both'
# 'sides'        -> keys: left, center, right, global
# 'eccentricity' -> keys: ecc_1, ecc_2, ... (excludes global & ecc=0)
# 'both'         -> keys: left_1, left_2, ..., global
SPLIT_BY = 'sides'

# Trigger type: 'onset' or 'offset'
# 'onset'  -> stimulus onset triggers
# 'offset' -> stimulus offset triggers
TRIGGER_TYPE = 'onset'

# Conditions to include (must match folder names in DATA_PATH/subject_folders/)
# Set to None to include all conditions
INCLUDED_CONDITIONS = ['baseline', 'post-patching']

# Subjects to exclude (set to None or [] to include all)
EXCLUDED_SUBJECTS = ['STMD2', 'STMD3', 'STMD4', 'STMD5', 'STMD7', 'STMD8', 'STMD10', 'STMD16', 'STMD17', 'STMD20_Gabi', 'STMD22', 'STMD23', 'STMD24', 'STMD26', 'STMD30']

# Number of recordings per condition (None for all)
# For baseline: uses LAST x recordings; for post-patching: uses FIRST x recordings
RECORDINGS_PER_CONDITION = None

# Epoch rejection measures
AUTOREJECT_EPOCH_REJECTION = True # Autoreject epoch rejection based on dynamic threshold
REJECT_AMPLITUDE_UV = 120  # Manual; Peak-to-peak amplitude threshold in µV; ignored if autoreject enabled


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_raw_data():
    """
    Generate preprocessed Raw objects for all subjects and conditions.
    
    Returns:
        dict: Nested dictionary with structure {subject_name: {condition_name: mne.io.Raw}}
    """
    print("=" * 70)
    print("RAW DATA GENERATION PIPELINE")
    print("=" * 70)
    
    print(f"\nData path: {DATA_PATH}")
    print(f"Included conditions: {INCLUDED_CONDITIONS}")
    print(f"Excluded subjects: {len(EXCLUDED_SUBJECTS) if EXCLUDED_SUBJECTS else 0}")
    
    # Initialize preprocessor
    print("\n[STEP 1] Initializing DataPreprocessor...")
    preprocessor = DataPreprocessor(
        directory=DATA_PATH,
        channels=CHANNELS,
        re_reference=RE_REFERENCE,
        bad_channels=BAD_CHANNELS,
        ica_enabled=ICA_ENABLED,
        ica_n_components=ICA_N_COMPONENTS,
        filter_params=FILTER_PARAMS,
        baseline_win=BASELINE_WIN,
        reject_amplitude_uv=REJECT_AMPLITUDE_UV,
        split_by=SPLIT_BY,
        trigger_type=TRIGGER_TYPE,
        time_window=TIME_WINDOW,
        use_global_autoreject=AUTOREJECT_EPOCH_REJECTION
    )
    
    # Process subjects to get Raw objects
    print("\n[STEP 2] Processing subjects...")
    raw_data = preprocessor.process_subjects(
        excluded_subjects=EXCLUDED_SUBJECTS, 
        included_conditions=INCLUDED_CONDITIONS,
        recordings_per_condition=RECORDINGS_PER_CONDITION
    )
    
    if not raw_data:
        print("ERROR: No data found. Check your data path and folder structure.")
        return None
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total subjects processed: {len(raw_data)}")
    for subject, conditions in raw_data.items():
        print(f"  {subject}:")
        for condition, raw in conditions.items():
            n_channels = len(raw.ch_names)
            duration = raw.times[-1]
            print(f"    {condition}: {n_channels} channels, {duration:.1f}s duration")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    return raw_data


# Variable to store the result for later use
raw_data = None

if __name__ == '__main__':
    raw_data = generate_raw_data()
    
    # The raw_data variable is now available for further analysis
    # Structure: {subject_name: {condition_name: mne.io.Raw}}
    print("\nData stored in 'raw_data' variable.")
    print("Access example: raw_data['STMD1']['baseline']")

    # print dictionary structure
    if raw_data:
        print("\n" + "=" * 70)
        print("DICTIONARY STRUCTURE")
        print("=" * 70)
        print(f"raw_data  ({len(raw_data)} subjects)")
        for subj, conditions in raw_data.items():
            print(f"  └─ '{subj}'  ({len(conditions)} conditions)")
            for cond, triggers in conditions.items():
                print(f"      └─ '{cond}'  ({len(triggers)} keys)")
                for key, data in triggers.items():
                    n = len(data) if hasattr(data, '__len__') else data.shape[1]
                    print(f"          └─ '{key}'  ({n} raws)")
