"""
Generate MNE Epochs objects from EEG data.

This script:
1. Preprocesses EEG data from multiple subjects/conditions
2. Creates epochs organized by trigger type
3. Returns a dictionary of Epochs objects organized by subject, condition, and trigger
4. Assigns the result to a variable for later use

Output structure: {subject_name: {condition_name: {trigger_type: mne.Epochs or np.ndarray}}}
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from preprocessing.preprocessing import DataPreprocessor
import preprocessing.eye_dominance

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data path (root directory containing subject folders)
DATA_PATH = None  # NOTE: configure this path

# Eye dominance CSV file (set to None to disable eye dominance handling)
# CSV should have columns: 'subject_id' and 'eye_dominance' (values: 'left' or 'right')
EYE_DOMINANCE_CSV = str(Path(DATA_PATH) / "eye_dominance.csv")
SUBJECT_COL = 'subject_id'  # Column name for subject ID in eye dominance CSV
DOMINANCE_COL = 'eye_dominance'  # Column name for eye dominance

# Preprocessing parameters
CHANNELS = []  # Set to None or [] for all channels or channel format 'EEG 1' - 'EEG 64'
FILTER_PARAMS = {'lowpass': 30, 'highpass': 0.1, 'notch': 50}  # Set to None to disable filtering
BASELINE_WIN = (-0.2, 0)  # Baseline window in seconds
TIME_WINDOW = (-0.1, 1)  # Final epoch time window (baseline period auto-included then cropped)

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
SPLIT_BY = 'both'

# Trigger type: 'onset' or 'offset'
# 'onset'  -> stimulus onset
# 'offset' -> stimulus offset
TRIGGER_TYPE = 'offset'

# Conditions to include (must match folder names in DATA_PATH/subject_folders/)
# Set to None to include all conditions
INCLUDED_CONDITIONS = ['baseline', 'post-patching']

# Subjects to exclude (set to None or [] to include all)
EXCLUDED_SUBJECTS = ['STMD2', 'STMD3', 'STMD4', 'STMD5', 'STMD7', 'STMD8', 'STMD10', 'STMD16', 'STMD17', 'STMD20_Gabi', 'STMD22', 'STMD23', 'STMD24', 'STMD26', 'STMD30']

# Number of recordings per condition (None for all)
# For baseline: uses LAST x recordings; for post-patching: uses FIRST x recordings
RECORDINGS_PER_CONDITION = None

# Epochs output format: 'epochs' for MNE Epochs objects, 'array' for numpy arrays
# If 'array', shape will be [n_channels, n_trials, n_times]
OUTPUT_FORMAT = 'epochs'

# Baselining: True for manual per-trial baselining, False for MNE's built-in baselining
MANUAL_BASELINE = True

# Epoch rejection measures
AUTOREJECT_EPOCH_REJECTION = True # Autoreject epoch rejection based on dynamic threshold
REJECT_AMPLITUDE_UV = 120  # Manual; Peak-to-peak amplitude threshold in µV; ignored if autoreject enabled

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_epochs_data():
    """
    Generate preprocessed Epochs objects for all subjects, conditions, and triggers.
    
    Returns:
        dict: Nested dictionary with structure 
              {subject_name: {condition_name: {trigger_type: mne.Epochs or np.ndarray}}}
    """
    print("=" * 70)
    print("EPOCHS DATA GENERATION PIPELINE")
    print("=" * 70)
    
    print(f"\nData path: {DATA_PATH}")
    print(f"Included conditions: {INCLUDED_CONDITIONS}")
    print(f"Excluded subjects: {len(EXCLUDED_SUBJECTS) if EXCLUDED_SUBJECTS else 0}")
    print(f"Output format: {OUTPUT_FORMAT}")
    print(f"Manual baselining: {MANUAL_BASELINE}")
    
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
    
    print(f"  Found {len(raw_data)} subjects")
    
    # Create epochs
    print("\n[STEP 3] Creating epochs...")
    epochs_data = preprocessor.create_subject_epochs(
        raw_data, 
        output_format=OUTPUT_FORMAT,
        manual_baseline=MANUAL_BASELINE
    )
    
    if not epochs_data:
        print("ERROR: Failed to create epochs.")
        return None
    
    # Step 4: Apply eye dominance remapping if enabled
    if EYE_DOMINANCE_CSV:
        print("\n[STEP 4] Applying eye dominance remapping...")
        try:
            eye_dom_data = preprocessing.eye_dominance.load_eye_dominance(EYE_DOMINANCE_CSV, SUBJECT_COL, DOMINANCE_COL)
            epochs_data = preprocessing.eye_dominance.apply_eye_dominance_remapping(epochs_data, eye_dom_data)
        except Exception as e:
            print(f"  WARNING: Could not load eye dominance data: {e}")
            print("  Continuing without eye dominance remapping...")
    else:
        print("\n[STEP 4] Skipping eye dominance remapping (disabled)")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total subjects processed: {len(epochs_data)}")
    
    for subject, conditions in epochs_data.items():
        print(f"\n  {subject}:")
        for condition, triggers in conditions.items():
            print(f"    {condition}:")
            for trigger_type, data in triggers.items():
                if OUTPUT_FORMAT == 'epochs':
                    n_epochs = len(data)
                    n_channels = len(data.ch_names)
                    tmin, tmax = data.times[0], data.times[-1]
                    print(f"      {trigger_type}: {n_epochs} epochs, {n_channels} channels, [{tmin:.2f}, {tmax:.2f}]s")
                else:
                    # numpy array format: [n_channels, n_trials, n_times]
                    shape = data.shape
                    print(f"      {trigger_type}: shape {shape} (channels, trials, times)")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    return epochs_data


# Variable to store the result for later use
epochs_data = None

if __name__ == '__main__':
    epochs_data = generate_epochs_data()
    
    # The epochs_data variable is now available for further analysis
    # Structure: {subject_name: {condition_name: {trigger_type: Epochs/ndarray}}}
    print("\nData stored in 'epochs_data' variable.")
    print("Access example: epochs_data['STMD1']['baseline']['left']")

    # print dictionary structure
    if epochs_data:
        print("\n" + "=" * 70)
        print("DICTIONARY STRUCTURE")
        print("=" * 70)
        print(f"epochs_data  ({len(epochs_data)} subjects)")
        for subj, conditions in epochs_data.items():
            print(f"  └─ '{subj}'  ({len(conditions)} conditions)")
            for cond, triggers in conditions.items():
                print(f"      └─ '{cond}'  ({len(triggers)} keys)")
                for key, data in triggers.items():
                    n = len(data) if hasattr(data, '__len__') else data.shape[1]
                    print(f"          └─ '{key}'  ({n} epochs/trials)")