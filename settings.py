# This file contains the settings for the STMD EEG data analysis project.
from pathlib import Path

# LUTs for mapping and reordering EGI channel names to standard 10-20 system names
EGI_TO_1020_MAP = {
    "EEG 10": "EEG FP1-REF",
    "EEG 5": "EEG FP2-REF",
    "EEG 12": "EEG F3-REF",
    "EEG 60": "EEG F4-REF",
    "EEG 20": "EEG C3-REF",
    "EEG 50": "EEG C4-REF",
    "EEG 28": "EEG P3-REF",
    "EEG 42": "EEG P4-REF",
    "EEG 35": "EEG O1-REF",
    "EEG 39": "EEG O2-REF",
    "EEG 18": "EEG F7-REF",
    "EEG 58": "EEG F8-REF",
    "EEG 24": "EEG T3-REF",
    "EEG 52": "EEG T4-REF",
    "EEG 30": "EEG T5-REF",
    "EEG 44": "EEG T6-REF",
    "EEG 29": "EEG A1-REF",
    "EEG 47": "EEG A2-REF",
    "EEG 6": "EEG FZ-REF",
    "EEG VREF": "EEG CZ-REF",
    "EEG 34": "EEG PZ-REF",
    "EEG 17": "EEG T1-REF",
    "EEG 1": "EEG T2-REF",
}
CHORDER_STANDARD = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

SEED = 42  # Set a fixed seed for reproducibility, use for shuffling and sampling training dataset

# Preprocessing parameters
L_FREQ = 0.1
H_FREQ = 75.0
RSFREQ = 200
INCLUDED_CONDITIONS = ['baseline', 'post-patching']
EXCLUDED_SUBJECTS = ['STMD35', 'STMD37', 'STMD38', 'STMD39', 'STMD40', 'STMD41', 'STMD42', 'STMD43', 'STMD44', 'STMD45']
RECORDINGS_PER_CONDITION = None


# Base directories //TODO change back to \data\STMD_data
DATA_ROOT = r".\STMD_data"
FINETUNE_DIR = Path('.') / "finetune_epochs"
PATCH_DIR = Path('.') / "postpatch_epochs"
FINETUNE_OUTPUT_DIR = Path('.') / "finetune_outputs"

# Model
LABRAM_VERSION = 'labram_base_patch200_200'  # Set the model architecture to use for fine-tuning (must be supported by timm.create_model())
