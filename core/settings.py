# This file contains the settings for the STMD EEG data analysis project.
from pathlib import Path

import numpy as np

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
N_CHANNELS = len(CHORDER_STANDARD)


EPS = 1e-8  # Small constant to avoid division by zero in normalization and log calculations

SEED = 42  # Set a fixed seed for reproducibility, use for shuffling and sampling training dataset

# Preprocessing parameters
L_FREQ = 0.1
H_FREQ = 75.0
RSFREQ = 200
INCLUDED_CONDITIONS = ['baseline', 'post-patching']
EXCLUDED_SUBJECTS = ['STMD35', 'STMD37', 'STMD38', 'STMD39', 'STMD40', 'STMD41', 'STMD42', 'STMD43', 'STMD44', 'STMD45']
RECORDINGS_PER_CONDITION = None
TIME_WINDOW = (-0.2, 0.8)#  1 second (200 sample) time window for labram compatibility
EXTENDED_OFFSET = 1.5
PAD_TIME = 1.5
PADDING = int(PAD_TIME * RSFREQ)  # Padding in samples for wavelet/superlet targets
EXTENDED_TIME_WINDOW = (TIME_WINDOW[0] - EXTENDED_OFFSET, TIME_WINDOW[1] + EXTENDED_OFFSET)  # Extended time window for wavelet/superlet targets
WINDOW_SIZE = int((TIME_WINDOW[1] - TIME_WINDOW[0]) * RSFREQ)

EPOCHS_DIR_NAME = "epochs"
SUPERLET_DIR_NAME = "superlet_targets"
WAVELET_DIR_NAME = "wavelet_targets"
# Base directories
#CORE_DIR = Path(__file__).parent # get path to core directory
BASE_DIR = Path(".")
DATA_ROOT = BASE_DIR / "data" / "raw"/ "STMD_data"
PROCESS_DIR = BASE_DIR / "data" / "processed"

FINETUNE_DIR = PROCESS_DIR / "pre_patching"
FINETUNE_EPOCHS_DIR = FINETUNE_DIR / EPOCHS_DIR_NAME
FINETUNE_SUPERLET_DIR = FINETUNE_DIR / SUPERLET_DIR_NAME
FINETUNE_WAVELET_DIR = FINETUNE_DIR / WAVELET_DIR_NAME

PATCH_DIR = PROCESS_DIR / "post_patching"
PATCH_EPOCHS_DIR = PATCH_DIR / EPOCHS_DIR_NAME
PATCH_SUPERLET_DIR = PATCH_DIR / SUPERLET_DIR_NAME
PATCH_WAVELET_DIR = PATCH_DIR / WAVELET_DIR_NAME

FINETUNE_OUTPUT_DIR = BASE_DIR / "outputs" / "finetune_outputs"
PLOT_DIR = BASE_DIR / "outputs" / "plots"
INFERENCE_DIR = BASE_DIR / "outputs" / "inference_outputs"
PLOT_MACRO = PLOT_DIR / "macro_plots"
PLOT_INFERENCE = PLOT_DIR / "inference_plots"
CHECKPOINT_DIR = BASE_DIR / "libs" / "labram" / "checkpoints"

# Model
LABRAM_VERSION = 'labram_base_patch200_200'

# Model head params

AUTOENCODER_PARAMS = {
    'n_chans': N_CHANNELS,
    'target_dim': WINDOW_SIZE,
    'model_version': LABRAM_VERSION,
}

N_FREQS = 50
SPECTRAL_DECODER_PARAMS = {
    'n_chans': N_CHANNELS,
    'freq_window': (2, 60),
    'n_freqs': N_FREQS,
    'target_dim': N_FREQS*WINDOW_SIZE,
    'model_version': LABRAM_VERSION,
    'wavelet': 'cmor1.5-1.0'
}

FREQS = np.geomspace(SPECTRAL_DECODER_PARAMS["freq_window"][0], SPECTRAL_DECODER_PARAMS["freq_window"][1], N_FREQS)

FREQ_BANDS = {
    'delta (2-4Hz)': (2, 4),
    'theta (4-8Hz)': (4, 8),
    'alpha (8-12Hz)': (8, 12),
    'beta (12-30Hz)': (12, 30),
    'gamma (30-60Hz)': (30, 60)
}