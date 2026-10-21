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
NOTCH_WIDTH = 3.0
INCLUDED_CONDITIONS = ['baseline', 'post-patching']
EXCLUDED_SUBJECTS = ["STMD67", "STMD71"]
NO_PRESS_FROM_SUBJECT = 59
RECORDINGS_PER_CONDITION = None
# Every trial is split into a pre- and a post-stimulus patch of one LaBraM patch (1 s, 200 samples) each, meeting at -0.3 s.
# The patches go through the model in separate forward passes and get their own wavelet targets. The scored targets keep their
# ~0.29 s wavelet reach inside their own patch, so every scored bin is computed from real data of its patch only (no padding),
# and the pre and post errors share no data. The post patch starts before onset to give the early post-stimulus bins real data.
PRE_WINDOW = (-1.3, -0.3)
POST_WINDOW = (-0.3, 0.7)
EPOCH_WINDOW = (PRE_WINDOW[0], POST_WINDOW[1])  # saved input epoch and wavelet target: both patches, split during training
PATCH_SIZE = round((POST_WINDOW[1] - POST_WINDOW[0]) * RSFREQ)
TARGET_WINDOW_SIZE = PATCH_SIZE  # the autoencoder predicts the raw signal of one patch
# wavelet targets are pooled in time: the power of TARGET_POOL consecutive samples is averaged (utils.pool_power), 25 ms bins.
# The wavelets are far wider than one sample (~90 ms at 45 Hz, ~300 ms at 5 Hz), so the pooling loses little resolution
TARGET_POOL = 5
TARGET_FS = RSFREQ // TARGET_POOL  # bins per second of the wavelet targets
SPECTRAL_TIME_BINS = PATCH_SIZE // TARGET_POOL  # time bins of the wavelet target of one patch
# scored parts of the patches, bins whose wavelet reaches outside their patch are masked as well (utils.valid_wavelet_bins)
PRE_TARGET_WINDOW = (-1.0, -0.7)  # wavelet support -1.29 to -0.41 s
POST_TARGET_WINDOW = (0.0, 0.4)  # wavelet support -0.29 to 0.69 s at 5 Hz
TARGET_WINDOWS = [PRE_TARGET_WINDOW, POST_TARGET_WINDOW]
# time window arguments of FinetuneDataset: the saved inputs and targets span EPOCH_WINDOW and are cut into one sample per patch
PATCH_WINDOWS = {'pre': PRE_WINDOW, 'post': POST_WINDOW}
PATCH_PARAMS = {
    'epoch_window': EPOCH_WINDOW,
    'patch_windows': PATCH_WINDOWS,
    'fs': RSFREQ,
    'target_fs': TARGET_FS,
}
# target bins of every patch the training loss, the validation metrics and the target stats use, masked by utils.valid_wavelet_bins:
# all bins of the patch computed from its own data, set to PRE_TARGET_WINDOW / POST_TARGET_WINDOW to train on the scored bins only
LOSS_WINDOWS = {'pre': PRE_WINDOW, 'post': POST_WINDOW}
# masked-channel reconstruction: a target channel is hidden together with its N_NEIGHBOURS nearest electrodes
# (HydroCel-65 positions), the same cluster as the "cluster" scheme of ridge_leakage.py
N_NEIGHBOURS = 4
# gain injection (ridge_leakage.py, labram_inj_test.py): a visual-cortex source is made stronger after stimulus onset
INJECTION_ONSET_RAMP = 0.02  # s, raised-cosine ramp of the injected gain after stimulus onset
INJECTION_SOURCE_DEPTH = 0.7  # sphere model: dipole position as a fraction of the head radius, towards the midpoint of O1 and O2
INJECTION_LABELS = ["pericalcarine"]  # fsaverage model: aparc labels of the injected source (pericalcarine = primary visual cortex)
# mixed injection: fsaverage aparc labels injected as separate sources and their gains, an injection value scales all of them
MIXED_INJECTION_GAINS = {"pericalcarine": 0.1, "lateraloccipital": 0.25, "lingual": -0.1}
INJECTION_N_ROI = 4  # the injection results are averaged over the channels where the injected pattern is strongest (largest |g|)
MIN_REACTION_TIME = 0.5  # s, trials with a button press earlier than this after onset are dropped in make_dataset.py (~1.4% of trials)
PAD_TIME = 0.3  # reflect padding of each patch for its wavelet transform, only reaches bins outside the scored targets
PADDING = int(PAD_TIME * RSFREQ)  # Padding in samples for wavelet/superlet targets
WHOLE_EPOCH_WINDOW = EPOCH_WINDOW  # Whole epoch time window for preprocessing
BASELINE_WINDOW = None  # no MNE baseline correction: a baseline window would shift each patch by data of the other one
ANALYSIS_WINDOW = POST_TARGET_WINDOW  # before every button press

# time windows the inference metrics are computed on, cut from each patch; bins computed partly from the patch's reflect
# padding are left out (run_inference), so 'full' is every valid bin of the patch and 'analysis' the scored windows
METRIC_WINDOWS = {
    'full': PATCH_WINDOWS,
    'analysis': {'pre': PRE_TARGET_WINDOW, 'post': POST_TARGET_WINDOW},
}


ICA_PARAMS = {
    'n_components': 15,
    'method': 'infomax',
    'random_state': SEED,
    'fit_params': {'extended': True},
    'max_iter': 'auto',
}

ICLABEL_THRESHOLDS = {
    'eye blink': 0.7,
    'heart beat': 0.8,
    'line noise': 0.8,
    'channel noise': 0.8,
    'muscle artifact': 0.9,
}

AUTOREJECT_WINDOW = WHOLE_EPOCH_WINDOW  # autoreject checks every sample the model input and the wavelet targets are made of
AUTOREJECT_PARAMS = {
    'random_state': SEED,
    'n_jobs': 4,
    'n_interpolate': np.array([1, 2, 4, 8])
}

AUTO_BAD_CHANNEL_PARAMS = {
    'clip_fraction': 0.005,  # channel-specific saturation in more than this fraction of samples -> clipping
    'common_clip_share': 0.25,  # samples where at least this share of channels saturate together are ignored
    'flat_std_uv': 1.0,  # raw 1-40 Hz std below this -> flat
    'noise_ratio': 10.0,  # average-referenced 1-40 Hz std above this multiple of the channel median -> noisy
    'line_freq': 50.0,
    'line_min_db': 20.0,  # line band power this many dB above its neighbouring bands ...
    'line_excess_db': None,  # ... and this many dB above the median channel -> line noise (None = off, see module docs)
    'max_iterations': 3,
    'exclude': ['EEG VREF'],  # recording reference, flat before re-referencing
}


QC_MAX_DROPPED_SHARE = 1/3
QC_MAX_DROPPED_CHANS = 12

TRIGGER_MAPPING = Path("libs") / "unscrambly" / "preprocessing" / "trigger_mapping.yaml"

INPUT_EPOCHS_DIR_NAME = "input_epochs"
WAVELET_DIR_NAME = "wavelet_targets"
# Base directories
#CORE_DIR = Path(__file__).parent # get path to core directory
BASE_DIR = Path(".")
DATA_ROOT = BASE_DIR / "data" / "raw"/ "STMD_data"
PROCESS_DIR = BASE_DIR / "data" / "processed"
CACHE_DIR = PROCESS_DIR / "cache"

FINETUNE_DIR = PROCESS_DIR / "pre_patching"
FINETUNE_INPUT_EPOCHS_DIR = FINETUNE_DIR / INPUT_EPOCHS_DIR_NAME
FINETUNE_WAVELET_DIR = FINETUNE_DIR / WAVELET_DIR_NAME

PATCH_DIR = PROCESS_DIR / "post_patching"
PATCH_INPUT_EPOCHS_DIR = PATCH_DIR / INPUT_EPOCHS_DIR_NAME
PATCH_WAVELET_DIR = PATCH_DIR / WAVELET_DIR_NAME

FINETUNE_OUTPUT_DIR = BASE_DIR / "outputs" / "finetune_outputs"
PLOT_DIR = BASE_DIR / "outputs" / "plots"
INFERENCE_DIR = BASE_DIR / "outputs" / "inference_outputs"
PLOT_MACRO = PLOT_DIR / "macro_plots"
PLOT_INFERENCE = PLOT_DIR / "inference_plots"
QC_DIR = BASE_DIR / "outputs" / "qc"
CHECKPOINT_DIR = BASE_DIR / "libs" / "labram" / "checkpoints"

# Model
LABRAM_VERSION = 'labram_base_patch200_200'

# Model head params

AUTOENCODER_PARAMS = {
    'n_chans': N_CHANNELS,
    'target_dim': TARGET_WINDOW_SIZE,
    'model_version': LABRAM_VERSION,
}

N_FREQS = 50
SPECTRAL_DECODER_PARAMS = {
    'n_chans': N_CHANNELS,
    # below 5 Hz the wavelets reach further than the 1 s target window, above 45 Hz muscle activity and line noise dominate
    'freq_window': (5, 45),
    'n_freqs': N_FREQS,
    'target_dim': N_FREQS * SPECTRAL_TIME_BINS,
    'model_version': LABRAM_VERSION,
}

FREQS = np.geomspace(SPECTRAL_DECODER_PARAMS["freq_window"][0], SPECTRAL_DECODER_PARAMS["freq_window"][1], N_FREQS)

# complex Morlet wavelet targets: as many cycles as possible up to max_cycles while the 3 SD reach stays within max_reach,
# n(f) = min(max_cycles, 2*pi*f*max_reach/3), i.e. MIN_CYCLES at the lowest frequency rising to max_cycles at ~9 Hz
MORLET_CYCLES = 2 * np.pi * np.sqrt(1.5 / 2)  # ~5.44 cycles, same as the previous cmor1.5-1.0
MIN_CYCLES = 3  # cycles at the lowest frequency
WAVELET_PARAMS = {
    'max_cycles': MORLET_CYCLES,
    'max_reach': 3 * MIN_CYCLES / (2 * np.pi * FREQS[0]),  # ~0.29 s, the 3 SD reach of MIN_CYCLES at the lowest frequency
}

FREQ_BANDS = {
    'theta (5-8Hz)': (5, 8),
    'alpha (8-12Hz)': (8, 12),
    'beta (12-30Hz)': (12, 30),
    'gamma (30-45Hz)': (30, 45)
}
FREQ_WINDOW_LABEL = f"{SPECTRAL_DECODER_PARAMS['freq_window'][0]}-{SPECTRAL_DECODER_PARAMS['freq_window'][1]} Hz"  # all bins
# indices of the FREQS bins falling into each band, used to index the frequency dimension of spectral tensors
FREQ_BAND_INDICES = {name: np.where((FREQS >= low) & (FREQS <= high))[0] for name, (low, high) in FREQ_BANDS.items()}