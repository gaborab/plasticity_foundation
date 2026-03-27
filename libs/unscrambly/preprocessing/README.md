# Preprocessing Module

This module provides a hierarchical preprocessing pipeline for EEG data analysis, built around the `DataPreprocessor` class. The architecture separates **low-level** single-directory operations from **high-level** multi-subject batch processing.

---

## Architecture Overview

```
                         ┌─────────────────────────────────┐
                         │     HIGH-LEVEL INTERFACE        │
                         │   (Multi-subject processing)    │
                         └─────────────────────────────────┘
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        │                                                               │
        ▼                                                               ▼
┌───────────────────┐                                       ┌───────────────────────┐
│ preprocess_subjects│                                       │ create_subject_epochs │
│                   │                                       │                       │
│ Iterates over:    │──────────────────────────────────────▶│ Processes output of   │
│ subjects →        │        returns Dict[subj][cond]       │ preprocess_subjects   │
│   conditions →    │              of Raw objects           │ into epochs/arrays    │
│     recordings    │                                       │                       │
└─────────┬─────────┘                                       └───────────┬───────────┘
          │                                                             │
          │ calls internally                                            │ calls internally
          ▼                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           LOW-LEVEL INTERFACE                                        │
│                        (Single-directory processing)                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
          │                                                             │
          ▼                                                             ▼
┌───────────────────┐                                       ┌───────────────────────┐
│   preprocess_dir  │                                       │_create_epochs_from_raw│
│                   │                                       │                       │
│ • Loads EDF files │                                       │ • Event extraction    │
│ • Bad channels    │                                       │ • Epoch creation      │
│ • ICA cleaning    │                                       │ • Amplitude rejection │
│ • Filtering       │                                       │                       │
│ • Re-referencing  │                                       ├───────────────────────┤
│ • ChkEEG triggers │                                       │_apply_manual_baseline │
│                   │                                       │                       │
│ Returns: Raw obj  │                                       │ • Per-trial baseline  │
└───────────────────┘                                       │   correction          │
                                                            └───────────────────────┘
```

---

## Function Hierarchy

### Low-Level Functions

These functions operate on a **single directory path** and perform the core preprocessing operations. They are the building blocks of the pipeline.

#### `preprocess_dir()`

**Purpose:** Loads and preprocesses all EDF recordings from a single directory, returning a concatenated `mne.io.Raw` object.

**Input:** Uses `self.directory` (set at initialization or temporarily modified)

**Output:** `(mne.io.Raw, pd.DataFrame)` - preprocessed continuous data and optional trial info

**Preprocessing steps applied:**
- EDF file loading with montage assignment
- Bad channel handling (exclusion or interpolation)
- ICA artifact removal (eye blinks, muscle artifacts)
- Bandpass and notch filtering
- Re-referencing (e.g., average reference)
- ChkEEG trigger extraction (optional)

```python
# Example: Process a single condition folder
preprocessor = DataPreprocessor(
    directory=r"C:\data\subject01\baseline",
    filter_params={'highpass': 0.1, 'lowpass': 40},
    time_window=(-0.1, 0.5)
)
raw, trial_info = preprocessor.preprocess_dir()
```

#### `create_epochs(manual_baseline=False)`

**Purpose:** Creates epochs from a single directory by calling `preprocess_dir()` internally, then segmenting around triggers.

**Input:** Uses `self.directory` and preprocessing parameters from initialization

**Output:** `(Dict[str, Epochs], Dict[str, Evoked])` - unaveraged and averaged epochs by trigger type

```python
# Example: Get epochs for a single recording session
unaveraged, averaged = preprocessor.create_epochs(manual_baseline=True)
# unaveraged['left'] -> mne.Epochs for left mismatch
# averaged['global'] -> mne.Evoked for global mismatch
```

#### `_create_epochs_from_raw(raw, apply_mne_baseline=False)` *(private)*

**Purpose:** Converts any `mne.io.Raw` object into epochs for all defined trigger types.

**Key features:**
- Handles single triggers or lists of triggers per type
- Flexible baseline control (MNE built-in or disabled for manual)
- Amplitude-based epoch rejection

#### `_apply_manual_baseline(epochs)` *(private)*

**Purpose:** Applies per-trial baseline correction.

**Algorithm:**
```
For each trial:
    For each channel:
        baseline_mean = mean(signal[baseline_start:baseline_end])
        signal[:] = signal[:] - baseline_mean
```

This differs from MNE's default baselining which can average across trials first.

---

### High-Level Functions

These functions provide a **batch processing interface** for multi-subject studies. They iterate over directory structures and aggregate results.

#### `preprocess_subjects(excluded_subjects=None, included_conditions=None)`

**Purpose:** Processes an entire study dataset organized by subject and condition folders.

**Expected directory structure:**
```
base_directory/
├── subject_01/
│   ├── baseline/
│   │   ├── recording_001/
│   │   │   └── data.edf
│   │   └── recording_002/
│   │       └── data.edf
│   └── post_patching/
│       └── recording_001/
│           └── data.edf
├── subject_02/
│   ├── baseline/
│   └── post_patching/
└── subject_03/
    └── ...
```

**Parameters:**
- `excluded_subjects`: List of subject folder names to skip (case-insensitive)
- `included_conditions`: List of condition folder names to include (case-insensitive). If `None`, all conditions are processed.

**Output:** Nested dictionary with preprocessed Raw objects
```python
{
    'subject_01': {
        'baseline': mne.io.Raw,      # All baseline recordings concatenated
        'post_patching': mne.io.Raw
    },
    'subject_02': {
        'baseline': mne.io.Raw,
        'post_patching': mne.io.Raw
    }
}
```

**Example:**
```python
preprocessor = DataPreprocessor(
    directory=r"C:\study_data",  # Root containing subject folders
    filter_params={'highpass': 0.1, 'lowpass': 40, 'notch': 50},
    ica_enabled=True,
    ica_n_components=15,
    bad_channels='interpolate',
    time_window=(-0.1, 0.5),
    baseline_win=(-0.2, 0),
    triggers={'left': 'DIN21', 'center': 'DIN31', 'right': 'DIN41', 'global': 'DIN51'}
)

# Process all subjects except pilot, only baseline condition
subject_raws = preprocessor.preprocess_subjects(
    excluded_subjects=['pilot_01', 'pilot_02'],
    included_conditions=['baseline']
)
```

#### `create_subject_epochs(subject_data, output_format='epochs', manual_baseline=True)`

**Purpose:** Converts the output of `preprocess_subjects()` into epoched data, with support for per-trial manual baselining.

**Parameters:**
- `subject_data`: Output dictionary from `preprocess_subjects()`
- `output_format`: 
  - `'epochs'`: Returns `mne.Epochs` objects
  - `'array'`: Returns numpy arrays with shape `[n_channels, n_trials, n_times]`
- `manual_baseline`: If `True`, applies per-trial baseline correction. If `False`, uses MNE's standard baselining.

**Output:** Nested dictionary
```python
{
    'subject_01': {
        'baseline': {
            'left': mne.Epochs or np.ndarray,
            'center': mne.Epochs or np.ndarray,
            'right': mne.Epochs or np.ndarray,
            'global': mne.Epochs or np.ndarray
        },
        'post_patching': { ... }
    },
    'subject_02': { ... }
}
```

**Example:**
```python
# Continue from preprocess_subjects example...

# Get MNE Epochs objects with manual per-trial baselining
epochs_data = preprocessor.create_subject_epochs(
    subject_raws,
    output_format='epochs',
    manual_baseline=True
)

# Access specific data
subj1_baseline_left = epochs_data['subject_01']['baseline']['left']
print(f"Subject 01 baseline left trials: {len(subj1_baseline_left)}")

# Or get numpy arrays for custom analysis
array_data = preprocessor.create_subject_epochs(
    subject_raws,
    output_format='array',
    manual_baseline=True
)
# Shape: [n_channels, n_trials, n_times]
```

---

## Baselining Methods

### MNE Built-in Baselining (`manual_baseline=False`)

MNE applies baseline correction during epoch creation. The baseline is computed and subtracted in a single operation, which is efficient but less transparent for per-trial analysis.

### Manual Per-Trial Baselining (`manual_baseline=True`)

For each individual trial:
1. Extract the signal in the baseline window (e.g., -200ms to 0ms)
2. Calculate the mean voltage across this window
3. Subtract this value from **all** timepoints in that trial

**Advantages:**
- Explicit control over the baseline computation
- Each trial is corrected independently
- Better for analyses that require trial-by-trial consistency

**Implementation:**
```python
# Vectorized operation in _apply_manual_baseline()
baseline_mask = (times >= baseline_start) & (times <= baseline_end)
baseline_means = data[:, :, baseline_mask].mean(axis=2, keepdims=True)
data -= baseline_means
```

---

## Complete Workflow Example

```python
from preprocessing import DataPreprocessor

# 1. Initialize with all preprocessing parameters
preprocessor = DataPreprocessor(
    directory=r"C:\EEG_Study\data",
    
    # Channel selection
    channels=['EEG 35', 'EEG 36', 'EEG 37', 'EEG 39', 'EEG 40'],
    
    # Artifact handling
    bad_channels='interpolate',
    ica_enabled=True,
    ica_n_components=15,
    
    # Filtering
    filter_params={
        'highpass': 0.1,
        'lowpass': 40,
        'notch': 50
    },
    
    # Epoching
    time_window=(-0.1, 0.8),
    baseline_win=(-0.2, 0),
    reject_amplitude_uv=100,
    
    # Triggers
    triggers={
        'left': 'DIN21',
        'center': 'DIN31', 
        'right': 'DIN41',
        'global': 'DIN51'
    }
)

# 2. HIGH-LEVEL: Process all subjects
subject_raws = preprocessor.preprocess_subjects(
    excluded_subjects=['test_subject', 'bad_data_01'],
    included_conditions=['baseline', 'post_intervention']
)

# 3. HIGH-LEVEL: Create epochs with manual baselining
epochs = preprocessor.create_subject_epochs(
    subject_raws,
    output_format='epochs',
    manual_baseline=True
)

# 4. Use the data
for subject, conditions in epochs.items():
    for condition, triggers in conditions.items():
        for trigger_type, epoch_obj in triggers.items():
            print(f"{subject}/{condition}/{trigger_type}: {len(epoch_obj)} trials")
```

---

## Related Modules

| Module | Purpose |
|--------|---------|
| `epoch_extraction.py` | Standalone epoch functions (legacy, not used by DataPreprocessor) |
| `file_io.py` | File discovery, EDF/ETI/NPZ loading utilities |
| `ica_selection.py` | ICA component selection and persistence |
| `bad_channel_selection.py` | Bad channel marking and persistence |

---

## Notes

- **Case insensitivity:** Both `excluded_subjects` and `included_conditions` matching is case-insensitive
- **Preprocessing inheritance:** All parameters set in `__init__` are applied when using high-level functions
- **Memory:** `preprocess_subjects()` holds all Raw objects in memory; for very large datasets, consider processing in batches
