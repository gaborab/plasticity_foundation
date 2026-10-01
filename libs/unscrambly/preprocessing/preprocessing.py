import os
import mne
import numpy as np
import pandas as pd
import yaml
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import pickle
from autoreject import AutoReject, get_rejection_threshold

from .ica_selection import directory_ica as get_ica
from .bad_channel_selection import directory_bad_channels as get_bad_channels
from .auto_bad_channels import detect_bad_channels
from .file_io import extract_edf_from_session as edf_from_dir
from .file_io import extract_trigger_npz
from .file_io import extract_pkl_from_directory
from mne_icalabel.iclabel import iclabel_label_components
from core.io_utils import generate_hash
from importlib.metadata import version

ICLABEL_CLASSES = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']


class DataPreprocessor:
    """
    A class for preprocessing data with various methods.
    Available preprocessing steps: ICA, re-referencing, filtering, baselining, bad channel handling, bad epoch handling.
    Applies preprocessing to either a single recordings or a directory of recordings.
    Supports multiple output types for different analyses (mne objects or numpy arrays organized in dictionaries).
    """

    # Contains all parameters of initiated preprocessing
    def __init__(self,
                 directory: str,
                 channels: Optional[List[str]] = None,
                 re_reference: Optional[str] = None,
                 bad_channels: Optional[str] = None,
                 ica_n_components: Optional[int] = None,
                 ica_method: Optional[str] = None,
                 ica_params: Optional[Dict] = None,
                 iclabel_thresholds: Optional[Dict] = None,
                 filter_params: Optional[Dict] = None,
                 baseline_win: Optional[Tuple] = None,
                 reject_amplitude_uv: Optional[int] = None,
                 time_window: Optional[Tuple] = None,
                 use_chkeeg_triggers: bool = True,
                 split_by: str = 'sides',
                 trigger_type: str = 'onset',
                 autoreject_mode: str = None,
                 autoreject_params: Optional[Dict] = None,
                 auto_bad_channel_params: Optional[Dict] = None,
                 qc_dir: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 autoreject_window: Optional[Tuple] = None,
                 interactive_selection: Optional[bool] = True
                 ):
        """
        Initialize the DataPreprocessor with configuration parameters.

        Parameters:
        directory (str): Path to directory / single recording folder to preprocess.
        channels (list): The list of channels to include. Format: e.g. "EEG 35". If set to None / empty list, all available channels are included.
        re_reference (str): Re-referencing method to use (e.g. average, laplacian). If set to None, no re-referencing is applied.
        bad_channels (str): Method to handle bad channels. Possible values: "exclude", "interpolate", None. 
                            If set to None, all channels are maintained. If enabled, bad channels can be picked manually.
        ica_n_components (int): Number of components to use in ICA.
        filter_params (dict): Dictionary of filter parameters (highpass, lowpass, notch). If set to None, no filtering is applied.
        baseline_win (tuple): Baseline window to apply (before trigger onset), in seconds. If set to None, no baselining is applied.
        reject_amplitude_uv (int): Peak-to-peak amplitude cut-off for epochs (in µV). Any epochs with higher peak-to-peak amplitude values are excluded.
                                   If set to None, automatic bad epoch rejection based on max peak-to-peak amplitude is not performed.
        time_window (tuple): The time window around triggers to create epochs (start, end).
        use_chkeeg_triggers (bool): Whether to use trigger timestamps and eccentricity data from pre-generated NPZ files (ChkEEG format).
                                     When enabled, EDF annotations are replaced with NPZ trigger data, and eccentricity info is encoded
                                     into annotation descriptions (e.g. 'DIN21/ecc_3') and stored on the preprocessor instance.
        split_by (str): How to organize epoch dictionary keys. Must be one of:
                        'sides'        -> keys: left, center, right, global  (split by mismatch side)
                        'eccentricity' -> keys: ecc_1, ecc_2, ecc_3, ...     (split by eccentricity value; excludes ecc=0 and global stimuli)
                        'both'         -> keys: left_1, left_2, ..., global  (split by side AND eccentricity; global has no eccentricity subdivision)
                        Trigger codes are loaded from trigger_mapping.yaml.
        trigger_type (str): Which trigger type to use for epoching. Must be 'onset' or 'offset'.
                            'onset'  -> uses stimulus onset triggers (e.g. DIN21, DIN31, DIN41, DIN51)
                            'offset' -> uses stimulus offset triggers (e.g. DIN23, DIN33, DIN43, DIN53)
        use_global_autoreject (bool): Whether to use the Autoreject algorithm to identify the optimal max. peak-to-peak amplitude threshold for epoch rejection.
                        If enabled, "reject_amplitude_uv" parameter is ignored, instead the threshold value chosen by Autoreject is used.
        auto_bad_channel_params (dict): Parameters for automatic bad channel detection (see auto_bad_channels.py), run on every raw
                        recording before ICA. Detected channels are added to the manual bad channel selection and handled
                        according to "bad_channels" (requires bad_channels to be set). If None, automatic detection is disabled.
                        Detection results are collected in self.auto_bad_channel_report.
        """

        # --- Required parameters ---
        if not directory:
            raise ValueError('ERROR: No directory path.')
        self.directory = directory

        # --- Trigger type ---
        valid_trigger_types = ('onset', 'offset')
        if trigger_type not in valid_trigger_types:
            raise ValueError(f"trigger_type must be one of {valid_trigger_types}, got: '{trigger_type}'")
        self.trigger_type = trigger_type

        # --- Trigger mapping (loaded from YAML) ---
        self.triggers = self._load_triggers(trigger_type)
        self.local_triggers = {k: v for k, v in self.triggers.items() if k != 'global'}

        # --- Split-by mode ---
        valid_split_modes = ('sides', 'eccentricity', 'both')
        if split_by not in valid_split_modes:
            raise ValueError(f"split_by must be one of {valid_split_modes}, got: '{split_by}'")
        self.split_by = split_by

        if not time_window:
            raise ValueError('ERROR: Time window is not defined.')
        self.time_window = time_window

        # --- Channel and re-referencing ---
        self.channels = channels if channels is not None else []
        self.re_reference = re_reference
        self.bad_channels = bad_channels
        self.interpolation_log = {}

        # --- ICA ---

        self.ica_method = ica_method  # 'manual' (ica_selection.pkl) or 'iclabel'
        self.ica_params = ica_params or {}
        self.iclabel_thresholds = iclabel_thresholds or {}
        self.ica_report = []

        self.qc_dir = qc_dir or Path(self.directory) / 'qc'
        self.cache_dir = cache_dir or Path(self.directory) / 'cache'

        if self.ica_method == 'manual':
            if not ica_n_components:
                raise ValueError('ERROR: ICA enabled, but number of components is not defined.')
            self.ica_n_components = ica_n_components
        elif self.ica_method == 'iclabel':
            if not ica_params:
                raise ValueError('ERROR: ICA enabled, but ICA parameters are not defined.')

        # --- Filtering ---
        self.filter_params = filter_params or {}
        self.highpass = self.filter_params.get('highpass', None)
        self.lowpass = self.filter_params.get('lowpass', None)
        self.notch = self.filter_params.get('notch', None)
        self.notch_width = self.filter_params.get('notch_width', None)

        # ChkEEG triggers (NPZ-based)
        self.use_chkeeg_triggers = use_chkeeg_triggers
        # Stores per-recording eccentricity mappings: {edf_path: {sample_idx: eccentricity_value}}
        self._trigger_eccentricities = {}

        # --- Other preprocessing parameters ---
        self.baseline_win = baseline_win
        # automatically extend trial time window to fit baseline window
        if self.baseline_win and self.time_window:
            self.time_window_with_baseline = (
                min(self.time_window[0], self.baseline_win[0]),
                max(self.time_window[1], self.baseline_win[1])
            )
        else:
            self.time_window_with_baseline = None

        self.reject_amplitude_uv = reject_amplitude_uv

        self.autoreject_mode = autoreject_mode
        self.autoreject_params = autoreject_params or {}
        self.autoreject_window = autoreject_window
        self.autoreject_report = []

        # --- Automatic bad channel detection ---
        self.auto_bad_channel_params = auto_bad_channel_params
        self.auto_bad_channel_report = []  # one dict per checked channel per recording
        if self.auto_bad_channel_params is not None and not self.bad_channels:
            print(
                'WARNING: Automatic bad channel detection enabled, but bad_channels is None. Detected channels will not be handled.')
        self.interactive_selection = interactive_selection
    @staticmethod
    def _load_triggers(trigger_type: str = 'onset') -> Dict[str, str]:
        """
        Load trigger mappings from trigger_mapping.yaml for the given trigger type.
        
        Parameters:
        trigger_type (str): 'onset' or 'offset'.
        
        Returns:
        triggers (dict): Mapping of side name to DIN code, e.g. {'left': 'DIN21', ...}
        """
        yaml_path = Path(__file__).parent / 'trigger_mapping.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f'Trigger mapping file not found: {yaml_path}')

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        suffix = f'_{trigger_type}'
        triggers = {}
        for din_code, trigger_name in config['trigger_mappings'].items():
            if trigger_name.endswith(suffix):
                side = trigger_name.replace(suffix, '')
                triggers[side] = din_code

        if not triggers:
            raise ValueError(f'No {trigger_type} triggers found in trigger_mapping.yaml')

        print(f'\nTRIGGERS: {triggers}\n')
        return triggers

    # Method to preprocess epochs from multiple files & apply preprocessing
    def preprocess_dir(self,
                       bad_channel_selection: Optional[Dict] = None,
                       ica_selection: Optional[Dict] = None):
        """
        Applies preprocessing to recordings in directory, creates raw (continous) object.

        Parameters:
        bad_channel_selection (dict): Optional dictionary containing selected bad channels. Enables condition-level functionality.
                                      Used in higher-level directory processing (process_subjects), where selection interface is not called from preprocess_dir.
                                      If provided, skips interactive bad channel selection from within preprocess_dir. 
        ica_selection (dict): Optional pre-computed ICA selections (key: Path to EDF, value: tuple of (n_components, excluded_components)). Enables condition-level functionality.
                              Used in higher-level directory processing (process_subjects), where selection interface is not called from preprocess_dir.
                              If provided, skips interactive ICA selection from within preprocess_dir.

        Returns:
        raw (mne.io.Raw): Continous (& preprocessed) EEG data.
        """

        edf_paths = edf_from_dir(self.directory)
        raws_list = []

        # Determine bad channels
        bad_channels_dict = None
        if self.bad_channels:
            if bad_channel_selection is not None:
                print('INFO: Using provided bad channel selection.')
                bad_channels_dict = bad_channel_selection
            else:
                print('INFO: Bad channel handling enabled.')
                try:
                    bad_channels_dict = get_bad_channels(self.directory, edf_paths)
                except Exception as e:
                    print(f'WARNING: Failed retrieving bad channel data. Bad channel handling disabled. <{e}>')
                    self.bad_channels = None

        # Determine bad ICA components
        ica_dict = None
        if self.ica_method == 'manual':
            if ica_selection is not None:
                print('INFO: Using provided ICA selection.')
                ica_dict = ica_selection
            else:
                print('INFO: ICA enabled.')
                try:
                    ica_dict = get_ica(directory_path=self.directory, n_components=self.ica_n_components,
                                       edf_paths=edf_paths)
                except Exception as e:
                    print(f'WARNING: Failed retrieving ICA data. ICA disabled. <{e}>')
                    self.ica_method = None

        # Change working directory to handle relative paths in pickle files
        prev_cwd = os.getcwd()
        try:
            # os.chdir(self.directory)

            # If ChkEEG triggers enabled: load trigger data (incl. eccentricity) from NPZ files
            if self.use_chkeeg_triggers:
                print('INFO: Loading triggers from NPZ files.')
                trigger_dict = extract_trigger_npz(self.directory)

                # Warn about any recordings missing NPZ data
                for edf in edf_paths:
                    if edf not in trigger_dict or trigger_dict[edf] is None:
                        print(f'WARNING: No NPZ trigger data found for {Path(edf).stem}. '
                              f'EDF annotations will be used as fallback.')

            # Iterate over recordings in directory
            for edf in edf_paths:
                edf_path = Path(edf)
                raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)

                # Set Montage
                egi_montage = mne.channels.make_standard_montage('GSN-HydroCel-65_1.0')

                # Create a mapping from the standard montage names (e.g., 'E1') to own format ('EEG 1')
                ch_mapping = {}
                for ch_name in egi_montage.ch_names:
                    if ch_name == 'Cz':
                        ch_mapping[ch_name] = 'EEG VREF'
                    elif ch_name.startswith('E'):
                        number = ch_name[1:]
                        ch_mapping[ch_name] = f'EEG {number}'

                # Create an inverse mapping to temporarily rename your channels to the standard names
                inv_mapping = {v: k for k, v in ch_mapping.items()}

                # Temporarily rename, set the montage, and then rename back
                raw.rename_channels(inv_mapping)
                raw.set_montage(egi_montage, on_missing='warn')  # Use 'warn' to see if any channels don't match
                raw.rename_channels(ch_mapping)

                # Automatic bad channel detection on the raw recording (before ICA, which would smear channel artifacts)
                auto_bads = []
                if self.auto_bad_channel_params is not None:
                    auto_bads, auto_report = detect_bad_channels(raw, self.auto_bad_channel_params)
                    self.auto_bad_channel_report.extend({'recording': str(edf_path), **row} for row in auto_report)
                    if auto_bads:
                        reasons = {row['channel']: row['reasons'] for row in auto_report if row['reasons']}
                        print(f"Auto-detected {len(auto_bads)} bad channels in {edf_path.stem}: "
                              + ', '.join(f"{ch} ({reasons[ch]})" for ch in auto_bads))

                interpolated_channels = []
                # Handle bad channels (manual selection + automatic detection)
                if self.bad_channels:
                    if bad_channels_dict is not None and edf_path in bad_channels_dict and bad_channels_dict[edf_path]:
                        raw.info['bads'].extend(bad_channels_dict[edf_path])
                    raw.info['bads'].extend(ch for ch in auto_bads if ch not in raw.info['bads'])
                    interpolated_channels = raw.info['bads'].copy()
                    self.interpolation_log[edf_path.stem] = interpolated_channels
                    if self.bad_channels == 'exclude':
                        print(
                            f"Excluding {len(raw.info['bads'])} bad channels from {edf_path.stem}: {raw.info['bads']}")
                        raw.pick(picks='eeg', exclude='bads')
                    elif self.bad_channels == 'interpolate':
                        if raw.info['bads']:
                            print(
                                f"Interpolating {len(raw.info['bads'])} bad channels in {edf_path.stem}: {raw.info['bads']}")
                            raw.interpolate_bads(reset_bads=True)

                if self.re_reference:
                    if self.re_reference == 'average':
                        raw.set_eeg_reference('average', projection=False, verbose=False)
                        print('Re-referenced to average.')
                    else:
                        raw.set_eeg_reference(self.re_reference, projection=False, verbose=False)
                        print(f'Re-referenced to {self.re_reference}.')

                ica = None
                if self.ica_method == 'iclabel':
                    print("INFO: Running ICA with ICLabel component selection for", edf_path.stem)
                    # Automatic ICA component selection using ICLabel
                    ica_info = {
                        'edf': [str(edf_path), edf_path.stat().st_size, edf_path.stat().st_mtime],
                        'interpolated': sorted(interpolated_channels),
                        # manual + auto bads, collected before interpolate_bads resets info['bads']
                        're_reference': self.re_reference, 'notch_width': self.notch_width, 'fit_band': [1.0, 100.0],
                        'ica_params': self.ica_params, 'mne': mne.__version__, 'iclabel': version('mne-icalabel')
                    }
                    key = generate_hash(ica_info)
                    ica_path = self.cache_dir / 'ica' / f'{edf_path.stem}_{key}-ica.fif'
                    ica_path.parent.mkdir(parents=True, exist_ok=True)
                    proba_path = self.cache_dir / 'ica' / f'{edf_path.stem}_{key}-iclabel.npy'
                    proba_path.parent.mkdir(parents=True, exist_ok=True)

                    if ica_path.exists() and proba_path.exists():
                        print(f"Loading cached ICA results from {ica_path}")
                        ica = mne.preprocessing.read_ica(ica_path, verbose=False)
                        proba = np.load(proba_path)
                        ica.exclude = [i for i in range(proba.shape[0])
                                       if any(
                                proba[i, ICLABEL_CLASSES.index(c)] >= t for c, t in self.iclabel_thresholds.items())]
                        for i in range(proba.shape[0]):
                            self.ica_report.append(
                                {'recording': str(edf_path), 'component': i, 'excluded': i in ica.exclude,
                                 **dict(zip(ICLABEL_CLASSES, proba[i]))})
                    else:
                        fit_raw = raw.copy().filter(1.0, 100.0, fir_design='firwin', verbose=False)
                        fit_raw.notch_filter([50, 100], notch_widths=self.notch_width, fir_design='firwin',
                                             verbose=False)
                        ica = mne.preprocessing.ICA(**self.ica_params, verbose=False)
                        ica.fit(fit_raw, verbose=False)
                        proba = iclabel_label_components(fit_raw, ica, inplace=False)  # (n_components, 7)

                        print(f"Caching ICLabel results to {ica_path}")
                        ica.save(ica_path)
                        np.save(proba_path, proba)

                        ica.exclude = [i for i in range(proba.shape[0])
                                       if any(
                                proba[i, ICLABEL_CLASSES.index(c)] >= t for c, t in self.iclabel_thresholds.items())]

                        for i in range(proba.shape[0]):
                            self.ica_report.append(
                                {'recording': str(edf_path), 'component': i, 'excluded': i in ica.exclude,
                                 **dict(zip(ICLABEL_CLASSES, proba[i]))})
                        del fit_raw

                        print(f"ICLabel {edf_path.stem}: removed {len(ica.exclude)}/{proba.shape[0]} "
                              + ', '.join(
                            f"{i} ({ICLABEL_CLASSES[proba[i].argmax()]} {proba[i].max():.2f})" for i in ica.exclude))

                # Apply ICA to each recording
                if self.ica_method == 'manual':
                    ica = mne.preprocessing.ICA(n_components=self.ica_n_components, max_iter='auto', verbose=False)
                    ica.fit(raw, verbose=False)
                    if ica_dict[edf_path][1]:
                        ica.exclude = ica_dict[edf_path][1]
                        ica.apply(raw, verbose=False)
                    else:
                        print(f'No components found for {edf_path.stem}, skipping ICA...')

                # Apply filtering
                if self.highpass is not None or self.lowpass is not None:
                    print(f'Applying filter: ', end='')
                    if self.highpass is not None:
                        print(f'highpass={self.highpass}Hz ', end='')
                    if self.lowpass is not None:
                        print(f'lowpass={self.lowpass}Hz ', end='')
                    print(f'to {edf_path.stem}')

                    raw.filter(
                        l_freq=self.highpass,
                        h_freq=self.lowpass,
                        fir_design='firwin',
                        verbose=False
                    )

                    if self.notch is not None:
                        print(f'Applying notch filter at {self.notch}Hz to {edf_path.stem}')
                        raw.notch_filter(
                            freqs=self.notch,
                            notch_widths=self.notch_width,
                            fir_design='firwin',
                            verbose=False
                        )

                if ica is not None and ica.exclude:
                    ica.apply(raw, verbose=False)

                # Replace annotations with ChkEEG triggers (incl. eccentricity) if enabled
                if self.use_chkeeg_triggers and str(edf) in trigger_dict and trigger_dict[str(edf)] is not None:
                    trigger_data = trigger_dict[str(edf)]
                    print(f'INFO: Replacing annotations with NPZ triggers for {edf_path.stem}')

                    if trigger_data.get('codes') is not None and trigger_data.get('times') is not None:
                        codes = np.asarray(trigger_data['codes'], dtype=int)
                        times = np.asarray(trigger_data['times'])
                        eccentricities = np.asarray(trigger_data['eccentricity']) if trigger_data.get(
                            'eccentricity') is not None else None

                        times_samples = times.astype(int)
                        times_samples = np.clip(times_samples, 0, raw.n_times - 1)

                        if len(codes) > 0 and len(codes) == len(times_samples):
                            # Build annotation descriptions encoding eccentricity (e.g. 'DIN21/ecc_3')
                            descriptions = []
                            ecc_map = {}  # sample_idx -> eccentricity value for this recording
                            for i in range(len(codes)):
                                base_desc = f'DIN{int(codes[i])}/rec_{edf_path.stem}'
                                if eccentricities is not None and i < len(eccentricities):
                                    ecc_val = eccentricities[i]
                                    descriptions.append(f'{base_desc}/ecc_{ecc_val}')
                                    ecc_map[int(times_samples[i])] = ecc_val
                                else:
                                    descriptions.append(base_desc)

                            # Store eccentricity mapping for this recording
                            self._trigger_eccentricities[str(edf)] = ecc_map

                            # Convert sample indices to seconds for MNE Annotations
                            onsets_sec = times_samples / raw.info['sfreq']
                            durations = np.zeros(len(codes))

                            # Clear existing annotations, then set new ones
                            raw.set_annotations(mne.Annotations(onset=[], duration=[], description=[]))
                            ann = mne.Annotations(onset=onsets_sec, duration=durations, description=descriptions)
                            raw.set_annotations(ann)

                            n_with_ecc = sum(1 for d in descriptions if '/ecc_' in d)
                            print(
                                f'INFO: Added {len(descriptions)} trigger annotations to {edf_path.stem} ({n_with_ecc} with eccentricity)')
                        else:
                            print(f'WARNING: Missing/invalid trigger data for {edf_path.stem}')

                raws_list.append(raw)

        finally:
            os.chdir(prev_cwd)

        if not raws_list:
            print('WARNING: No data was loaded. Returning None.')
            return None

        # Concatenate raw objects
        concatenated_raw = mne.concatenate_raws(raws_list)

        return concatenated_raw

    def create_epochs(self, manual_baseline: bool = False):
        '''
        Process raw objects into epochs for a single directory.
        
        Parameters:
        manual_baseline (bool): If True, applies per-trial manual baselining. If False, uses MNE's built-in baselining.
        
        Returns:
        combined_unaveraged_epochs (dict): Dictionary of unaveraged epochs by mismatch type (Epochs objects).
        combined_averaged_epochs (dict): Dictionary of averaged epochs by mismatch type (Evoked objects).
        '''

        raw = self.preprocess_dir()

        if raw is None:
            print('ERROR: No data was preprocessed.')
            return {}, {}

        # Use internal method for epoch creation
        unaveraged_epochs = self._create_epochs_from_raw(raw, apply_mne_baseline=not manual_baseline)

        if not unaveraged_epochs:
            print('ERROR: No epochs created.')
            return {}, {}

        # Apply manual baselining if requested
        if manual_baseline and self.baseline_win:
            for trigger_type in unaveraged_epochs:
                unaveraged_epochs[trigger_type] = self._apply_manual_baseline(unaveraged_epochs[trigger_type])

        # Crop to time window (exclude baseline period)
        if self.time_window != self.time_window_with_baseline:
            for trigger_type, epochs in unaveraged_epochs.items():
                epochs.crop(tmin=self.time_window[0], tmax=self.time_window[1])

        # Create averaged epochs (Evoked objects)
        averaged_epochs = {}
        for trigger_type, epochs in unaveraged_epochs.items():
            averaged_epochs[trigger_type] = epochs.average()

        return unaveraged_epochs, averaged_epochs

    # Method to preprocess & convert epochs to arrays used by spectral analyses (iterate over all recordings in a specified directory)
    def superlet_input(self):
        """
        Convert a dictionary of epoch objects to numpy arrays suitable for superlet analysis.
        
        Returns:
        result (dict): Dictionary with trigger types as keys and numpy arrays as values.
            Each array has shape [channels, trials, time] for easy channel-wise processing.
        sampling_rate (int): The sampling rate of the processed recording based on MNE object info.
        """

        # Get preprocessed epochs
        unaveraged_epochs, _ = self.create_epochs()
        if not unaveraged_epochs:
            print('ERROR: No epochs found to process.')
            return {}

        # Convert each mismatch type to appropriate numpy array
        result = {}
        for mismatch_type, epochs in unaveraged_epochs.items():
            # Convert to numpy array [n epochs, n channels, n time]
            data = epochs.get_data()

            # Transpose to [channels, trials, time] for easier channel extraction
            data = np.transpose(data, (1, 0, 2))

            # Store in a result dictionary
            result[mismatch_type] = data
            print(
                f"Prepared {mismatch_type} data: {data.shape[0]} channels, {data.shape[1]} trials, {data.shape[2]} timepoints")
            print(f'Array shape: {data.shape}')

        # Get sampling rate from the first Epochs object in the unaveraged_epochs dict
        first_epochs = next(iter(unaveraged_epochs.values()))
        sampling_rate = int(first_epochs.info['sfreq'])

        return result, sampling_rate

    ##################################################
    ## --    HIGHER LEVEL DIRECTORY PROCESSING     --
    ##################################################
    def _check_existing_selections(self,
                                   base_directory: Path,
                                   excluded_subjects_lower: Optional[List[str]],
                                   included_conditions_lower: Optional[List[str]],
                                   recordings_per_condition: Optional[int],
                                   selection_type: str) -> Tuple[bool, Dict[str, Dict], List[str]]:
        """
        Check if all recordings across all subjects/conditions have existing pkl (ICA / bad channel) selections.
        
        Parameters:
        base_directory: Base directory containing subject folders.
        excluded_subjects_lower: Lowercase list of excluded subjects.
        included_conditions_lower: Lowercase list of included conditions.
        recordings_per_condition: Number of recordings per condition limit.
        selection_type: 'bad_ch_selection' or 'ica_selection'.
        
        Returns:
        Tuple of:
            all_have_data (bool): True if all recordings have existing selections.
            selections (dict): Mapping of resolved condition folder path (str) -> loaded pkl data.
            missing (list): List of human-readable strings identifying recordings without selections.
        """
        all_have_data = True
        selections_by_condition = {}  # resolved condition path (str) -> {edf_path: selection_data}
        missing_recordings = []  # human-readable list of recordings without selections

        for subject_folder in base_directory.iterdir():
            if not subject_folder.is_dir():
                continue
            if excluded_subjects_lower and subject_folder.name.lower() in excluded_subjects_lower:
                continue

            for condition_folder in subject_folder.iterdir():
                if not condition_folder.is_dir():
                    continue
                if included_conditions_lower and condition_folder.name.lower() not in included_conditions_lower:
                    continue

                # Get recording folders for this condition
                recording_folders = self._select_recording_folders(
                    condition_folder, condition_folder.name, recordings_per_condition
                )
                if not recording_folders:
                    continue

                # Count total EDF files across selected recording folders
                edf_paths_in_condition = []
                for rf in recording_folders:
                    edf_paths_in_condition.extend(edf_from_dir(str(rf)))

                if not edf_paths_in_condition:
                    continue

                # Use resolved path string as key for consistent dictionary lookup
                condition_key = str(condition_folder.resolve())

                # Check for existing pkl files (one pkl per EDF file)
                existing_pkl = extract_pkl_from_directory(str(condition_folder), selection_type)

                # Build set of EDF filenames that DO have a pkl
                edf_stems_with_pkl = set()
                for pkl_path in existing_pkl:
                    try:
                        with open(pkl_path, 'rb') as pkl:
                            pkl_data = pickle.load(pkl)
                            edf_filename = next(iter(pkl_data))
                            edf_stems_with_pkl.add(Path(edf_filename).stem)
                    except Exception:
                        pass

                # Identify which EDF files are missing selections
                condition_missing = []
                for edf_path in edf_paths_in_condition:
                    if Path(edf_path).stem not in edf_stems_with_pkl:
                        condition_missing.append(edf_path)

                if condition_missing:
                    all_have_data = False
                    selections_by_condition[condition_key] = None
                    for edf_path in condition_missing:
                        missing_recordings.append(
                            f'{subject_folder.name}/{condition_folder.name}/{Path(edf_path).stem}'
                        )
                else:
                    # All EDF files have selections - load the pkl data
                    condition_selections = {}
                    for pkl_path in existing_pkl:
                        try:
                            with open(pkl_path, 'rb') as pkl:
                                pkl_data = pickle.load(pkl)
                                edf_filename = next(iter(pkl_data))
                                edf_absolute_path = Path(pkl_path).with_name(edf_filename)
                                condition_selections[edf_absolute_path] = pkl_data[edf_filename]
                        except Exception as e:
                            print(f'WARNING: Failed reading pkl file: {pkl_path} ({e})')
                    selections_by_condition[condition_key] = condition_selections

        return all_have_data, selections_by_condition, missing_recordings

    def process_subjects(self,
                         excluded_subjects: Optional[List[str]] = None,
                         included_conditions: Optional[List[str]] = None,
                         recordings_per_condition: Optional[int] = None) -> Dict[str, Dict[str, mne.io.Raw]]:
        '''
        Process subjects from a specified directory. By default, data found for all subjects found in the directory are processed.
        High-level wrapper around preprocess_dir that preprocesses multiple subjects to create a pooled dataset.
        Directory structure: DIRECTORY -> subject_name -> condition_name -> individual recording (folder)
        
        Parameters:
        excluded_subjects (list): A list of subject names to exclude. If None, all subjects are processed.
        included_conditions (list): A list of condition names to include. If None, all conditions are processed.
        recordings_per_condition (int): Number of recording folders to use per condition (1-4).
            For baseline conditions: uses the LAST x recordings (sorted chronologically by folder name).
            For post-patching conditions: uses the FIRST x recordings (sorted chronologically by folder name).
            If None, all recordings in the condition folder are processed.
        
        Returns:
        result (dict): Nested dictionary with structure {subject_name: {condition_name: concatenated_raw_object}}
        '''

        # Validate recordings_per_condition
        if recordings_per_condition is not None:
            if not isinstance(recordings_per_condition, int) or not 1 <= recordings_per_condition <= 4:
                raise ValueError(
                    f'recordings_per_condition must be an integer between 1 and 4, got: {recordings_per_condition}')

        result = {}
        base_directory = Path(self.directory)

        # Normalize included_conditions to lowercase for case-insensitive matching
        excluded_subjects_lower = [c.lower() for c in excluded_subjects] if excluded_subjects else None
        included_conditions_lower = [c.lower() for c in included_conditions] if included_conditions else None

        # --- Check for existing bad channel / ICA selections across ALL subjects/conditions ---
        skip_bad_channel_prompts = False
        skip_ica_prompts = False
        preloaded_bad_channels = {}  # condition_folder -> {edf_path: bad_channels}
        preloaded_ica = {}  # condition_folder -> {edf_path: (n_components, excluded)}

        if self.bad_channels:
            all_have_bad_ch, preloaded_bad_channels, missing_bad_ch = self._check_existing_selections(
                base_directory, excluded_subjects_lower, included_conditions_lower,
                recordings_per_condition, 'bad_ch_selection'
            )
            if all_have_bad_ch and preloaded_bad_channels:
                print('\n=== Bad channel selections found for ALL recordings ===')
                print('Using existing bad channel selections for all recordings.\n')
                skip_bad_channel_prompts = True
            elif missing_bad_ch:
                print(f'\n=== Bad channel selections MISSING for {len(missing_bad_ch)} recording(s) ===')
                for entry in missing_bad_ch:
                    print(f'  - {entry}')

        if self.ica_method == 'manual':
            all_have_ica, preloaded_ica, missing_ica = self._check_existing_selections(
                base_directory, excluded_subjects_lower, included_conditions_lower,
                recordings_per_condition, 'ica_selection'
            )
            if all_have_ica and preloaded_ica:
                print('\n=== ICA selections found for ALL recordings ===')
                print('Using existing ICA selections for all recordings.\n')
                skip_ica_prompts = True
            elif missing_ica:
                print(f'\n=== ICA selections MISSING for {len(missing_ica)} recording(s) ===')
                for entry in missing_ica:
                    print(f'  - {entry}')

        # Iterate over subject folders (first level below base directory)
        for subject_folder in base_directory.iterdir():
            if not subject_folder.is_dir():
                continue

            subject_name = subject_folder.name

            # Skip excluded subjects
            if excluded_subjects_lower and subject_name.lower() in excluded_subjects_lower:
                print(f'INFO: Skipping excluded subject: {subject_name}')
                continue

            print(f'\n--- Processing subject: {subject_name} ---')
            result[subject_name] = {}

            # Iterate over condition folders (second level)
            for condition_folder in subject_folder.iterdir():
                if not condition_folder.is_dir():
                    continue

                condition_name = condition_folder.name

                # Skip conditions not in included_conditions (case-insensitive matching)
                if included_conditions_lower and condition_name.lower() not in included_conditions_lower:
                    print(f'INFO: Skipping condition not in included list: {condition_name}')
                    continue

                print(f'  Processing condition: {condition_name}')

                # Get recording folders to process (filtered by recordings_per_condition if set)
                recording_folders = self._select_recording_folders(
                    condition_folder, condition_name, recordings_per_condition
                )

                if not recording_folders:
                    print(f'  WARNING: No recording folders found for {subject_name}/{condition_name}')
                    continue

                # Select ICA components and bad channels at condition level
                # Collect all EDF paths across all selected recording folders for this condition
                condition_edf_paths = []
                for recording_folder in recording_folders:
                    folder_edfs = edf_from_dir(str(recording_folder))
                    condition_edf_paths.extend(folder_edfs)

                # Use resolved path string as key for consistent dictionary lookup
                condition_key = str(condition_folder.resolve())

                # Get bad channel selections for this condition
                bad_channel_selection = None
                if self.bad_channels and condition_edf_paths:
                    if preloaded_bad_channels.get(condition_key) is not None:
                        bad_channel_selection = preloaded_bad_channels[condition_key]
                    elif self.interactive_selection:
                        bad_channel_selection = get_bad_channels(str(condition_folder), condition_edf_paths)
                    else:
                        print(
                            f'  WARNING: no manual bad channel selection for some recordings of {subject_name}/{condition_name}; '
                            f'using automatic detection only.')
                        bad_channel_selection = {}

                        # Get ICA selections for this condition
                ica_selection = None
                if self.ica_method == 'manual' and condition_edf_paths:
                    if skip_ica_prompts and condition_key in preloaded_ica:
                        ica_selection = preloaded_ica[condition_key]
                    else:
                        ica_selection = get_ica(
                            directory_path=str(condition_folder),
                            n_components=self.ica_n_components,
                            edf_paths=condition_edf_paths
                        )

                # Process selected recording folders
                raws_list = []
                for recording_folder in recording_folders:
                    original_directory = self.directory
                    self.directory = str(recording_folder)

                    try:
                        raw_result = self.preprocess_dir(
                            bad_channel_selection=bad_channel_selection,
                            ica_selection=ica_selection
                        )

                        if raw_result is not None:
                            raws_list.append(raw_result)
                        else:
                            print(f'    WARNING: No data from {recording_folder.name}')

                    except Exception as e:
                        print(f'    ERROR: Failed to process {recording_folder.name}: {e}')

                    finally:
                        self.directory = original_directory

                # Concatenate raw objects from all selected recordings
                if raws_list:
                    if len(raws_list) == 1:
                        concatenated_raw = raws_list[0]
                    else:
                        concatenated_raw = mne.concatenate_raws(raws_list)

                    result[subject_name][condition_name] = concatenated_raw
                    print(
                        f'  Successfully processed {condition_name}: {concatenated_raw.n_times} samples from {len(raws_list)} recordings')
                else:
                    print(f'  WARNING: No data found for {subject_name}/{condition_name}')

            # Remove subject from result if no conditions were successfully processed
            if not result[subject_name]:
                print(f'WARNING: No valid conditions found for subject {subject_name}')
                del result[subject_name]

        print(f'\n=== Finished processing {len(result)} subjects ===')
        return result

    def _select_recording_folders(self,
                                  condition_folder: Path,
                                  condition_name: str,
                                  recordings_per_condition: Optional[int]) -> List[Path]:
        """
        Select which recording folders to process based on condition type and limit.
        
        For baseline conditions: returns the LAST x recordings (sorted by folder name).
        For post-patching conditions: returns the FIRST x recordings (sorted by folder name).
        
        Parameters:
        condition_folder (Path): Path to the condition folder containing recording subfolders.
        condition_name (str): Name of the condition (e.g., 'Baseline', 'Post-patching').
        recordings_per_condition (int): Number of recordings to use. If None, returns all.
        
        Returns:
        recording_folders (list): List of Path objects for recording folders to process.
        """
        # Get all subdirectories (recording folders), excluding hidden folders
        all_recording_folders = [
            f for f in condition_folder.iterdir()
            if f.is_dir() and not f.name.startswith('.')
        ]

        if not all_recording_folders:
            return []

        # Sort folders by name (chronological order for names like "09_05", "09_12", etc.)
        all_recording_folders.sort(key=lambda x: x.name)

        # If no limit specified, return all folders
        if recordings_per_condition is None:
            print(f'    Using all {len(all_recording_folders)} recording folders')
            return all_recording_folders

        # Determine selection based on condition type (case-insensitive)
        condition_lower = condition_name.lower()

        if 'baseline' in condition_lower:
            # Baseline: use the LAST x recordings (closest to patching)
            selected = all_recording_folders[-recordings_per_condition:]
            print(f'    Baseline: selecting last {len(selected)} of {len(all_recording_folders)} recordings')
            print(f'    Selected folders: {[f.name for f in selected]}')
        elif 'post' in condition_lower:
            # Post-patching: use the FIRST x recordings (closest to patching)
            selected = all_recording_folders[:recordings_per_condition]
            print(f'    Post-patching: selecting first {len(selected)} of {len(all_recording_folders)} recordings')
            print(f'    Selected folders: {[f.name for f in selected]}')
        else:
            # Unknown condition: default to using first x recordings
            selected = all_recording_folders[:recordings_per_condition]
            print(
                f'    Unknown condition type: selecting first {len(selected)} of {len(all_recording_folders)} recordings')

        return selected

    def create_subject_epochs(self,
                              subject_data: Dict[str, Dict[str, mne.io.Raw]],
                              output_format: str = 'epochs',
                              manual_baseline: bool = True) -> Dict[str, Dict[str, Dict[str, any]]]:
        """
        Process raw objects from preprocess_subjects into epochs or numpy arrays.
        Applies manual per-trial baselining: for each trial, the mean voltage in the 
        baseline window is calculated and subtracted from every data point in that trial.
        
        Parameters:
        subject_data (dict): Nested dictionary from preprocess_subjects with structure 
                            {subject_name: {condition_name: Raw}}
        output_format (str): Output format - 'epochs' for MNE Epochs objects, 'array' for numpy arrays.
                            If 'array', shape will be [n_channels, n_trials, n_times]
        manual_baseline (bool): If True, applies per-trial manual baselining. If False, uses MNE's built-in baselining.
        
        Returns:
        result (dict): Nested dictionary with structure:
                        {subject_name: {condition_name: {trigger_type: Epochs or np.ndarray}}}
        """
        if output_format not in ['epochs', 'array']:
            raise ValueError(f"output_format must be 'epochs' or 'array', got: {output_format}")

        if manual_baseline and not self.baseline_win:
            print(
                'WARNING: Manual baselining requested but no baseline window defined. Baselining will not be applied.')

        result = {}

        for subject_name, conditions in subject_data.items():
            print(f'\n--- Creating epochs for subject: {subject_name} ---')
            result[subject_name] = {}

            for condition_name, raw in conditions.items():
                print(f'  Processing condition: {condition_name}')
                result[subject_name][condition_name] = {}
                context = {
                    'subject_name': subject_name,
                    'conditions': condition_name,
                }

                try:
                    # Create epochs for each trigger type
                    epochs_by_trigger = self._create_epochs_from_raw(raw, apply_mne_baseline=not manual_baseline,
                                                                     context=context)

                    if not epochs_by_trigger:
                        print(f'  WARNING: No epochs created for {subject_name}/{condition_name}')
                        continue

                    for trigger_type, epochs in epochs_by_trigger.items():
                        # Apply manual per-trial baselining if requested
                        if manual_baseline and self.baseline_win:
                            epochs = self._apply_manual_baseline(epochs)

                        # Crop to time window (exclude baseline period)
                        if self.time_window:
                            epochs.crop(tmin=self.time_window[0], tmax=self.time_window[1])

                        if output_format == 'array':
                            # Convert to numpy array [n_channels, n_trials, n_times]
                            data = epochs.get_data()  # [n_trials, n_channels, n_times]
                            data = np.transpose(data, (1, 0, 2))  # [n_channels, n_trials, n_times]
                            result[subject_name][condition_name][trigger_type] = data
                            print(
                                f'    {trigger_type}: {data.shape[1]} trials, {data.shape[0]} channels, {data.shape[2]} timepoints')
                        else:
                            result[subject_name][condition_name][trigger_type] = epochs
                            print(f'    {trigger_type}: {len(epochs)} epochs')

                except Exception as e:
                    print(f'  ERROR: Failed to create epochs for {subject_name}/{condition_name}: {e}')
                    continue

        print(f'\n=== Finished creating epochs for {len(result)} subjects ===')
        return result

    def _create_epochs_from_raw(self, raw: mne.io.Raw, apply_mne_baseline: bool = False,
                                context: Dict[str, str] = None) -> Dict[str, mne.Epochs]:
        """
        Create epochs from a raw object, organized according to self.split_by.

        Parameters:
        raw (mne.io.Raw): Preprocessed raw object.
        apply_mne_baseline (bool): If True, applies MNE's built-in baselining. If False, no baselining is applied (for manual baselining later).

        Returns:
        epochs_dict (dict): Dictionary mapping split keys to Epochs objects.
            Keys depend on self.split_by:
                'sides'        -> left, center, right, global
                'eccentricity' -> ecc_1, ecc_2, ecc_3, ... (excludes ecc=0 and global)
                'both'         -> left_1, left_2, ..., global (global has no eccentricity subdivision)
        """
        # Step 1: Create epochs per side (always needed as base)
        epochs_by_side = self._epoch_by_side(raw, apply_mne_baseline, context=context)

        if not epochs_by_side:
            return {}

        # Step 2: Apply split_by logic
        if self.split_by == 'sides':
            return epochs_by_side
        elif self.split_by == 'eccentricity':
            return self._split_by_eccentricity(epochs_by_side)
        elif self.split_by == 'both':
            return self._split_by_both(epochs_by_side)
        else:
            return epochs_by_side

    def _epoch_by_side(self, raw, apply_mne_baseline=False, context: Dict[str, str] = None):
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        if len(events) == 0:
            return {}
        tmin, tmax = self.time_window_with_baseline or self.time_window
        baseline = self.baseline_win if apply_mne_baseline else None

        # event names per side, and the union of all of them
        side_names = {side: [name for name in event_id if trig == name.split('/')[0]] for side, trig in
                      self.triggers.items()}
        all_ids = {name: event_id[name] for names in side_names.values() for name in names}

        epochs = mne.Epochs(raw, events, event_id=all_ids, tmin=tmin, tmax=tmax, picks='eeg',
                            baseline=baseline, preload=True, event_repeated='merge', verbose=False)

        ar_interpolated = None
        if self.autoreject_mode == 'local':
            raw_hp = raw.copy().filter(l_freq=1, h_freq=None, fir_design='firwin', verbose=False)
            fit_epochs = mne.Epochs(raw_hp, events, event_id=all_ids, tmin=tmin, tmax=tmax, picks='eeg',
                                    baseline=baseline, preload=True, event_repeated='merge', verbose=False)
            assert np.array_equal(fit_epochs.events, epochs.events)
            fit_epochs.crop(tmin=self.autoreject_window[0], tmax=self.autoreject_window[1])
            ar = AutoReject(picks='eeg', verbose=False, **self.autoreject_params)
            ar.fit(fit_epochs)
            reject_log = ar.get_reject_log(fit_epochs)
            # apply the same per-epoch decisions to the full-length (extended) epochs
            epochs = ar.transform(epochs, reject_log=reject_log)

            kept = ~reject_log.bad_epochs
            per_epoch = (reject_log.labels[kept] == 2).sum(axis=1)  # number of interpolated channels per kept epoch

            # select interpolated channels for each kept epoch
            ar_interpolated = [[reject_log.ch_names[j] for j in np.where(reject_log.labels[i] == 2)[0]] for i in np.where(kept)[0]]
            assert len(ar_interpolated) == len(epochs)
            self.autoreject_report.append({'n_epochs': len(kept),
                                           'n_dropped': int((~kept).sum()),
                                           'mean_interp_per_kept_epoch': float(per_epoch.mean()) if len(per_epoch) else 0,
                                           'max_interp_per_kept_epoch': int(per_epoch.max()) if len(per_epoch) else 0,
                                           'n_interpolate': ar.n_interpolate_['eeg'], 'consensus': ar.consensus_['eeg'],
                                           **(context or {})
                                           })

        elif self.autoreject_mode == 'global':
            epochs.drop_bad(reject=get_rejection_threshold(epochs, ch_types='eeg', verbose=False), verbose=False)

        epochs = self._attach_trial_metadata(epochs, event_id, ar_interpolated=ar_interpolated, first_samp=raw.first_samp)
        return {side: epochs[[n for n in names if n in epochs.event_id]]
                for side, names in side_names.items()
                if any(n in epochs.event_id for n in names) and len(epochs[[n for n in names if n in epochs.event_id]])}

    def _split_by_eccentricity(self, epochs_by_side: Dict[str, mne.Epochs]) -> Dict[str, mne.Epochs]:
        """
        Re-organize epochs by eccentricity value. Excludes global stimuli and eccentricity=0.
        Pools all local sides together and groups by eccentricity.
        
        Parameters:
        epochs_by_side (dict): Base epochs dict keyed by side (from _epoch_by_side).
        
        Returns:
        result (dict): Epochs keyed by eccentricity, e.g. {'ecc_1': Epochs, 'ecc_2': Epochs, ...}
        """
        result = {}

        # Collect all local (non-global) epochs into one list
        local_epochs_list = []
        for side, epochs in epochs_by_side.items():
            if side == 'global':
                continue  # global stimuli excluded in eccentricity mode
            if epochs.metadata is not None and 'eccentricity' in epochs.metadata.columns:
                local_epochs_list.append(epochs)
            else:
                print(f'    WARNING: No eccentricity metadata for "{side}" epochs, skipping in eccentricity split.')

        if not local_epochs_list:
            print('    WARNING: No local epochs with eccentricity data found.')
            return result

        # Concatenate all local epochs
        all_local = mne.concatenate_epochs(local_epochs_list)

        # Get unique eccentricity values (excluding 0 and None)
        ecc_values = all_local.metadata['eccentricity'].dropna().unique()
        ecc_values = sorted([v for v in ecc_values if v != 0])

        if not ecc_values:
            print('    WARNING: No non-zero eccentricity values found.')
            return result

        for ecc_val in ecc_values:
            mask = all_local.metadata['eccentricity'] == ecc_val
            sub_epochs = all_local[mask]
            if len(sub_epochs) > 0:
                key = f'ecc_{ecc_val}'
                result[key] = sub_epochs
                print(f'    {key}: {len(sub_epochs)} epochs')

        return result

    def _split_by_both(self, epochs_by_side: Dict[str, mne.Epochs]) -> Dict[str, mne.Epochs]:
        """
        Re-organize epochs by both side AND eccentricity.
        Local triggers: split into side_ecc keys (e.g. 'left_1', 'left_2'). Excludes eccentricity=0.
        Global triggers: kept as 'global' (no eccentricity subdivision).
        
        Parameters:
        epochs_by_side (dict): Base epochs dict keyed by side (from _epoch_by_side).
        
        Returns:
        result (dict): Epochs keyed by side+eccentricity, e.g. {'left_1': Epochs, ..., 'global': Epochs}
        """
        result = {}

        for side, epochs in epochs_by_side.items():
            if side == 'global':
                # Global stimuli cannot have eccentricity -> keep as-is
                result['global'] = epochs
                print(f'    global: {len(epochs)} epochs')
                continue

            # For local triggers: split by eccentricity
            if epochs.metadata is None or 'eccentricity' not in epochs.metadata.columns:
                print(f'    WARNING: No eccentricity metadata for "{side}" epochs, skipping split.')
                continue

            ecc_values = epochs.metadata['eccentricity'].dropna().unique()
            ecc_values = sorted([v for v in ecc_values if v != 0])

            for ecc_val in ecc_values:
                mask = epochs.metadata['eccentricity'] == ecc_val
                sub_epochs = epochs[mask]
                if len(sub_epochs) > 0:
                    key = f'{side}_{ecc_val}'
                    result[key] = sub_epochs
                    print(f'    {key}: {len(sub_epochs)} epochs')

        return result

    def _attach_trial_metadata(self, epochs: mne.Epochs, event_id: Dict,
                               ar_interpolated: Optional[List[List[str]]] = None,
                               first_samp: int = 0) -> mne.Epochs:
        """
        Attaches one metadata row per epoch:
            eccentricity      parsed from 'DIN21/rec_<stem>/ecc_3' (None if absent)
            recording         EDF stem the epoch came from (None for EDF-annotation fallback)
            onset_s           onset in seconds within the concatenated session
            rec_interpolated  channels interpolated for the whole recording (manual + auto), ';'-separated
            ar_interpolated   channels autoreject interpolated in this epoch, ';'-separated
        Channel names are the EGI names (e.g. 'EEG 52').
        """
        code_to_desc = {code: desc for desc, code in event_id.items()}
        sfreq = epochs.info['sfreq']
        rows = []
        for k, event in enumerate(epochs.events):
            desc = code_to_desc.get(event[2], '')

            ecc_val = None
            if '/ecc_' in desc:
                ecc_str = desc.split('/ecc_')[-1]
                try:
                    ecc_val = float(ecc_str) if '.' in ecc_str else int(ecc_str)
                except (ValueError, TypeError):
                    ecc_val = ecc_str

            recording = desc.split('/rec_')[1].split('/')[0] if '/rec_' in desc else None

            rows.append({
                'eccentricity': ecc_val,
                'recording': recording,
                'onset_s': (event[0] - first_samp) / sfreq,
                'rec_interpolated': ';'.join(self.interpolation_log.get(recording, [])) if recording else None,
                'ar_interpolated': ';'.join(ar_interpolated[k]) if ar_interpolated is not None else '',
            })

        epochs.metadata = pd.DataFrame(rows)
        return epochs

    def _apply_manual_baseline(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Apply manual per-trial baselining to epochs.
        For each trial, calculates the mean voltage in the baseline window
        and subtracts it from every data point in that trial.
        
        Parameters:
        epochs (mne.Epochs): Epochs object to baseline.
        
        Returns:
        epochs (mne.Epochs): Baselined epochs object (modified in place).
        """
        if not self.baseline_win:
            return epochs

        baseline_start, baseline_end = self.baseline_win

        # Get the data [n_trials, n_channels, n_times]
        data = epochs.get_data(copy=True)
        times = epochs.times

        # Find indices for baseline window
        baseline_mask = (times >= baseline_start) & (times <= baseline_end)

        if not np.any(baseline_mask):
            print(
                f'    WARNING: Baseline window ({baseline_start}, {baseline_end}) outside epoch time range ({times[0]:.3f}, {times[-1]:.3f})')
            return epochs

        # Apply per-trial, per-channel baselining
        # Calculate mean across baseline timepoints for each trial and channel, then subtract
        baseline_means = data[:, :, baseline_mask].mean(axis=2, keepdims=True)
        data -= baseline_means

        # Update epochs data
        epochs._data = data

        return epochs

    def create_epochs_for_single_subject(self,
                                         subject_name: str,
                                         manual_baseline: bool = True,
                                         included_conditions: Optional[List[str]] = None,
                                         skip_crop: bool = True) -> Dict[str, Dict[str, mne.Epochs]]:
        """
        Load and create epochs for a SINGLE subject only. Memory-efficient: only one subject's data in RAM at a time.
        Used in morlet_tfr for single trial TFR computation.
        
        Parameters:
        subject_name (str): Name of the subject folder to process.
        manual_baseline (bool): If True, applies per-trial manual baselining.
        included_conditions (list): If specified, only process these condition folders.
        skip_crop (bool): If True, skip final cropping.
        
        Returns:
        result (dict): {condition_name: {trigger_type: Epochs}} for this subject only.
        """
        base_directory = Path(self.directory)
        subject_folder = base_directory / subject_name

        if not subject_folder.is_dir():
            raise ValueError(f"Subject folder not found: {subject_folder}")

        result = {}

        # Normalize included_conditions for case-insensitive matching
        included_lower = [c.lower() for c in included_conditions] if included_conditions else None

        # Iterate over condition folders
        for condition_folder in subject_folder.iterdir():
            if not condition_folder.is_dir():
                continue

            condition_name = condition_folder.name

            # Skip conditions not in the included list
            if included_lower and condition_name.lower() not in included_lower:
                continue

            result[condition_name] = {}

            # Temporarily change directory to process this condition
            original_directory = self.directory
            self.directory = str(condition_folder)

            try:
                raw = self.preprocess_dir()

                if raw is None:
                    continue

                # Create epochs
                epochs_by_trigger = self._create_epochs_from_raw(raw, apply_mne_baseline=not manual_baseline)

                # Apply manual baselining and crop
                for trigger_type, epochs in epochs_by_trigger.items():
                    if manual_baseline and self.baseline_win:
                        epochs = self._apply_manual_baseline(epochs)

                    if self.time_window and not skip_crop:
                        epochs.crop(tmin=self.time_window[0], tmax=self.time_window[1])

                    result[condition_name][trigger_type] = epochs

                del raw

            finally:
                self.directory = original_directory

        return result

    def get_subject_names(self,
                          excluded_subjects: Optional[List[str]] = None,
                          included_conditions: Optional[List[str]] = None) -> List[str]:
        """
        Get list of subject names from the directory without loading any data.
        
        Parameters:
        excluded_subjects (list): Subject names to exclude.
        included_conditions (list): If specified, only include subjects that have these conditions.
        
        Returns:
        subject_names (list): List of subject folder names.
        """
        base_directory = Path(self.directory)
        excluded_lower = [s.lower() for s in excluded_subjects] if excluded_subjects else []

        subject_names = []
        for folder in base_directory.iterdir():
            if not folder.is_dir():
                continue

            if folder.name.lower() in excluded_lower:
                continue

            if included_conditions:
                has_conditions = any(
                    (folder / cond).is_dir()
                    for cond in included_conditions
                )
                if not has_conditions:
                    continue

            subject_names.append(folder.name)

        return sorted(subject_names)
