# Create initial CSV file; read existing file

'''
Data hierarchy:
- Subject: a single participant in the study
- Session: a collection of observations made during a single visit to the lab (usually 1 / subject)
- Recording (/measurement): a single observation (EDF + related auxilliary files)

CSV types:
- Subjects data file: containing data for each subject (row: all sessions averaged / subject)
- Sessions data file: containing data for each session (row: all recordings averaged / session)
- Recordings data file: containing data for each recording (row: single observation)

Data columns:
- Location: location of the file (for single observations) / directory (for sessions and subjects)
- Subject: subject ID
- Session: session directory name
- Date: date of the session (YYYY-MM-DD)
- Num_sessions: number of sessions for the subject
- Num_recordings: number of recordings for the session

- Amplitude: amplitude of the peak (in µV)
- Time: time of the peak (in seconds)
- Baseline_STD: std of the baseline (in µV)

Mismatch abbreviations:
- LOC: Local mismatch
- GLO: Global mismatch
- LOC_ALL: All local mismatches pooled
- LOC_L: Left local mismatch
- LOC_T: Top local mismatch
- LOC_R: Right local mismatch
'''

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import hashlib
import json
import mne
import logging
import gzip
import pickle


class SimpleCache:
    """Simplified hashing-based cache using compressed pickles."""
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    @staticmethod
    def compute_hash(params: Dict) -> str:
        """Compute SHA256 hash of a dictionary of parameters."""
        p_str = json.dumps(params, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(p_str).hexdigest()[:16]

    def get_path(self, identifier: str, params: Dict) -> Path:
        h = self.compute_hash(params)
        return self.cache_dir / f"{identifier}_{h}.pkl.gz"

    def save(self, data, identifier: str, params: Dict):
        if not self.enabled: return
        path = self.get_path(identifier, params)
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, identifier: str, params: Dict):
        if not self.enabled: return None
        path = self.get_path(identifier, params)
        if path.exists():
            try:
                with gzip.open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

logger = logging.getLogger(__name__)



def new_subjects_csv(csv_file):
    '''
    Create a new subjects data file CSV with appropriate headers.
    
    Parameters:
    csv_file (str): the name of the subjects CSV file to be created.
    '''

    df = pd.DataFrame(columns=['location', 'subject', 'date', 'num_sessions', 'num_recordings', 
                               'LOC_ALL_positive_amplitude', 'LOC_ALL_positive_time',
                               'LOC_ALL_negative_amplitude', 'LOC_ALL_negative_time',
                               'LOC_L_positive_amplitude', 'LOC_L_positive_time',
                               'LOC_T_positive_amplitude', 'LOC_T_positive_time',
                               'LOC_R_positive_amplitude', 'LOC_R_positive_time',
                               'GLO_negative_amplitude', 'GLO_negative_time',
                               'baseline_STD',])
    
    df.to_csv(csv_file, index=False)

def new_sessions_csv(csv_file):
    '''
    Create a new sessions data file CSV with appropriate headers.

    Parameters:
    csv_file (str): the name of the sessions CSV file to be created.
    '''

    df = pd.DataFrame(columns=['location', 'subject', 'session', 'date', 'num_recordings',
                               'LOC_ALL_positive_amplitude', 'LOC_ALL_positive_time',
                               'LOC_ALL_negative_amplitude', 'LOC_ALL_negative_time',
                               'LOC_L_positive_amplitude', 'LOC_L_positive_time',
                               'LOC_T_positive_amplitude', 'LOC_T_positive_time',
                               'LOC_R_positive_amplitude', 'LOC_R_positive_time',
                               'GLO_negative_amplitude', 'GLO_negative_time',
                               'baseline_STD',])
    
    df.to_csv(csv_file, index=False)
    print(f'New sessions data file created: {csv_file}')

def new_recordings_csv(csv_file):
    '''
    Create a new recordings data file CSV with appropriate headers.

    Parameters:
    csv_file (str): the name of the recordings CSV file to be created.
    '''

    df = pd.DataFrame(columns=['location', 'subject', 'session', 'recording_data', 'date',
                            'LOC_ALL_positive_amplitude', 'LOC_ALL_positive_time',
                            'LOC_ALL_negative_amplitude', 'LOC_ALL_negative_time',
                            'LOC_L_positive_amplitude', 'LOC_L_positive_time',
                            'LOC_T_positive_amplitude', 'LOC_T_positive_time',
                            'LOC_R_positive_amplitude', 'LOC_R_positive_time',
                            'GLO_negative_amplitude', 'GLO_negative_time',
                            'baseline_STD'])
    
    df.to_csv(csv_file, index=False)
    print(f'New recordings data file created: {csv_file}')

def read_csv(csv_file):
    '''
    Read a subjects / sessions / recordings data file CSV and return the data as a DataFrame.

    Parameters:
    csv_file (str): the name of the CSV file to be checked / created.

    Returns:
    existing_df (DataFrame): the data in the CSV file 
    '''

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        print(f'Existing data file found: {csv_file}')
        return existing_df
    else:
        print(f'No existing data file found: {csv_file}')
        return None

def date_from_filename(filename):
    """
    Extract date from an EDF filename where the date is in the second-to-last segment
    when split by underscores. Format is expected to be 'YYYYMMDD'.
    Example: "CTRLX_tunnelX_20250301_125143.edf" -> "2025-03-01"

    Parameters:
    filename (str): the filename to extract the date from (.edf)

    Returns:
    date (str): the extracted date in 'YYYY-MM-DD' format
    """

    try:
        # Split by underscores and extract the second-to-last segment
        parts = filename.split('_')
        if len(parts) < 2:
            return None # No date found
        
        date_string = parts[-2]

        # Check if the date is in the expected format
        if len(date_string) == 8 and date_string.isdigit():
            # Format as YYYY-MM-DD
            year = date_string[0:4]
            month = date_string[4:6]
            day = date_string[6:8]
            return f'{year}-{month}-{day}'
        return None
    except Exception:
        return None

def subject_name_from_filepath(edf_path):
    '''
    Extract the subject name from an EDF file path (first underscore-delimited segment).

    Parameters:
    edf_path (str): the EDF file path.

    Returns:
    subject_name (str): the subject name extracted from the path.
    '''

    basename = os.path.basename(edf_path)
    filename, _ = os.path.splitext(basename)
    parts = filename.split('_')

    if len(parts) > 0:
        return parts[0]
    else:
        return None

def session_name_from_dir(session_dir):
    '''
    Extract the session name from a directory path (last segment).

    Parameters:
    session_dir (str): the session directory path.

    Returns:
    session_name (str): the session name extracted from the path.
    '''
    session_dir = session_dir.rstrip(os.sep)
    return os.path.basename(session_dir)

def extract_edf_from_session(session_dir):
    '''
    Extract paths of all EDF files from a session directory.
    Also works when the specified directory directly contains the EDF file (such as for a single recording).

    Parameters:
    session_dir (str): the session directory path.

    Returns:
    edf_paths (list): list of EDF filepaths.
    '''

    edf_paths = []
    for root, dirs, files in os.walk(session_dir):
        for f in files:
            if f.lower().endswith('.edf'):
                edf_paths.append(os.path.join(root, f))
    return edf_paths

def extract_pkl_from_directory(directory, name):
    '''
    Extract paths of all PKL (pickle) files from a specified directory.

    Parameters:
    directory (str): directory path to extract pickle files from.
    name (str): pickle name type: "ica_selection" / "bad_ch_selection"

    Returns:
    pkl_paths (list): list of PKL filepaths.
    '''

    pkl_paths = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            base, ext = os.path.splitext(f)
            if ext == '.pkl' and base == name:
                pkl_paths.append(os.path.join(root, f))
    return pkl_paths

def extract_trigger_npz(directory, save_csv=False):
    '''
    Populate dictionary: key: recording path; value: dict with trigger arrays (times, codes, eccentricity) / None if NPZ file not found.
    NPZ file naming scheme: "recording name" + "-timestamps.npz"
    Reads NPZ files generated by ChkEEG. Expected arrays: 'times', 'codes', 'eccentricity'.

    Parameter:
    directory (str): Directory path to extract NPZ trigger data from.
    save_csv (bool): If True, save the trigger data to a CSV file in the same directory as each NPZ file.

    Returns:
    chkeeg_triggers (dict): Dictionary with EDF paths as keys. Values are dicts with keys
        'times' (np.ndarray), 'codes' (np.ndarray), and 'eccentricity' (np.ndarray),
        or None if the NPZ file was not found.
    '''

    chkeeg_triggers = {}
    for root, dirs, files in os.walk(directory):
        edf_files = [f for f in files if f.lower().endswith('.edf')]
        for edf_file in edf_files:
            edf_path = os.path.join(root, edf_file)
            npz_name = f'{os.path.splitext(edf_file)[0]}-timestamps.npz'
            npz_path = os.path.join(root, npz_name)
            if os.path.exists(npz_path):
                try:
                    npz_raw = np.load(npz_path, allow_pickle=True)
                    npz_content = {
                        'times': npz_raw['times'] if 'times' in npz_raw else None,
                        'codes': npz_raw['codes'] if 'codes' in npz_raw else None,
                        'eccentricity': npz_raw['eccentricity'] if 'eccentricity' in npz_raw else None,
                    }

                    # Warn if eccentricity column is missing
                    if npz_content['eccentricity'] is None:
                        print(f'WARNING: No eccentricity array in {npz_path}')
                    
                    # Save to CSV if requested
                    if save_csv:
                        csv_path = os.path.join(root, f'{os.path.splitext(edf_file)[0]}-timestamps.csv')
                        try:
                            # Convert npz content to DataFrame
                            df_data = {}
                            for key, value in npz_content.items():
                                if value is None:
                                    continue
                                # Handle different array shapes
                                if isinstance(value, np.ndarray):
                                    df_data[key] = value.flatten()
                                else:
                                    df_data[key] = [value]
                            
                            # Create DataFrame with equal length columns
                            max_len = max(len(v) if hasattr(v, '__len__') else 1 for v in df_data.values())
                            for key in df_data:
                                if len(df_data[key]) < max_len:
                                    df_data[key] = np.pad(df_data[key], (0, max_len - len(df_data[key])), constant_values=np.nan)
                            

                            df = pd.DataFrame(df_data)
                            df.to_csv(csv_path, index=False)
                            print(f'Saved trigger data to CSV: {csv_path}')
                        except Exception as e:
                            print(f"WARNING: Failed to save CSV for {npz_path}: {e}")
                    
                except Exception as e:
                    print(f"WARNING: Failed to load NPZ file {npz_path}: {e}")
                    npz_content = None
                chkeeg_triggers[edf_path] = npz_content
            else:
                chkeeg_triggers[edf_path] = None
    return chkeeg_triggers

def eti_paths(edf_paths):
    '''
    Extract ETI file paths (trial info files) associated with EDF paths (In the same folder).
    
    Parameters:
    edf_paths (list): List of EDF file paths.
    
    Returns:
    eti_path_dict (dict): Dictionary of EDF file paths as keys and associated trial info ETI files as values.
    '''
    eti_path_dict = {}
    for edf in edf_paths:
        edf_dir = os.path.dirname(edf)
        eti_file = os.path.join(edf_dir, 'trial-info.eti')
        if os.path.exists(eti_file):
            eti_path_dict[edf] = eti_file
        
    return eti_path_dict

def trial_info_from_eti(eti_path,
                        onset_trigger_mapping=None,
                        fields_to_keep=None):
    """
    Extract trial information from ETI file (of selected columns) and return in pandas DataFrame.

    Parameters:
    eti_path (str): Path to ETI file.
    onset_trigger_mapping (dict): Dictionary containing EDF trigger format mapping to ETI format.
    fields_to_keep (list): List of fields (columns) to keep in ETI data file.

    Returns:
    trial_info (pd.DataFrame): Pandas DataFrame with trial information (Trial, TSide, TFovDist).
    """

    if fields_to_keep is None:
        fields_to_keep = ['Trial', 'TSide', 'TFovDist']
    if onset_trigger_mapping is None:
        onset_trigger_mapping = {'L': 'DIN21',
                                 'C': 'DIN31',
                                 'R': 'DIN41',
                                 'G': 'DIN51'}
    try:
        # check if file exists
        if not os.path.exists(eti_path):
            print(f'WARNING: ETI file not found: {eti_path}')
            return pd.DataFrame()
        
        # skip the first 2 lines (contains metadata) and read the ETI
        eti = pd.read_csv(eti_path, sep=',', skiprows=2)
        
        # check if DataFrame is empty
        if eti.empty:
            print(f"WARNING: ETI file is empty: {eti_path}")
            return pd.DataFrame()
        
        # verify required fields exist
        if fields_to_keep:
            missing_fields = [field for field in fields_to_keep if field not in eti.columns]
            if missing_fields:
                print(f"WARNING: Missing fields {missing_fields} in {eti_path}")
                return pd.DataFrame()
            trial_info = eti[fields_to_keep].copy()
        else:
            trial_info = eti.copy()
        
        # add mapping for mismatch onset triggers (ETI -> EDF)
        if 'TSide' in trial_info.columns:
            trial_info['onset_trigger_id'] = trial_info['TSide'].map(onset_trigger_mapping)
        
        return trial_info
        
    except Exception as e:
        print(f"ERROR: Failed to read ETI file {eti_path}: {e}")
        return pd.DataFrame()


class PreprocessingCache:
    """
    Simple cache manager for preprocessed MNE objects.

    Uses MNE's native .fif.gz format (optimized for EEG data).
    Metadata stored as JSON for human readability.
    """

    def __init__(self, cache_dir: Path, enabled: bool = True):
        """
        Parameters:
            cache_dir: Directory for cached files
            enabled: Enable/disable caching
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching enabled: {self.cache_dir}")
        else:
            logger.info("Caching disabled")

    @staticmethod
    def compute_params_hash(params: Dict) -> str:
        """
        Compute hash from preprocessing parameters.

        Returns 8-character hash for cache versioning.
        """
        # Sort for consistent hashing
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]

    def get_cache_paths(self,
                        identifier: str,
                        params: Dict) -> Tuple[Path, Path]:
        """
        Get cache file paths for raw object and metadata.

        Parameters:
            identifier: Unique identifier (e.g., filename stem)
            params: Preprocessing parameters dict

        Returns:
            (raw_fif_path, metadata_json_path)
        """
        params_hash = self.compute_params_hash(params)
        base_name = f"{identifier}_v{params_hash}"

        raw_path = self.cache_dir / f"{base_name}_raw.fif.gz"
        meta_path = self.cache_dir / f"{base_name}_meta.json"

        return raw_path, meta_path

    def save(self,
             raw: mne.io.Raw,
             metadata: Dict,
             identifier: str,
             params: Dict) -> None:
        """
        Save preprocessed raw object and metadata to cache.

        Parameters:
            raw: Preprocessed MNE Raw object
            metadata: Dictionary with processing info
            identifier: Unique identifier for this recording
            params: Preprocessing parameters used
        """
        if not self.enabled:
            return

        raw_path, meta_path = self.get_cache_paths(identifier, params)

        try:
            # Save raw object using MNE's compressed format
            logger.info(f"Caching: {raw_path.name}")
            raw.save(raw_path, overwrite=True, verbose=False)

            # Save metadata as JSON
            metadata['cache_version'] = self.compute_params_hash(params)
            metadata['preprocessing_params'] = params

            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)

            file_size_mb = raw_path.stat().st_size / 1024 / 1024
            logger.info(f"Cached {file_size_mb:.1f} MB to {raw_path.name}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def load(self,
             identifier: str,
             params: Dict) -> Optional[Tuple[mne.io.Raw, Dict]]:
        """
        Load preprocessed raw object from cache if available.

        Parameters:
            identifier: Unique identifier for this recording
            params: Current preprocessing parameters

        Returns:
            (raw, metadata) if cache hit, None if cache miss
        """
        if not self.enabled:
            return None

        raw_path, meta_path = self.get_cache_paths(identifier, params)

        # Check if both files exist
        if not (raw_path.exists() and meta_path.exists()):
            logger.debug(f"Cache miss: {identifier}")
            return None

        try:
            # Load metadata first to check version
            with open(meta_path, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted cache metadata: {meta_path.name}")
                    logger.error(f"JSON error: {e}")
                    logger.info(f"Deleting corrupted cache files...")

                    # Delete both corrupted files
                    if raw_path.exists():
                        raw_path.unlink()
                    if meta_path.exists():
                        meta_path.unlink()

                    logger.info(f"Corrupted cache deleted. Will re-preprocess.")
                    return None

            # Verify cache version matches current params
            expected_hash = self.compute_params_hash(params)
            cached_hash = metadata.get('cache_version', '')

            if cached_hash != expected_hash:
                cached_params = metadata.get('preprocessing_params', {})
                diff = self._compare_params(cached_params, params)

                msg_parts = [f"Cache miss: {identifier} (parameters changed)"]

                if diff['changed']:
                    changes_str = ", ".join([f"{item['param']}: {item['old']}→{item['new']}"
                                             for item in diff['changed']])
                    msg_parts.append(f"  Changed: {changes_str}")

                if diff['added']:
                    msg_parts.append(f"  Added: {', '.join(diff['added'])}")

                if diff['removed']:
                    msg_parts.append(f"  Removed: {', '.join(diff['removed'])}")

                for msg in msg_parts:
                    logger.warning(msg)

                return None

            # Load raw object
            logger.info(f"Loading from cache: {raw_path.name}")
            raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

            logger.info(f"Cache hit: {raw.n_times} samples, "
                        f"{len(raw.ch_names)} channels")

            return raw, metadata

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted JSON in cache: {meta_path.name}")
            logger.error(f"Error: {e}")

            # Delete corrupted cache
            if raw_path.exists():
                raw_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

            logger.info(f"Deleted corrupted cache. Re-preprocessing.")
            return None

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-preprocessing.")
            return None

    def clear(self, identifier: Optional[str] = None):
        """
        Clear cached files.

        Parameters:
            identifier: Clear only files for this identifier (None = clear all)
        """
        if not self.enabled or not self.cache_dir.exists():
            return

        if identifier:
            # Clear specific identifier
            pattern = f"{identifier}_v*"
            files = list(self.cache_dir.glob(pattern + "*"))
            for f in files:
                f.unlink()
            logger.info(f"Cleared cache for {identifier}: {len(files)} files")
        else:
            # Clear all
            files = list(self.cache_dir.glob("*"))
            for f in files:
                if f.is_file():
                    f.unlink()
            logger.info(f"Cleared all cache: {len(files)} files")

    def _compare_params(self, cached_params: Dict, current_params: Dict) -> Dict:
        """
        Compare two parameter dictionaries and identify differences.

        Returns:
            dict with 'changed', 'added', 'removed' keys
        """
        result = {
            'changed': [],
            'added': [],
            'removed': []
        }

        all_keys = set(cached_params.keys()) | set(current_params.keys())

        for key in sorted(all_keys):
            if key not in cached_params:
                result['added'].append(key)
            elif key not in current_params:
                result['removed'].append(key)
            elif cached_params[key] != current_params[key]:
                result['changed'].append({
                    'param': key,
                    'old': cached_params[key],
                    'new': current_params[key]
                })

        return result

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class AperiodicCache:
    """
    Cache manager for aperiodic analysis results.

    Stores time-resolved aperiodic parameters (exponents, offsets, alpha power)
    as compressed NumPy files with JSON metadata for human readability.
    """

    def __init__(self, cache_dir: Path, enabled: bool = True):
        """
        Parameters:
            cache_dir: Directory for cached analysis results
            enabled: Enable/disable caching
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Aperiodic caching enabled: {self.cache_dir}")
        else:
            logger.info("Aperiodic caching disabled")

    @staticmethod
    def compute_params_hash(params: Dict) -> str:
        """
        Compute hash from analysis parameters.
        Returns 8-character hash for cache versioning.
        """
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]

    def get_cache_paths(self,
                        identifier: str,
                        channel_idx: int,
                        params: Dict) -> Tuple[Path, Path]:
        """
        Get cache file paths for analysis results and metadata.

        Parameters:
            identifier: Unique identifier (e.g., subject_condition_run)
            channel_idx: Channel index
            params: Analysis parameters dict

        Returns:
            (npz_path, metadata_json_path)
        """
        params_hash = self.compute_params_hash(params)
        base_name = f"{identifier}_ch{channel_idx:02d}_v{params_hash}"
        npz_path = self.cache_dir / f"{base_name}_results.npz"
        meta_path = self.cache_dir / f"{base_name}_meta.json"
        return npz_path, meta_path

    def save(self,
             results: Dict,
             identifier: str,
             channel_idx: int,
             params: Dict) -> None:
        """
        Save aperiodic analysis results to cache.

        Parameters:
            results: Dictionary with analysis results (from extract_aperiodic_pipeline)
            identifier: Unique identifier for this recording
            channel_idx: Channel index
            params: Analysis parameters used
        """
        if not self.enabled:
            return

        npz_path, meta_path = self.get_cache_paths(identifier, channel_idx, params)

        try:
            logger.debug(f"Caching: {npz_path.name}")
            np.savez_compressed(
                npz_path,
                times_min=results['times_min'],
                exponents=results['exponents'],
                offsets=results['offsets'],
                alpha_power=results['alpha_power'],
                r_squared=results['r_squared']
            )

            metadata = {
                'cache_version': self.compute_params_hash(params),
                'analysis_params': params,
                'roi_name': results.get('roi_name', 'unknown'),
                'channel_indices': results.get('channel_indices', [channel_idx]),
                'n_channels': results.get('n_channels', 1),
                'mean_r2': float(results.get('mean_r2', 0)),
                'fit_range': results.get('fit_range', (2, 45)),
                'window_duration': results.get('window_duration', 20.0),
                'step_duration': results.get('step_duration', 10.0),
                'method': results.get('method', 'unknown')
            }

            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)

            file_size_kb = npz_path.stat().st_size / 1024
            logger.debug(f"Cached {file_size_kb:.1f} KB to {npz_path.name}")

        except Exception as e:
            logger.error(f"Failed to save aperiodic cache: {e}")

    def load(self,
             identifier: str,
             channel_idx: int,
             params: Dict) -> Optional[Dict]:
        """
        Load aperiodic analysis results from cache if available.

        Parameters:
            identifier: Unique identifier for this recording
            channel_idx: Channel index
            params: Current analysis parameters

        Returns:
            Dictionary with results if cache hit, None if cache miss
        """
        if not self.enabled:
            return None

        npz_path, meta_path = self.get_cache_paths(identifier, channel_idx, params)

        if not (npz_path.exists() and meta_path.exists()):
            logger.debug(f"Cache miss: {identifier}_ch{channel_idx:02d}")
            return None

        try:
            with open(meta_path, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Corrupted cache metadata: {meta_path.name}")
                    if npz_path.exists():
                        npz_path.unlink()
                    if meta_path.exists():
                        meta_path.unlink()
                    logger.info(f"Corrupted cache deleted. Will re-analyze.")
                    return None

            expected_hash = self.compute_params_hash(params)
            cached_hash = metadata.get('cache_version', '')

            if cached_hash != expected_hash:
                cached_params = metadata.get('analysis_params', {})
                diff = self._compare_params(cached_params, params)

                if diff['changed'] or diff['added'] or diff['removed']:
                    logger.debug(f"Cache miss: {identifier}_ch{channel_idx:02d} (parameters changed)")
                    return None

            logger.debug(f"Loading from cache: {npz_path.name}")
            data = np.load(npz_path)

            results = {
                'times_min': data['times_min'],
                'exponents': data['exponents'],
                'offsets': data['offsets'],
                'alpha_power': data['alpha_power'],
                'r_squared': data['r_squared'],
                'roi_name': metadata['roi_name'],
                'channel_indices': metadata['channel_indices'],
                'n_channels': metadata['n_channels'],
                'mean_r2': metadata['mean_r2'],
                'fit_range': tuple(metadata['fit_range']),
                'window_duration': metadata['window_duration'],
                'step_duration': metadata['step_duration'],
                'method': metadata['method']
            }

            logger.debug(f"Cache hit: {len(results['times_min'])} windows, "
                         f"mean_r2={results['mean_r2']:.3f}")

            return results

        except Exception as e:
            logger.warning(f"Failed to load aperiodic cache: {e}. Re-analyzing.")
            return None

    def clear(self, identifier: Optional[str] = None, channel_idx: Optional[int] = None):
        """
        Clear cached files.

        Parameters:
            identifier: Clear only files for this identifier (None = clear all)
            channel_idx: Clear only for this channel (None = all channels for identifier)
        """
        if not self.enabled or not self.cache_dir.exists():
            return

        if identifier and channel_idx is not None:
            pattern = f"{identifier}_ch{channel_idx:02d}_v*"
            files = list(self.cache_dir.glob(pattern + "*"))
            for f in files:
                f.unlink()
            logger.info(f"Cleared cache for {identifier} ch{channel_idx:02d}: {len(files)} files")
        elif identifier:
            pattern = f"{identifier}_ch*_v*"
            files = list(self.cache_dir.glob(pattern + "*"))
            for f in files:
                f.unlink()
            logger.info(f"Cleared cache for {identifier}: {len(files)} files")
        else:
            files = list(self.cache_dir.glob("*"))
            for f in files:
                if f.is_file():
                    f.unlink()
            logger.info(f"Cleared all aperiodic cache: {len(files)} files")

    def _compare_params(self, cached_params: Dict, current_params: Dict) -> Dict:
        """Compare parameter dictionaries and identify differences."""
        result = {'changed': [], 'added': [], 'removed': []}

        all_keys = set(cached_params.keys()) | set(current_params.keys())

        for key in sorted(all_keys):
            if key not in cached_params:
                result['added'].append(key)
            elif key not in current_params:
                result['removed'].append(key)
            elif cached_params[key] != current_params[key]:
                result['changed'].append({
                    'param': key,
                    'old': cached_params[key],
                    'new': current_params[key]
                })

        return result

class TFRFeaturesCache:
    """
    Cache manager for TFR feature extraction results.

    Stores per-subject feature DataFrames as pickle files with JSON metadata.
    Follows the same pattern as PreprocessingCache and AperiodicCache.
    """

    def __init__(self, cache_dir: Path, enabled: bool = True):
        """
        Parameters:
            cache_dir: Directory for cached feature files
            enabled: Enable/disable caching
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"TFR features caching enabled: {self.cache_dir}")
        else:
            logger.info("TFR features caching disabled")

    @staticmethod
    def compute_params_hash(params: Dict) -> str:
        """
        Compute hash from analysis parameters.
        Returns 8-character hash for cache versioning.
        """
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]

    def get_cache_paths(self,
                        subject_name: str,
                        params: Dict) -> Tuple[Path, Path]:
        """
        Get cache file paths for features DataFrame and metadata.

        Parameters:
            subject_name: Subject identifier
            params: Analysis parameters dict

        Returns:
            (pkl_path, metadata_json_path)
        """
        params_hash = self.compute_params_hash(params)
        base_name = f"{subject_name}_tfr_features_v{params_hash}"

        pkl_path = self.cache_dir / f"{base_name}.pkl"
        meta_path = self.cache_dir / f"{base_name}_meta.json"

        return pkl_path, meta_path

    def save(self,
             features_df: pd.DataFrame,
             subject_name: str,
             params: Dict) -> None:
        """
        Save TFR features DataFrame to cache.

        Parameters:
            features_df: DataFrame with extracted TFR features
            subject_name: Subject identifier
            params: Analysis parameters used
        """
        if not self.enabled or features_df is None or len(features_df) == 0:
            return

        pkl_path, meta_path = self.get_cache_paths(subject_name, params)

        try:
            logger.info(f"Caching TFR features: {pkl_path.name}")

            # Save DataFrame as pickle
            features_df.to_pickle(pkl_path)

            # Create metadata
            metadata = {
                'cache_version': self.compute_params_hash(params),
                'analysis_params': params,
                'subject': subject_name,
                'n_rows': len(features_df),
                'n_channels': features_df['channel'].nunique() if 'channel' in features_df.columns else 0,
                'conditions': list(features_df['condition'].unique()) if 'condition' in features_df.columns else [],
                'mismatch_types': list(features_df['mismatch_type'].unique()) if 'mismatch_type' in features_df.columns else [],
                'columns': list(features_df.columns),
                'eye_dominance': features_df['eye_dominance'].iloc[0] if 'eye_dominance' in features_df.columns and len(features_df) > 0 else 'unknown'
            }

            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)

            file_size_kb = pkl_path.stat().st_size / 1024
            logger.info(f"Cached {file_size_kb:.1f} KB ({len(features_df)} rows) to {pkl_path.name}")

        except Exception as e:
            logger.error(f"Failed to save TFR features cache: {e}")

    def load(self,
             subject_name: str,
             params: Dict) -> Optional[pd.DataFrame]:
        """
        Load TFR features DataFrame from cache if available.

        Parameters:
            subject_name: Subject identifier
            params: Current analysis parameters

        Returns:
            DataFrame with features if cache hit, None if cache miss
        """
        if not self.enabled:
            return None

        pkl_path, meta_path = self.get_cache_paths(subject_name, params)

        # Check if both files exist
        if not (pkl_path.exists() and meta_path.exists()):
            logger.debug(f"Cache miss: {subject_name}")
            return None

        try:
            # Load metadata first to check version
            with open(meta_path, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Corrupted cache metadata: {meta_path.name}")
                    # Delete corrupted files
                    if pkl_path.exists():
                        pkl_path.unlink()
                    if meta_path.exists():
                        meta_path.unlink()
                    logger.info(f"Corrupted cache deleted. Will re-compute.")
                    return None

            # Verify cache version matches current params
            expected_hash = self.compute_params_hash(params)
            cached_hash = metadata.get('cache_version', '')

            if cached_hash != expected_hash:
                cached_params = metadata.get('analysis_params', {})
                diff = self._compare_params(cached_params, params)

                if diff['changed'] or diff['added'] or diff['removed']:
                    msg_parts = [f"Cache miss: {subject_name} (parameters changed)"]
                    if diff['changed']:
                        changes_str = ", ".join([f"{item['param']}: {item['old']}→{item['new']}"
                                                 for item in diff['changed'][:3]])  # Show first 3
                        msg_parts.append(f"  Changed: {changes_str}")

                    for msg in msg_parts:
                        logger.warning(msg)

                    return None

            # Load DataFrame
            logger.info(f"Loading from cache: {pkl_path.name}")
            features_df = pd.read_pickle(pkl_path)

            logger.info(f"Cache hit: {len(features_df)} rows, "
                        f"{features_df['channel'].nunique() if 'channel' in features_df.columns else 0} channels")

            return features_df

        except Exception as e:
            logger.warning(f"Failed to load TFR features cache: {e}. Re-computing.")
            return None

    def clear(self, subject_name: Optional[str] = None):
        """
        Clear cached files.

        Parameters:
            subject_name: Clear only files for this subject (None = clear all)
        """
        if not self.enabled or not self.cache_dir.exists():
            return

        if subject_name:
            # Clear specific subject
            pattern = f"{subject_name}_tfr_features_v*"
            files = list(self.cache_dir.glob(pattern + "*"))
            for f in files:
                f.unlink()
            logger.info(f"Cleared TFR cache for {subject_name}: {len(files)} files")
        else:
            # Clear all
            files = list(self.cache_dir.glob("*tfr_features_v*"))
            for f in files:
                if f.is_file():
                    f.unlink()
            logger.info(f"Cleared all TFR features cache: {len(files)} files")

    def _compare_params(self, cached_params: Dict, current_params: Dict) -> Dict:
        """Compare parameter dictionaries and identify differences."""
        result = {'changed': [], 'added': [], 'removed': []}

        all_keys = set(cached_params.keys()) | set(current_params.keys())

        for key in sorted(all_keys):
            if key not in cached_params:
                result['added'].append(key)
            elif key not in current_params:
                result['removed'].append(key)
            elif cached_params[key] != current_params[key]:
                result['changed'].append({
                    'param': key,
                    'old': cached_params[key],
                    'new': current_params[key]
                })

        return result

if __name__ == '__main__':
    data = extract_trigger_npz(r"C:\Users\fekbe\Documents\TTK\Current work\stmd_data\STMD1\Baseline\08_56")
    print(data)