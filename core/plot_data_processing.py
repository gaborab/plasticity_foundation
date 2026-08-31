from typing import List, Dict, Any, Tuple

import numpy as np

from core import settings
from core.io_models import Trial, Metrics
from scipy.stats import sem
from numpy import mean, std
from collections import defaultdict

def calc_subjectwise_channel_metric_stats(trial_list: List[Trial], bands: List[str], metric: str,  ch_names: List[str]=None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Calculates the mean NMAE and SEM for each subject across all trials in trial_list.
    Returns a dictionary with subject names as keys and channel-wise mean NMAE including global values as values.
    """
    subjectwise_nmae = {}
    subjectwise_sem = {}

    trials_by_subject = _group_by(trial_list, 'subject')

    for subject, trials in trials_by_subject.items():
        subjectwise_nmae[subject], subjectwise_sem[subject], _ = get_channel_bandwise_metric_stats(trials, metric_name=metric, ch_names=ch_names, bands=bands)
        global_metric_mean, global_metric_sem, _= get_channel_metric_stats(trials, metric_name=metric, ch_names=ch_names)
        for ch in subjectwise_nmae[subject]:
            subjectwise_nmae[subject][ch]['global'] = global_metric_mean[ch]
            subjectwise_sem[subject][ch]['global'] = global_metric_sem[ch]


    return subjectwise_nmae, subjectwise_sem

def calc_subjectwise_metric_stats(trial_list: List[Trial], bands: List[str], metric: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculates the mean NMAE and SEM for each subject across all trials in trial_list.
    Returns a dictionary with subject names as keys and bandwise mean NMAE including global values as values.
    """
    subjectwise_nmae = {}
    subjectwise_sem = {}

    trials_by_subject = _group_by(trial_list, 'subject')

    for subject, trials in trials_by_subject.items():
        subjectwise_nmae[subject], subjectwise_sem[subject], _ = get_bandwise_metric_stats(trials, metric_name=metric, bands=bands)
        subjectwise_nmae[subject]['global'], subjectwise_sem[subject]['global'], _ = get_metric_stats(trials, metric_name=metric)


    return subjectwise_nmae, subjectwise_sem

def calc_bandwise_ampl_mean(trial_list: List[Trial], bands: Dict[str, Any], freqs) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculates the mean power for each frequency band across all trials in trial_list.
    Returns a dictionary with band names as keys and mean power values as values.
    """
    bandwise_power = {b: [] for b in bands.keys()}
    bandwise_power["global"] = []
    for trial in trial_list:
        bandwise_power["global"].append(mean(std(trial.target_data, axis=2, ddof=0)))
        for band_name, band in bands.items():
            low, high = band
            band_indices = np.where((freqs >= low) & (freqs <= high))[0]
            b_target_band_power = mean(std(trial.target_data[:, band_indices], axis=2, ddof=0))
            bandwise_power[band_name].append(b_target_band_power)

    return {b: mean(vals) for b, vals in bandwise_power.items()}, {b: sem(vals, ddof=0) for b, vals in bandwise_power.items()}


def calc_channel_glass_delta(ch_nmaes_b: Dict[str, float], ch_nmaes_p: Dict[str, float], ch_stds: Dict[str, float]) -> List[float]:
    """
    Calculates glass's delta effect size for each channel given 2 dicts of channel-wise NMAEs and one dict of STDs.
    Returns a List of glass's delta values for each channel in the order of the keys in the input dictionaries.
    """
    ret = []
    if not (ch_nmaes_b.keys() == ch_nmaes_p.keys() == ch_stds.keys()):
        raise ValueError("Channel keys do not match between the input dictionaries.")

    for ch in ch_nmaes_b.keys():
        if ch not in ch_nmaes_b or ch not in ch_nmaes_p or ch not in ch_stds:
            raise ValueError(f"Channel key {ch} is missing in one of the input dictionaries.")
        ret.append((ch_nmaes_p[ch] - ch_nmaes_b[ch]) / ch_stds[ch])

    return ret

def calc_channel_bandwise_glass_delta(ch_bw_nmaes_b: Dict[str, Dict[str, float]], ch_bw_nmaes_p: Dict[str, Dict[str, float]], ch_bw_stds: Dict[str, Dict[str, float]]) -> Dict[str, List[float]]:
    """
    Calculates glass's delta effect size for each channel and band given 2 dicts of channel-wise bandwise NMAEs and one dict of STDs.
    Returns a dictionary with band names containing lists of channel glass's delta values for each channel in the order of the keys in the input dictionaries.
    """
    ret = {}
    if not (ch_bw_nmaes_b.keys() == ch_bw_nmaes_p.keys() == ch_bw_stds.keys()):
        raise ValueError("Channel keys do not match between the input dictionaries.")

    for ch in ch_bw_nmaes_b.keys():
        if ch not in ch_bw_nmaes_b or ch not in ch_bw_nmaes_p or ch not in ch_bw_stds:
            raise ValueError(f"Channel key {ch} is missing in one of the input dictionaries.")

        for band in ch_bw_nmaes_b[ch].keys():
            ret.setdefault(band, [])
            ret[band].append((ch_bw_nmaes_p[ch][band] - ch_bw_nmaes_b[ch][band]) / ch_bw_stds[ch][band])

    return ret



def rename_chans_to_mne(ch_names: List[str]):
    for i, name in enumerate(ch_names):
        mod_name = name.split(' ')[-1].split('-')[0]
        if mod_name == "T1":
            ch_names[i] = "FT9"
            continue
        if mod_name == "T2":
            ch_names[i] = "FT10"
            continue
        if mod_name[1].isdigit():
            ch_names[i] = mod_name
            continue
        mod_name = mod_name[0] + mod_name[1].lower() + mod_name[2:]
        ch_names[i] = mod_name

def order_to_channel_viewer_format(dataset_data: List[List[Trial]], dataset_metrics: List[List[Dict[str, Any]]], sample_idx, channel_idx) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ds_trials = [ds[sample_idx] for ds in dataset_data]  # select sample across all datasets
    trial_data_dict = [
        {
            "output": selected_trial.recon_data[channel_idx],
            "target": selected_trial.target_data[channel_idx],
        }
        for selected_trial in ds_trials
    ]  # select channel across all datasets
    ds_metrics = [ds_metrics[sample_idx][channel_idx] for ds_metrics in dataset_metrics]
    return trial_data_dict, ds_metrics

def get_channel_viewer_metrics(trial_list: List[Trial], metric_names: List[str], ch_prefix: str= 'ch_') -> List[List[Dict[str, Any]]]:
    ret = []
    for trial in trial_list:
        trial_data = []
        for i in range(len(trial.channel_metrics)):
            chan_data = {}
            # chan_data["target"] = trial.target_data[i]
            # chan_data["output"] = trial.recon_data[i]

            chan_data["metrics"] = {}
            for metric_name in metric_names:
                ch_metric_value = getattr(trial.channel_metrics[ch_prefix+str(i)], metric_name, None)
                if ch_metric_value is None:
                    print(f"WARNING: Trial {trial.subject}-{trial.condition}-{trial.idx} is missing channel-wise metric {metric_name} for channel {ch_prefix+str(i)}. Skipping...")
                    continue
                chan_data["metrics"][metric_name] = ch_metric_value

            chan_data["metadata"] = {
                "subject": trial.subject,
                "condition": trial.condition,
                "id": trial.idx,
            }
            trial_data.append(chan_data)
        ret.append(trial_data)
    return ret



def get_channel_metric_stats(trial_list: List[Trial], metric_name: str, ch_names: List[str]=None) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    channel_values = _extract_channel_metric(trial_list, metric_name, ch_names)
    ch_means = {}
    ch_sems = {}
    ch_stds = {}
    for ch_name, vals in channel_values.items():
        if len(vals) == 0:
            print(f"WARNING: No values found for channel {ch_name} for metric {metric_name}. This channel will be skipped in the averaging.")
            del channel_values[ch_name]
        ch_means[ch_name] = mean(vals)
        ch_sems[ch_name] = sem(vals, ddof=0)
        ch_stds[ch_name] = std(vals, ddof=0)
    return ch_means, ch_sems, ch_stds


def get_channel_bandwise_metric_stats(trial_list: List[Trial], metric_name: str, bands: List[str], ch_names: List[str]=None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    if 'bandwise' not in metric_name:
        metric_name += '_bandwise'

    channel_values = _extract_channel_metric(trial_list, metric_name, ch_names=ch_names, bands=bands, bandwise=True)
    ch_means = {}
    ch_sems = {}
    ch_stds = {}
    for ch_name, vals in channel_values.items():
        if len(vals) == 0:
            print(f"WARNING: No values found for channel {ch_name} for metric {metric_name}")
            continue
        ch_means[ch_name] = {b: mean(band_vals) for b, band_vals in vals.items()}
        ch_sems[ch_name] = {b: sem(band_vals, ddof=0) for b, band_vals in vals.items()}
        ch_stds[ch_name] = {b: std(band_vals, ddof=0) for b, band_vals in vals.items()}
    return ch_means, ch_sems, ch_stds


def get_metric_stats(trial_list: List[Trial], metric_name: str) -> Tuple[float, float, float]:
    values = _extract_metric(trial_list, metric_name)
    return mean(values), sem(values, ddof=0), std(values, ddof=0)


def get_bandwise_metric_stats(trial_list: List[Trial], metric_name: str, bands: List[str]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    if 'bandwise' not in metric_name:
        metric_name += '_bandwise'

    bandwise_values = _extract_metric(trial_list, metric_name, bands=bands, bandwise=True)

    means = {}
    sems = {}
    stds = {}
    for b, vals in bandwise_values.items():
        means[b] = mean(vals)
        sems[b] = sem(vals, ddof=0)
        stds[b] = std(vals, ddof=0)
    return means, sems, stds

def _extract_channel_metric(trial_list: List[Trial], metric_name: str, ch_names: List[str]=None, bands: List[str]=None, bandwise: bool=False) -> Dict[str, Any]:
    """
    Internal helper function to extract channel-wise metric values from a list of Trial objects.
    If ch_names is provided, only metrics for those channels will be extracted.
    If bands is provided, only band-wise metrics for those bands will be extracted.
    Returns a dictionary with channel names as keys and lists of metric values as values.
    """
    ret_dict = {}
    for trial in trial_list:
        for ch_name, ch_metrics in trial.channel_metrics.items():

            if ch_names is not None and ch_name not in ch_names:
                continue

            target_value = getattr(ch_metrics, metric_name, {})
            if not target_value:
                print(f"WARNING: Trial {trial.subject}-{trial.condition}-{trial.idx} is missing channel-wise metrics for requested metric {metric_name} on channel {ch_name}. Skipping...")
                continue

            if isinstance(target_value, dict) and bandwise:
                if ch_name not in ret_dict:
                    ret_dict[ch_name] = {b: [] for b in bands}
                _group_bandwise(target_value, trial, metric_name, bands, ret_dict[ch_name], ch_name=ch_name)

            elif not isinstance(target_value, dict) and not bandwise:
                if ch_name not in ret_dict:
                    ret_dict[ch_name] = []
                ret_dict[ch_name].append(target_value)
            else:
                print(f"WARNING: Trial {trial.subject}-{trial.condition}-{trial.idx} has a mismatch between requested metric type and actual metric type for {metric_name} on channel {ch_name}. Skipping...")
                continue
    return ret_dict

def _extract_metric(trial_list: List[Trial], metric_name: str, bands: List[str]=None, bandwise: bool=False) -> List[Any] | Dict[str, List[Any]]:
    """
    Internal helper function to extract global metric values from a list of Trial objects.
    If bands is provided, only band-wise metrics for those bands will be extracted.
    Returns a list of metric values.
    """
    ret = {b: [] for b in bands} if bandwise else []
    for trial in trial_list:
        target_value = getattr(trial.metrics, metric_name, {})
        if not target_value:
            print(f"WARNING: Trial {trial.subject}-{trial.condition}-{trial.idx} is missing requested metric {metric_name}. Skipping...")
            continue

        if isinstance(target_value, dict) and bandwise:
            _group_bandwise(target_value, trial, metric_name, bands, ret)

        elif not isinstance(target_value, dict) and not bandwise:
            ret.append(target_value)
        else:
            print(f"WARNING: Trial {trial.subject}-{trial.condition}-{trial.idx} has a mismatch between requested metric type and actual metric type for {metric_name}. Skipping...")
            continue

    return ret

def _group_bandwise(bandwise_dict: Dict[str, float], trial: Trial, metric_name: str, bands: List[str], master_dict: Dict[str, List[float]], ch_name: str=None):
    """
    Internal helper function to extract band-wise metric values from a dictionary of band-wise metrics and group them into a master dictionary.
    Only metrics in bands will be extracted.
    Updates master_dict with the extracted values.
    """
    for b in bands:
        if b in bandwise_dict:
            master_dict[b].append(bandwise_dict[b])
        else:
            if ch_name is not None:
                print(f"WARNING: Trial {trial.subject}-{trial.condition}-{trial.idx} is missing band-wise metrics for requested metric {metric_name} on channel {ch_name} for band {b}. Skipping...")
            else:
                print(f"WARNING: Trial {trial.subject}-{trial.condition}-{trial.idx} is missing band-wise metrics for requested metric {metric_name} for band {b}. Skipping...")


def _group_by(trial_list: List[Trial], attr_name: str) -> Dict[Any, List[Trial]]:
    trials_by_attr = defaultdict(list)
    for trial in trial_list:
        trials_by_attr[getattr(trial, attr_name)].append(trial)
    return trials_by_attr