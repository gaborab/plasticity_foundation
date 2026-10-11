from typing import List, Dict, Any, Tuple

import mne
import numpy as np

from core import settings
from core.io_models import Trial, Metrics
from scipy.stats import sem
import numpy as np
from collections import defaultdict

ALL_BANDS = "all bands"

TIME_SERIES_METRICS = {
    'mae': lambda target, recon: np.abs(target - recon),
    'error': lambda target, recon: target - recon,
    'target': lambda target, recon: target,
    'recon': lambda target, recon: recon,
}

def calc_metric_time_series(trial: Trial, metric_name: str, band_indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculates the time series of metric_name (a key of TIME_SERIES_METRICS) for every band and "all bands" of the given trial.
    The metric is computed on the full (channels, frequencies, time) tensors, then averaged over the band's frequency bins.
    Expects band_indices as {band name: indices of the band's frequency bins}, e.g. settings.FREQ_BAND_INDICES.
    Returns {band: array of shape (channels, time)}, average over the channels for the time series of the whole head.
    The output can be saved with io.save_time_series_metrics_hdf and read back with trial.time_series(metric_name) in the same format.
    """
    return calc_time_series_metrics(trial, [metric_name], band_indices)[metric_name]

def calc_time_series_metrics(trial: Trial, metric_names: List[str], band_indices: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculates the time series of several metrics like calc_metric_time_series, reading the trial's target and reconstruction only once.
    Returns {metric: {band: array of shape (channels, time)}}, the input format of io.save_time_series_metrics_hdf.
    """
    target, recon = trial.target_data, trial.recon_data
    metrics_time_series = {}
    for metric_name in metric_names:
        values = TIME_SERIES_METRICS[metric_name](target, recon)
        metric_time_series = {ALL_BANDS: values.mean(axis=1)}
        for band, indices in band_indices.items():
            metric_time_series[band] = values[:, indices, :].mean(axis=1)
        metrics_time_series[metric_name] = metric_time_series
    return metrics_time_series

def calc_subject_adjusted_metric(b_trial_list: List[Trial], p_trial_list: List[Trial], metric_name: str, bands: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Calculates volatility-adjusted subject means of metric_name for the whole tensor (every band and "all bands").
    Per band, one line is fitted on the subjects' metric change (post - baseline) against their volatility change, see _adjust_subject_means.
    Only subjects with both sessions are used and returned.
    Returns:
        baseline subject means {subject: {band: value}} (unadjusted, the baseline is the reference),
        adjusted post-patching subject means {subject: {band: value}},
        fitted slopes {band: slope}.
    """
    b_metric_means, _ = calc_subjectwise_metric_stats(b_trial_list, bands, metric_name)
    p_metric_means, _ = calc_subjectwise_metric_stats(p_trial_list, bands, metric_name)
    b_vol_means, _ = calc_subjectwise_metric_stats(b_trial_list, bands, 'signal_volatility')
    p_vol_means, _ = calc_subjectwise_metric_stats(p_trial_list, bands, 'signal_volatility')

    return _adjust_subject_means(b_metric_means, p_metric_means, b_vol_means, p_vol_means)


def calc_subject_adjusted_channel_metric(b_trial_list: List[Trial], p_trial_list: List[Trial], metric_name: str, bands: List[str]) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]]:
    """
    Calculates volatility-adjusted subject means of metric_name for every channel separately (every band and "all bands").
    Every channel gets its own fit of the subjects' metric change (post - baseline) against their volatility change, see _adjust_subject_means.
    Only subjects with both sessions are used and returned.
    Returns:
        baseline subject means {channel: {subject: {band: value}}} (unadjusted, the baseline is the reference),
        adjusted post-patching subject means {channel: {subject: {band: value}}},
        fitted slopes {channel: {band: slope}}.
    """
    b_metric_means, _ = calc_subjectwise_channel_metric_stats(b_trial_list, bands, metric_name, group_by='channel')
    p_metric_means, _ = calc_subjectwise_channel_metric_stats(p_trial_list, bands, metric_name, group_by='channel')
    b_vol_means, _ = calc_subjectwise_channel_metric_stats(b_trial_list, bands, 'signal_volatility', group_by='channel')
    p_vol_means, _ = calc_subjectwise_channel_metric_stats(p_trial_list, bands, 'signal_volatility', group_by='channel')

    b_means, p_adjusted, slopes = {}, {}, {}
    for ch in b_metric_means.keys():
        b_means[ch], p_adjusted[ch], slopes[ch] = _adjust_subject_means(b_metric_means[ch], p_metric_means[ch], b_vol_means[ch], p_vol_means[ch])

    return b_means, p_adjusted, slopes

def _adjust_subject_means(b_metric_means: Dict[str, Dict[str, float]], p_metric_means: Dict[str, Dict[str, float]],
                          b_vol_means: Dict[str, Dict[str, float]], p_vol_means: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Expects subject means as {subject: {band: value}} for the metric and the volatility of both sessions.
    For every band, one line is fitted on the subjects' metric change (post - baseline) against their volatility change.
    Adjusted post-patching mean = post-patching mean - slope * volatility change, i.e. the post-patching value at the subject's baseline volatility.
    Only subjects with both sessions are used and returned.
    Returns the baseline means, the adjusted post-patching means and the slopes per band.
    """
    subjects = [s for s in b_metric_means if s in p_metric_means]
    band_keys = list(b_metric_means[subjects[0]].keys())
    delta_vols = {s: {b: p_vol_means[s][b] - b_vol_means[s][b] for b in band_keys} for s in subjects}
    delta_metrics = {s: {b: p_metric_means[s][b] - b_metric_means[s][b] for b in band_keys} for s in subjects}

    slopes = {}
    for b in band_keys:
        vol_changes = [delta_vols[s][b] for s in subjects]
        metric_changes = [delta_metrics[s][b] for s in subjects]
        slopes[b] = float(np.polyfit(vol_changes, metric_changes, 1)[0])

    # filter baseline means for subjects that are in both sessions (adjustment not applied to baseline)
    b_means = {s: {b: b_metric_means[s][b] for b in band_keys} for s in subjects}
    # calculate adjusted post-patching means for subjects that are in both sessions
    p_adjusted = {s: {b: p_metric_means[s][b] - slopes[b] * delta_vols[s][b] for b in band_keys} for s in subjects}
    return b_means, p_adjusted, slopes


def fill_corrected_maes(b_trial_list: List[Trial], p_trial_list: List[Trial], bands: List[str]) -> Tuple[Dict[str, float], Dict[str, Dict[str,float]]]:
    """
    Fills in the corrected_mae and corrected_mae_bandwise fields of the global metrics and of every channel's metrics for each trial in b_trial_list and p_trial_list.
    The whole tensor and every channel are corrected separately: volatility and MAE are centered within each subject and session,
    then one line per band (and for all bands) is fitted on the centered trials of all subjects.
    Corrected MAE is calculated as: MAE - slope * (signal_volatility - mean_baseline_volatility).
    Returns the global slopes {band: slope} and the channel slopes {channel: {band: slope}}, with "all bands" included as a band.
    """
    # merge all bands with rest of the individual bands
    keys = [ALL_BANDS, *bands]
    b_trials_by_subject = _group_by(b_trial_list, 'subject')
    p_trials_by_subject = _group_by(p_trial_list, 'subject')

    session_trial_groups = []  # one list of trials per subject and session, only subjects with both sessions
    # create groups of trials for each subject and session, only subjects with both sessions are included
    for subject in b_trials_by_subject.keys():
        if subject not in p_trials_by_subject:
            print(
                f"WARNING: Subject {subject} has baseline trials but no post-patching trials. Skipping it when fitting the slopes.")
            continue
        session_trial_groups.append(b_trials_by_subject[subject])
        session_trial_groups.append(p_trials_by_subject[subject])
    all_trials = [*b_trial_list, *p_trial_list]

    # whole tensor
    all_channel_slopes = _fill_corrected_metrics(
        session_groups=[[t.metrics for t in trials] for trials in session_trial_groups],
        baseline_metrics=[t.metrics for t in b_trial_list],
        all_metrics=[t.metrics for t in all_trials],
        band_keys=keys)

    # per-channel
    channel_slopes = {}
    for ch in b_trial_list[0].channel_metrics.keys():
        # same function call with channel metrics instead of whole tensor metrics
        channel_slopes[ch] = _fill_corrected_metrics(
            session_groups=[[t.channel_metrics[ch] for t in trials] for trials in session_trial_groups],
            baseline_metrics=[t.channel_metrics[ch] for t in b_trial_list],
            all_metrics=[t.channel_metrics[ch] for t in all_trials],
            band_keys=keys)

    return all_channel_slopes, channel_slopes


def _fill_corrected_metrics(session_groups: List[List[Metrics]], baseline_metrics: List[Metrics], all_metrics: List[Metrics], band_keys: List[str]) -> Dict[str, float]:
    """
        Fits one slope per key and fills in corrected_mae / corrected_mae_bandwise for one signal (the whole tensor or one channel).
        session_groups: metrics of one subject and session per list, volatility and MAE are centered within each list before fitting.
        baseline_metrics: metrics of all baseline trials, their mean volatility is the reference of the correction.
        all_metrics: metrics of all baseline and post-patching trials, these get corrected.
        Returns the fitted slopes with keys as keys.
    """
    # one slope per band, then correct every trial relative to the mean baseline volatility
    slopes = {}

    # get one slope per band by fitting a line to the centered volatility and MAE of all trials across all subjects
    for k in band_keys:
        centered_vols, centered_maes = [], []
        # center volatility and MAE within each session group (subject + session)
        for group in session_groups:
            centered_vols.extend(_center([_band_value(m, 'signal_volatility', k) for m in group]))
            centered_maes.extend(_center([_band_value(m, 'mae', k) for m in group]))
        slopes[k] = float(np.polyfit(centered_vols, centered_maes, 1)[0])

    # correct every trial's MAE relative to the mean baseline volatility
    for m in all_metrics:
        m.corrected_mae_bandwise = {}
    for k in band_keys:
        b_mean_volatility = float(np.mean([_band_value(m, 'signal_volatility', k) for m in baseline_metrics]))
        for m in all_metrics:
            corrected = _band_value(m, 'mae', k) - slopes[k] * (
                        _band_value(m, 'signal_volatility', k) - b_mean_volatility)
            if k == ALL_BANDS:
                m.corrected_mae = corrected
            else:
                m.corrected_mae_bandwise[k] = corrected
    return slopes


def _band_value(metrics: Metrics, metric_name: str, band: str) -> float:
    """Returns the global value of metric_name for "all bands", otherwise its band-wise value for band."""
    return getattr(metrics, metric_name) if band == ALL_BANDS else getattr(metrics, f"{metric_name}_bandwise")[band]


def _center(values: List[float]) -> List[float]:
    """Subtracts the mean of values from each value."""
    mean_value = np.mean(values)
    return [value - mean_value for value in values]

def calc_subjectwise_channel_metric_stats(trial_list: List[Trial], bands: List[str], metric: str,  ch_names: List[str]=None, group_by: str = 'subject') -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Calculates the mean metric and SEM for each subject across all trials in trial_list.
    Returns a dictionary with subject names as keys and channel-wise mean metric including global(all bands) values as values.
    """
    subjectwise_metric = {}
    subjectwise_sem = {}

    trials_by_subject = _group_by(trial_list, 'subject')

    for subject, trials in trials_by_subject.items():
        subjectwise_metric[subject], subjectwise_sem[subject], _ = get_channel_bandwise_metric_stats(trials, metric_name=metric, ch_names=ch_names, bands=bands)
        global_metric_mean, global_metric_sem, _= get_channel_metric_stats(trials, metric_name=metric, ch_names=ch_names)
        for ch in subjectwise_metric[subject]:
            subjectwise_metric[subject][ch]['all bands'] = global_metric_mean[ch]
            subjectwise_sem[subject][ch]['all bands'] = global_metric_sem[ch]

    if group_by == 'channel':
        # Reorganize the data to be grouped by channel instead of subject
        by_channel_metric = {}
        by_channel_sem = {}
        for subject in subjectwise_metric.keys():
            for ch in subjectwise_metric[subject].keys():
                by_channel_metric.setdefault(ch, {})
                by_channel_sem.setdefault(ch, {})
                by_channel_metric[ch][subject] = subjectwise_metric[subject][ch]
                by_channel_sem[ch][subject] = subjectwise_sem[subject][ch]

        return by_channel_metric, by_channel_sem


    return subjectwise_metric, subjectwise_sem

def calc_subjectwise_metric_stats(trial_list: List[Trial], bands: List[str], metric: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Calculates the mean metric and SEM for each subject across all trials in trial_list.
    Returns a dictionary with subject names as keys and bandwise mean metric including global values as values.
    """
    subjectwise_metric = {}
    subjectwise_sem = {}

    trials_by_subject = _group_by(trial_list, 'subject')

    for subject, trials in trials_by_subject.items():
        subjectwise_metric[subject], subjectwise_sem[subject], _ = get_bandwise_metric_stats(trials, metric_name=metric, bands=bands)
        subjectwise_metric[subject]['all bands'], subjectwise_sem[subject]['all bands'], _ = get_metric_stats(trials, metric_name=metric)


    return subjectwise_metric, subjectwise_sem



def get_subjectwise_metric_mean(trial_list: List[Trial], bands: List[str], metric_name: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculates metric mean for each subject, then calculates the mean and sem of those means.
    Returns a dictionary with bandwise metric means and sems and global metric mean and sem.
    """
    by_subject_means, _ = calc_subjectwise_metric_stats(trial_list, bands, metric_name)
    band_means = {}
    band_sems = {}
    for subj in by_subject_means.keys():
        for b, mean in by_subject_means[subj].items():
            band_means.setdefault(b, []).append(mean)

    for b, means in band_means.items():
        band_means[b] = np.mean(means)
        band_sems[b] = sem(means)

    return band_means, band_sems

def calc_subjectwise_time_series(trial_list: List[Trial], metric_name: str, band: str, channels: List[int] = None) -> Dict[str, np.ndarray]:
    """
    Calculates the mean time series of metric_name in band for each subject, averaged over the subject's trials and the given channels.
    Expects trials loaded with a time series file. channels: indices into the channel dimension, None averages all channels.
    Returns {subject: array of shape (time,)}.
    """
    subjectwise_series = {}
    for subject, trials in _group_by(trial_list, 'subject').items():
        series = np.stack([trial.time_series(metric_name, bands=[band])[band] for trial in trials])  # (trials, channels, time)
        if channels is not None:
            series = series[:, channels, :]
        subjectwise_series[subject] = series.mean(axis=(0, 1))
    return subjectwise_series

def get_subjectwise_time_series_mean(trial_list: List[Trial], metric_name: str, band: str, channels: List[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculates the time series of metric_name for each subject (see calc_subjectwise_time_series), then the mean and SEM of those over subjects.
    Returns the mean and SEM, both of shape (time,), and the number of subjects.
    """
    subject_series = np.stack(list(calc_subjectwise_time_series(trial_list, metric_name, band, channels).values()))
    return subject_series.mean(axis=0), sem(subject_series, axis=0), subject_series.shape[0]

def get_subjectwise_time_series_difference(b_trial_list: List[Trial], p_trial_list: List[Trial], metric_name: str, band: str, channels: List[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculates each subject's post - baseline difference of their mean time series (see calc_subjectwise_time_series),
    then the mean and SEM of those differences over subjects. Only subjects with both sessions are used.
    Returns the mean and SEM, both of shape (time,), and the number of subjects.
    """
    b_series = calc_subjectwise_time_series(b_trial_list, metric_name, band, channels)
    p_series = calc_subjectwise_time_series(p_trial_list, metric_name, band, channels)
    subjects = [s for s in b_series if s in p_series]
    diffs = np.stack([p_series[s] - b_series[s] for s in subjects])
    return diffs.mean(axis=0), sem(diffs, axis=0), len(subjects)

def calc_subjectwise_channel_d(b_metrics: Dict[str, Dict[str, Dict[str, float]]], p_metrics: Dict[str, Dict[str, Dict[str, float]]], group_by_bands: bool, ddof: int = 1, effect_size: str = "d_z") -> Dict[str, Dict[str, float]]:
    """
    Calculates a subject-level effect size of the change after patching for every channel and band, with d = the subjects' differences (post - baseline):
        "d_z":   Cohen's d_z = mean(d) / sd(d), the paired effect size (consistency of the change across subjects)
        "glass": Glass's delta = mean(d) / sd(baseline subject means), the change relative to how much subjects differ at baseline
    Expects subject means as {channel: {subject: {band: value}}} for both sessions, e.g. the outputs of calc_subject_adjusted_channel_metric
    or of calc_subjectwise_channel_metric_stats(..., group_by='channel'). Only subjects with both sessions are used.
    ddof: delta degrees of freedom of the standard deviation in the denominator. 1 (default) gives the sample SD, which for d_z matches the paired t-test (d_z = t / sqrt(N)).
    Returns {channel: {band: effect size}}, or {band: {channel: effect size}} if group_by_bands is True (one dict per topomap).
    """
    if effect_size not in ("d_z", "glass"):
        raise ValueError(f"Unknown effect_size '{effect_size}', expected 'd_z' or 'glass'")

    channel_d = {}
    for ch in b_metrics.keys():
        bw_subject_diffs = {}
        bw_baselines = {}
        for subject in b_metrics[ch].keys():
            if subject not in p_metrics[ch]:
                continue
            for band in b_metrics[ch][subject].keys():
                bw_subject_diffs.setdefault(band, [])
                bw_baselines.setdefault(band, [])
                subj_diff = p_metrics[ch][subject][band] - b_metrics[ch][subject][band]
                bw_subject_diffs[band].append(subj_diff)
                bw_baselines[band].append(b_metrics[ch][subject][band])
        denominators = bw_subject_diffs if effect_size == "d_z" else bw_baselines
        bw_d = {band: float(np.mean(diffs) / np.std(denominators[band], ddof=ddof)) for band, diffs in bw_subject_diffs.items()}
        channel_d[ch] = bw_d

    if group_by_bands:
        band_d = {}
        for ch, bw_d in channel_d.items():
            for band, d in bw_d.items():
                band_d.setdefault(band, {})
                band_d[band][ch] = d
        return band_d

    return channel_d


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
    """
    Renames channel names in place to the MNE standard_1005 montage names, e.g. "EEG FP1-REF" -> "Fp1", "EEG T1-REF" -> "FT9", "EEG T2-REF" -> "FT10".
    """
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

def create_topomap_info() -> mne.Info:
    """
    Creates the mne.Info for topomaps of the model channels, in CHORDER_STANDARD order (= ch_0 ... ch_22 of the metrics).
    The channels keep their 10-20 labels (see rename_chans_to_mne), but are placed at the actual positions of the EGI HydroCel electrodes
    they were recorded with (EGI_TO_1020_MAP, GSN-HydroCel-65 montage, where the reference electrode VREF is "Cz").
    """
    egi_by_channel = {channel: egi for egi, channel in settings.EGI_TO_1020_MAP.items()}
    egi_names = ["Cz" if egi_by_channel[ch] == "EEG VREF" else "E" + egi_by_channel[ch].split(" ")[-1] for ch in settings.CHORDER_STANDARD]

    info = mne.create_info(ch_names=egi_names, sfreq=settings.RSFREQ, ch_types='eeg')  # sfreq is required but not relevant for static topomaps
    info.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-65_1.0"))

    labels = list(settings.CHORDER_STANDARD)
    rename_chans_to_mne(labels)
    mne.rename_channels(info, dict(zip(egi_names, labels)))
    return info

def order_to_channel_viewer_format(dataset_data: List[List[Trial]], dataset_metrics: List[List[Dict[str, Any]]], sample_idx, channel_idx) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Selects one sample and channel across all datasets for the single channel viewer.
    dataset_data: one list of trials per dataset, dataset_metrics: the matching output of get_channel_viewer_metrics per dataset.
    Returns per dataset the {"output", "target"} arrays of the channel and its metrics dict.
    """
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
    """
    Collects the requested channel-level metrics of every trial for the single channel viewer.
    Returns one list per trial with one {"metrics": {metric_name: value}, "metadata": {subject, condition, id}} dict per channel.
    Channels missing a metric are reported and the metric is left out.
    """
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
    """
    Calculates the mean, SEM (ddof=1) and SD (ddof=0) over the trials in trial_list of a global (not band-wise) channel-level metric, e.g. "nmae".
    A "_bandwise" suffix is removed, so "nmae_bandwise" also returns the global channel value.
    If ch_names is provided, only those channels are used.
    Returns three dictionaries {channel: value}.
    """
    if "bandwise" in metric_name:
        metric_name = metric_name.replace("_bandwise", "")
    channel_values = _extract_channel_metric(trial_list, metric_name, ch_names)
    ch_means = {}
    ch_sems = {}
    ch_stds = {}
    for ch_name, vals in channel_values.items():
        if len(vals) == 0:
            print(f"WARNING: No values found for channel {ch_name} for metric {metric_name}. This channel will be skipped in the averaging.")
            continue
        ch_means[ch_name] = np.mean(vals)
        ch_sems[ch_name] = sem(vals)
        ch_stds[ch_name] = np.std(vals)
    return ch_means, ch_sems, ch_stds


def get_channel_bandwise_metric_stats(trial_list: List[Trial], metric_name: str, bands: List[str], ch_names: List[str]=None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Calculates the mean, SEM (ddof=1) and SD (ddof=0) over the trials in trial_list of a band-wise channel-level metric for every band in bands.
    "_bandwise" is appended to metric_name if missing, so "nmae" and "nmae_bandwise" are equivalent.
    If ch_names is provided, only those channels are used.
    Returns three dictionaries {channel: {band: value}}.
    """
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
        ch_means[ch_name] = {b: np.mean(band_vals) for b, band_vals in vals.items()}
        ch_sems[ch_name] = {b: sem(band_vals) for b, band_vals in vals.items()}
        ch_stds[ch_name] = {b: np.std(band_vals) for b, band_vals in vals.items()}
    return ch_means, ch_sems, ch_stds


def get_metric_stats(trial_list: List[Trial], metric_name: str) -> Tuple[float, float, float]:
    """
    Calculates the mean, SEM (ddof=1) and SD (ddof=0) over the trials in trial_list of a global (not band-wise) trial-level metric, e.g. "nmae".
    A "_bandwise" suffix is removed, so "nmae_bandwise" also returns the global value.
    """
    if "bandwise" in metric_name:
        metric_name = metric_name.replace("_bandwise", "")
    values = extract_metric(trial_list, metric_name)
    return np.mean(values), sem(values), np.std(values)


def get_bandwise_metric_stats(trial_list: List[Trial], metric_name: str, bands: List[str]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Calculates the mean, SEM (ddof=1) and SD (ddof=0) over the trials in trial_list of a band-wise trial-level metric for every band in bands.
    "_bandwise" is appended to metric_name if missing, so "nmae" and "nmae_bandwise" are equivalent.
    Returns three dictionaries {band: value}.
    """
    if 'bandwise' not in metric_name:
        metric_name += '_bandwise'

    bandwise_values = extract_metric(trial_list, metric_name, bands=bands, bandwise=True)

    means = {}
    sems = {}
    stds = {}
    for b, vals in bandwise_values.items():
        means[b] = np.mean(vals)
        sems[b] = sem(vals)
        stds[b] = np.std(vals)
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

def extract_metric(trial_list: List[Trial], metric_name: str, bands: List[str]=None, bandwise: bool=False) -> List[Any] | Dict[str, List[Any]]:
    """
    Internal helper function to extract global metric values from a list of Trial objects.
    If bands is provided, only band-wise metrics for those bands will be extracted.
    Returns a list of metric values if bandwise is False, or a dictionary with band names as keys and lists of metric values as values if bandwise is True.
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
    """Groups the trials by the value of one of their attributes, e.g. "subject". Returns {attribute value: [trials]}."""
    trials_by_attr = defaultdict(list)
    for trial in trial_list:
        trials_by_attr[getattr(trial, attr_name)].append(trial)
    return trials_by_attr