"""
Automatic bad channel detection for a single raw recording.

Criteria:
    - clipping:    (raw data) the channel sits at its own maximum absolute value (amplifier saturation) for more than
                   `clip_fraction` of the samples, not counting samples where at least `common_clip_share` of the
                   channels saturate together (whole-cap events such as the start/end of a recording)
    - flat:        (raw data) standard deviation of the 1-40 Hz band-passed signal below `flat_std_uv`
    - noisy:       (average reference) 1-40 Hz standard deviation more than `noise_ratio` times the median across channels
    - line noise:  (average reference, off by default) power in line_freq +- 1 Hz relative to the neighbouring bands is
                   above `line_min_db` AND more than `line_excess_db` above the median of the good channels.
                   Disabled (`line_excess_db=None`) because in the STMD data line noise is systematic for some electrodes
                   (e.g. E34 = Pz in ~80 % of recordings) and is better handled by the notch filter than by interpolation.

Defaults were calibrated on the STMD data (qc_bad_channels.py): they flag ~2.7 % of channels, close to the ~2.6 %
selected manually. The noisy threshold is deliberately high because the data is checked before ICA, so periocular and
facial channels carry blinks and muscle activity that are physiological, not channel faults.

The noisy / line noise checks are computed on data re-referenced to the average of the channels that are still good, and
repeated until no new channel is flagged (a bad channel in the average reference would contaminate every other channel).
The raw data is referenced to the recording reference (Cz / VREF), whose own signal is common to all channels and would
otherwise mask channel-specific noise.

Detected channels are meant to be added to raw.info['bads'] and interpolated.
"""
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

DEFAULT_PARAMS = {
    'clip_fraction': 0.005,   # fraction of samples at the channel's max |x| to count as clipping
    'common_clip_share': 0.25,  # samples where at least this share of channels clip together are ignored (whole-cap events)
    'flat_std_uv': 1.0,       # uV, 1-40 Hz band, raw data
    'noise_ratio': 10.0,      # 1-40 Hz std relative to the channel median, average reference
    'line_freq': 50.0,        # Hz
    'line_min_db': 20.0,      # line band power above the neighbouring bands (dB)
    'line_excess_db': None,   # ... and this much above the median of the good channels (dB); None disables the check
    'max_iterations': 3,
    'exclude': ['EEG VREF'],  # channels never flagged (the recording reference is flat before re-referencing)
}


def _line_ratio_db(raw: mne.io.BaseRaw, picks: List[str], line_freq: float) -> np.ndarray:
    psd = raw.compute_psd(method='welch', fmin=line_freq - 7, fmax=line_freq + 7,
                          n_fft=int(raw.info['sfreq'] * 4), picks=picks, exclude=(), verbose=False)
    power, freqs = psd.get_data(), psd.freqs
    line = (freqs >= line_freq - 1) & (freqs <= line_freq + 1)
    neighbours = (((freqs >= line_freq - 6) & (freqs <= line_freq - 3)) |
                  ((freqs >= line_freq + 3) & (freqs <= line_freq + 6)))
    return 10 * np.log10(power[:, line].mean(axis=1) / power[:, neighbours].mean(axis=1))


def detect_bad_channels(raw: mne.io.BaseRaw, params: Optional[Dict] = None) -> Tuple[List[str], List[Dict]]:
    """
    Detects bad channels in a raw recording. Does not modify `raw`.

    Returns:
        bads (list): names of the flagged channels
        report (list): one dict per checked channel with its metrics and the reasons it was flagged ('' if good)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    eeg = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True, exclude=())).ch_names
    candidates = [ch for ch in eeg if ch not in p['exclude']]
    if not candidates:
        return [], []
    reasons = {ch: [] for ch in candidates}

    # --- raw-data checks ---
    data = raw.get_data(picks=candidates) * 1e6  # uV
    max_abs = np.abs(data).max(axis=1, keepdims=True)
    at_max = np.abs(data) >= max_abs * (1 - 1e-3)
    # only count channel-specific saturation: samples where many channels hit the rail together are whole-cap
    # transients (e.g. recording start/end), a segment problem rather than a channel problem
    common_mode = at_max.sum(axis=0) >= p['common_clip_share'] * len(candidates)
    clip_frac = (at_max & ~common_mode).mean(axis=1)
    band = raw.copy().pick(eeg).filter(1.0, 40.0, fir_design='firwin', verbose=False)
    raw_std = band.get_data(picks=candidates).std(axis=1) * 1e6
    for i, ch in enumerate(candidates):
        if clip_frac[i] > p['clip_fraction']:
            reasons[ch].append('clipping')
        if raw_std[i] < p['flat_std_uv']:
            reasons[ch].append('flat')

    # --- average-reference checks, iterated ---
    noise_ratio = np.full(len(candidates), np.nan)
    line_db = np.full(len(candidates), np.nan)
    line_excess = np.full(len(candidates), np.nan)
    for _ in range(p['max_iterations']):
        good = [ch for ch in candidates if not reasons[ch]]
        if len(good) < 3:
            break
        good_idx = [candidates.index(ch) for ch in good]

        band_ref = band.copy().set_eeg_reference(ref_channels=good, projection=False, verbose=False)
        ref_std = band_ref.get_data(picks=candidates).std(axis=1)
        noise_ratio = ref_std / np.median(ref_std[good_idx])

        raw_ref = raw.copy().pick(eeg).set_eeg_reference(ref_channels=good, projection=False, verbose=False)
        line_db = _line_ratio_db(raw_ref, candidates, p['line_freq'])
        line_excess = line_db - np.median(line_db[good_idx])

        new_flags = False
        for i, ch in enumerate(candidates):
            if reasons[ch]:
                continue
            if noise_ratio[i] > p['noise_ratio']:
                reasons[ch].append('noisy')
            if p['line_excess_db'] is not None and line_db[i] > p['line_min_db'] and line_excess[i] > p['line_excess_db']:
                reasons[ch].append('line_noise')
            new_flags = new_flags or bool(reasons[ch])
        if not new_flags:
            break

    report = [{
        'channel': ch,
        'clip_fraction': float(clip_frac[i]),
        'raw_std_uv': float(raw_std[i]),
        'noise_ratio': float(noise_ratio[i]),
        'line_db': float(line_db[i]),
        'line_excess_db': float(line_excess[i]),
        'reasons': '+'.join(reasons[ch]),
    } for i, ch in enumerate(candidates)]
    return [ch for ch in candidates if reasons[ch]], report
