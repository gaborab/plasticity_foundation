"""
Per-subject metric tables of the sliding decoder, the cluster test against chance (Task A) and the latency, jackknife and
side contrasts of Task B (HANDOFF_location_decoding.md).
"""
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy.ndimage import uniform_filter1d
from scipy.special import softmax
from scipy.stats import rankdata, t as t_dist, ttest_1samp, wilcoxon

# Q4 slots: the location classes relative to the patched eye, (side relative to the patched eye, eccentricity)
SIDE_SLOTS = [("patched", "far"), ("patched", "mid"), ("patched", "near"), ("other", "far"), ("other", "mid"),
              ("other", "near"), ("top", "mid"), ("top", "near")]
# the pre-specified contrasts of handoff B5: (name, eccentricity)
Q4_CONTRASTS = [("patched - other", "near"), ("patched - other", "mid"), ("patched - other", "far"),
                ("top - sides", "near"), ("top - sides", "mid")]


# ---------------------------------------------------------------- Task A: metrics per subject
def balanced_accuracy(labels: np.ndarray, scores: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced accuracy of scores (n_trials, ..., n_classes) for every index of the middle axes: mean recall over the classes present."""
    pred = scores.argmax(axis=-1)
    recalls = [(pred[labels == k] == k).mean(axis=0) for k in range(n_classes) if (labels == k).any()]
    return np.mean(recalls, axis=0)


def ovr_auc(labels: np.ndarray, scores: np.ndarray, n_classes: int) -> np.ndarray:
    """
    One-vs-rest AUC of every class for every index of the middle axes of scores (n_trials, ..., n_classes), as
    sklearn's roc_auc_score (Mann-Whitney U with averaged ties). Returns (..., n_classes), NaN for a class without trials.
    """
    ranks = rankdata(scores, axis=0)
    auc = np.full(scores.shape[1:], np.nan)
    for k in range(n_classes):
        positive = labels == k
        n_pos, n_neg = positive.sum(), (~positive).sum()
        if n_pos and n_neg:
            auc[..., k] = (ranks[positive][..., k].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def time_metrics_frame(trials, model: str, session: str, set_names: Sequence[str], times: np.ndarray, n_classes: int) -> pd.DataFrame:
    """
    Long table of the per-subject metrics of a sliding decoder over time, from its trials (load_inference_trials with
    stim_patch=None; output (n_sets, n_times, n_classes) of decision values, e.g. one set per band):
    model, session, subject, set, time, metric, class, value, n_trials.
    metric "balanced_accuracy" (class -1, chance 1 / n_classes) and "auc" (one-vs-rest per class, chance 0.5), both on
    the softmax probabilities of each subject's trials.
    """
    by_subject = defaultdict(list)
    for trial in trials:
        by_subject[trial.subject].append(trial)
    frames = []
    for subject, subject_trials in by_subject.items():
        scores = softmax(np.stack([trial.recon_data for trial in subject_trials]), axis=-1)  # (n, sets, times, classes)
        labels = np.array([int(trial.target_data) for trial in subject_trials])
        bal = balanced_accuracy(labels, scores, n_classes)  # (sets, times)
        auc = ovr_auc(labels, scores, n_classes)  # (sets, times, classes)
        set_idx, time_idx = np.meshgrid(np.arange(len(set_names)), np.arange(len(times)), indexing="ij")
        common = {"model": model, "session": session, "subject": subject, "n_trials": len(subject_trials)}
        frames.append(pd.DataFrame({**common, "set": np.asarray(set_names)[set_idx.ravel()], "time": times[time_idx.ravel()],
                                    "metric": "balanced_accuracy", "class": -1, "value": bal.ravel()}))
        set_idx, time_idx, class_idx = np.meshgrid(np.arange(len(set_names)), np.arange(len(times)), np.arange(n_classes), indexing="ij")
        frames.append(pd.DataFrame({**common, "set": np.asarray(set_names)[set_idx.ravel()], "time": times[time_idx.ravel()],
                                    "metric": "auc", "class": class_idx.ravel(), "value": auc.ravel()}))
    return pd.concat(frames, ignore_index=True)


def subject_curves(metrics_df: pd.DataFrame, value_filter: Dict[str, object]) -> pd.DataFrame:
    """The per-subject curves of the rows matching value_filter ({column: value}), as a subject x time table."""
    rows = metrics_df
    for column, value in value_filter.items():
        rows = rows[rows[column] == value]
    return rows.pivot_table(index="subject", columns="time", values="value")


def cluster_test_vs_chance(curves: np.ndarray, chance: float, n_permutations: int = 5000, seed: int = 0) -> List[Tuple[int, int, float]]:
    """
    One-sample cluster permutation test of subject curves (n_subjects, n_times) against chance, one-sided (above chance),
    as handoff A5. Returns (first index, last index, p) of every cluster.
    """
    _, clusters, cluster_p, _ = permutation_cluster_1samp_test(curves - chance, tail=1, n_permutations=n_permutations,
                                                               seed=seed, out_type="indices", verbose=False)
    # every cluster is a tuple with one array of time indices (1-D data)
    return [(int(c[0][0]), int(c[0][-1]), float(p)) for c, p in zip(clusters, cluster_p)]


# ---------------------------------------------------------------- Task B: latency, jackknife, side contrasts
def smooth(curves: np.ndarray, n_samples: int) -> np.ndarray:
    """Moving average of n_samples along the last axis (time), the edges padded with their nearest value."""
    return uniform_filter1d(curves, size=max(n_samples, 1), axis=-1, mode="nearest") if n_samples > 1 else curves


def fractional_peak_latency(times: np.ndarray, curve: np.ndarray, chance: float, fraction: float = 0.5,
                            search: Tuple[float, float] = (0.0, 0.5)) -> float:
    """
    First time in the search window where curve - chance reaches fraction * (peak - chance), the peak taken in the search
    window, linearly interpolated between the samples (handoff B2). NaN if the curve never rises above chance.
    """
    inside = (times >= search[0] - 1e-9) & (times <= search[1] + 1e-9)
    t, c = times[inside], curve[inside]
    peak = np.nanmax(c)
    if not np.isfinite(peak) or peak <= chance:
        return np.nan
    threshold = chance + fraction * (peak - chance)
    i = int(np.argmax(c >= threshold))
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (threshold - c[i - 1]) / (c[i] - c[i - 1]) * (t[i] - t[i - 1]))


def fractional_area_latency(times: np.ndarray, curve: np.ndarray, chance: float, fraction: float = 0.5,
                            search: Tuple[float, float] = (0.0, 0.5), floor: float = 0.0) -> float:
    """
    Time in the search window at which the area of the curve above a floor (its positive part, max(curve - floor level, 0),
    trapezoidal), accumulated from the start of the window, reaches fraction of its total in the window, linearly
    interpolated. The floor level is chance + floor * (peak in the window - chance): with floor > 0 the low-level noise
    before and after the response adds no area. Unlike the fractional peak latency it changes continuously with the
    curve, so a small bump near a threshold can't make it jump, and the jackknife over subjects stays valid. NaN if the
    curve never rises above chance in the window.
    """
    inside = (times >= search[0] - 1e-9) & (times <= search[1] + 1e-9)
    t, c = times[inside], np.nan_to_num(curve[inside] - chance, nan=0.0)
    peak = c.max()
    if peak <= 0:
        return np.nan
    above = np.clip(c - floor * peak, 0, None)
    cumulative = np.concatenate([[0.0], np.cumsum((above[1:] + above[:-1]) / 2 * np.diff(t))])
    if cumulative[-1] <= 0:
        return np.nan
    target = fraction * cumulative[-1]
    i = int(np.searchsorted(cumulative, target))  # first sample with cumulative >= target, cumulative[i - 1] < target
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (target - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1]) * (t[i] - t[i - 1]))


def jackknife(statistic: Callable[[np.ndarray], float], data: np.ndarray) -> Dict[str, float]:
    """
    Jackknife test of statistic, computed on the mean over subjects of data (n_subjects, ...), as handoff B3
    (Miller, Patterson & Ulrich, 1998): estimate = statistic of the full mean, SE_J from the leave-one-subject-out
    means, t = estimate / SE_J with n - 1 df, two-sided p.
    """
    n = len(data)
    estimate = statistic(data.mean(axis=0))
    loo = np.array([statistic(np.delete(data, i, axis=0).mean(axis=0)) for i in range(n)])
    se = np.sqrt((n - 1) / n * np.nansum((loo - np.nanmean(loo)) ** 2))
    t = estimate / se if se > 0 else np.nan
    return {"estimate": estimate, "se": se, "t": t, "df": n - 1, "p": 2 * t_dist.sf(abs(t), n - 1) if np.isfinite(t) else np.nan, "n": n}


def patched_eyes(dominance_csv: Path, rightpatch_csv: Path) -> Dict[str, str]:
    """
    {subject: "L" or "R"}, the patched eye: subjects of dominance_csv were patched on their dominant eye, those of
    rightpatch_csv (from STMD35 on) on the right eye.
    """
    early, late = pd.read_csv(dominance_csv), pd.read_csv(rightpatch_csv)
    eyes = {s: ("L" if d == "left" else "R") for s, d in zip(early.subject_id, early.eye_dominance)}
    eyes.update({s: "R" for s in late.subject_id})
    return eyes


def side_slots(class_values: np.ndarray, subjects: Sequence[str], eyes: Dict[str, str], location_classes: Dict[str, dict]):
    """
    Reorders per-class values (n_subjects, n_classes, ...) into SIDE_SLOTS (n_subjects, n_slots, ...), the sides relative
    to every subject's patched eye: "patched" = the side of the patched eye, "other" the other side, "top" the top wall.
    location_classes: settings.LOCATION_CLASSES. Subjects without a known patched eye are left out.
    Returns (slot values, kept subjects).
    """
    by_class = {info["class_id"]: info for info in location_classes.values()}
    kept = [i for i, subject in enumerate(subjects) if subject in eyes]
    out = np.full((len(kept), len(SIDE_SLOTS), *class_values.shape[2:]), np.nan)
    for row, i in enumerate(kept):
        for class_id, info in by_class.items():
            side = "top" if info["side"] == "T" else ("patched" if info["side"] == eyes[subjects[i]] else "other")
            out[row, SIDE_SLOTS.index((side, info["ecc"]))] = class_values[i, class_id]
    return out, [subjects[i] for i in kept]


def contrast(slot_values: np.ndarray, name: str, ecc: str) -> np.ndarray:
    """A Q4_CONTRASTS contrast of slot values (..., n_slots): patched - other side, or top - mean of both sides, at ecc."""
    value = lambda side: slot_values[..., SIDE_SLOTS.index((side, ecc))]
    if name == "patched - other":
        return value("patched") - value("other")
    return value("top") - (value("patched") + value("other")) / 2


def accuracy_contrasts(baseline: np.ndarray, postpatch: np.ndarray) -> pd.DataFrame:
    """
    Q4 accuracy contrasts (handoff B4) of per-subject slot values (n_subjects, n_slots), e.g. the mean AUC in a window:
    Δ = post-patching - baseline per class, the contrast per subject, one-sample t-test against 0 across subjects
    (Wilcoxon as robustness check). Also the baseline-only contrasts (same contrasts without Δ).
    """
    rows = []
    for name, ecc in Q4_CONTRASTS:
        for kind, values in (("change", contrast(postpatch - baseline, name, ecc)), ("baseline only", contrast(baseline, name, ecc))):
            values = values[np.isfinite(values)]
            test = ttest_1samp(values, 0)
            rows.append({"contrast": name, "measure": "accuracy", "ecc": ecc, "kind": kind, "estimate": values.mean(),
                         "se": values.std(ddof=1) / np.sqrt(len(values)), "t": test.statistic, "df": len(values) - 1,
                         "p": test.pvalue, "p_wilcoxon": wilcoxon(values).pvalue, "n": len(values)})
    return pd.DataFrame(rows)


def latency_contrasts(times: np.ndarray, baseline: np.ndarray, postpatch: np.ndarray, chance: float, fraction: float,
                      search: Tuple[float, float], floor: float = 0.0) -> pd.DataFrame:
    """
    Q4 latency contrasts of per-subject slot curves (n_subjects, n_slots, n_times), already smoothed: the
    fractional area latency (fraction, floor: fractional_area_latency) of every slot on the mean curves, Δ = post-patching
    - baseline, the contrast, tested with the jackknife (each subject left out of both sessions). Also the baseline-only
    contrasts.
    """
    data = np.stack([baseline, postpatch], axis=1)  # (n_subjects, 2, n_slots, n_times)

    def slot_latencies(mean_curves):  # (n_slots, n_times) -> (n_slots,)
        return np.array([fractional_area_latency(times, curve, chance, fraction, search, floor) for curve in mean_curves])

    rows = []
    for name, ecc in Q4_CONTRASTS:
        statistics = {"change": lambda m: contrast(slot_latencies(m[1]) - slot_latencies(m[0]), name, ecc),
                      "baseline only": lambda m: contrast(slot_latencies(m[0]), name, ecc)}
        for kind, statistic in statistics.items():
            rows.append({"contrast": name, "measure": "latency", "ecc": ecc, "kind": kind, **jackknife(statistic, data)})
    return pd.DataFrame(rows)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-corrected p-values (step-down, monotone), NaN stays NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out
