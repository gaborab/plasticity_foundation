"""
Subject-level paired sign-flip permutation test with max-t correction (Nichols & Holmes 2002), single-step and
Westfall-Young step-down (Westfall & Young 1993). Used by notebooks/max_t_permutation.ipynb to find the channels and bands
that differ between baseline and post-patching while controlling the family-wise error rate (FWER) over all of them.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MaxTResult:
    """
    Result of max_t_sign_flip_test for the m cells (columns) of the difference matrix, every array in the column order of the input.
    t, mean, n: paired t-value, mean difference and number of subjects of every cell. t and the p-values are NaN for cells left out of the family.
    p_single_step: FWER-corrected p-values of the single-step test, every |t| against the null distribution of max |t| over all cells.
    p_step_down: FWER-corrected p-values of the step-down test, never larger than p_single_step, with the same FWER control.
    null_max: max |t| over all cells of every permutation, the null distribution of the single-step test.
    """
    t: np.ndarray
    mean: np.ndarray
    n: np.ndarray
    p_single_step: np.ndarray
    p_step_down: np.ndarray
    null_max: np.ndarray

    def critical_t(self, alpha: float = 0.05) -> float:
        """|t| a cell has to exceed (strictly) to have p_single_step <= alpha, inf if no |t| can reach alpha with this many permutations."""
        n_exceed = int(np.floor(alpha * (len(self.null_max) + 1))) - 1  # permutations allowed at or above |t| for p <= alpha
        if n_exceed < 0:
            return np.inf
        return float(np.sort(self.null_max)[::-1][n_exceed])


def _t_values(sums: np.ndarray, n: np.ndarray, sum_squares: np.ndarray) -> np.ndarray:
    """Paired t = mean / (SD / sqrt(n)), SD with ddof=1, from the sums and sums of squares of the differences of every cell."""
    mean = sums / n
    var = (sum_squares - n * mean ** 2) / (n - 1)
    return mean / np.sqrt(var / n)


def max_t_sign_flip_test(diffs: np.ndarray, n_permutations: int = 10000, seed: int = None, min_subjects: int = 10,
                         batch_size: int = 1000) -> MaxTResult:
    """
    Paired sign-flip permutation test of all columns of diffs at once, corrected for multiple comparisons with max |t|.
    diffs: (subjects, cells) paired differences (post - baseline), NaN where a subject has no value in a cell (left out of that cell only).
    Per cell, t = mean / (SD / sqrt(n)) over the subjects' finite differences.
    Under H0 (no change, every subject's differences symmetric around 0) the sign of each subject is exchangeable. A permutation
    flips the sign of each subject's whole row, so the dependence between cells (neighbouring channels, overlapping bands, metrics)
    is kept, and comparing every |t| with the null distribution of the maximum controls the FWER over all cells.
    The step-down test compares the k-th largest |t| only with the maximum over the cells with smaller or equal |t|, and p-values
    are made monotone in |t|. The observed signs count as one permutation: p = (1 + #{null >= |t|}) / (n_permutations + 1).
    min_subjects: cells with fewer finite differences, or without variance, are left out of the family (t and p-values NaN).
    seed: seed of the random signs, for reproducible p-values.
    batch_size: permutations computed at once, (batch_size, cells) arrays are kept in memory.
    """
    diffs = np.asarray(diffs, dtype=float)
    finite = np.isfinite(diffs)
    n = finite.sum(axis=0)
    filled = np.where(finite, diffs, 0.0)  # a zero adds nothing to the sums, whatever its sign
    sum_squares = (filled ** 2).sum(axis=0)  # the same in every permutation
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = filled.sum(axis=0) / n
        t = _t_values(filled.sum(axis=0), n, sum_squares)
    tested = np.flatnonzero((n >= min_subjects) & np.isfinite(t))
    if len(tested) == 0:
        raise ValueError(f"No cell has at least {min_subjects} subjects with finite differences and non-zero variance")
    t[np.setdiff1d(np.arange(len(t)), tested)] = np.nan

    # the tested cells in order of decreasing observed |t|, the order of the step-down test
    order = tested[np.argsort(-np.abs(t[tested]))]
    observed = np.abs(t[order])
    columns, n_sorted, sum_squares_sorted = filled[:, order], n[order], sum_squares[order]

    rng = np.random.default_rng(seed)
    null_max = np.empty(n_permutations)
    n_step_down_exceed = np.zeros(len(order), dtype=np.int64)
    for start in range(0, n_permutations, batch_size):
        n_batch = min(batch_size, n_permutations - start)
        signs = rng.choice([-1.0, 1.0], size=(n_batch, diffs.shape[0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            perm_t = np.abs(_t_values(signs @ columns, n_sorted, sum_squares_sorted))
        # column k: max |t| over the k-th cell and all cells with a smaller observed |t|
        successive_max = np.maximum.accumulate(perm_t[:, ::-1], axis=1)[:, ::-1]
        n_step_down_exceed += (successive_max >= observed).sum(axis=0)
        null_max[start:start + n_batch] = successive_max[:, 0]

    sorted_null = np.sort(null_max)
    n_single_step_exceed = n_permutations - np.searchsorted(sorted_null, observed, side="left")
    p_single_step = np.full(len(t), np.nan)
    p_single_step[order] = (1 + n_single_step_exceed) / (n_permutations + 1)
    p_step_down = np.full(len(t), np.nan)
    # a cell can't be more significant than a cell with a larger |t|
    p_step_down[order] = np.maximum.accumulate((1 + n_step_down_exceed) / (n_permutations + 1))
    return MaxTResult(t=t, mean=mean, n=n, p_single_step=p_single_step, p_step_down=p_step_down, null_max=null_max)
