"""The sliding (time-resolved) LDA decoder."""
from typing import Callable, List

import numpy as np
from mne.decoding import SlidingEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def subject_standardize(X_train: np.ndarray, subjects_train: List[str], X_test: np.ndarray, subjects_test: List[str]):
    """
    Z-scores every feature (all axes after the trial axis) per subject, with the mean and SD of that subject's training
    trials, so the pooled decoder isn't dominated by amplitude differences between heads. Returns (X_train, X_test).
    Every test subject needs training trials (within-subject folds).
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    subjects_train, subjects_test = np.asarray(subjects_train), np.asarray(subjects_test)
    for subject in np.unique(subjects_train):
        train, test = subjects_train == subject, subjects_test == subject
        mean, sd = X_train[train].mean(axis=0), X_train[train].std(axis=0) + 1e-12
        X_train[train] = (X_train[train] - mean) / sd
        X_test[test] = (X_test[test] - mean) / sd
    return X_train, X_test


def sliding_lda(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """
    Fits one shrinkage LDA per time point (mne SlidingEstimator; lsqr solver, Ledoit-Wolf shrinkage, which standardizes
    the features internally) on X_train (n_trials, n_features, n_times) and returns the decision function of X_test,
    (n_test, n_times, n_classes). Its softmax is the LDA's class probabilities.
    """
    estimator = SlidingEstimator(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), n_jobs=n_jobs, verbose=False)
    estimator.fit(X_train, y_train)
    return estimator.decision_function(X_test)


def decode_bands(X_train: np.ndarray, y_train: np.ndarray, subjects_train: List[str], X_test: np.ndarray,
                 subjects_test: List[str], n_jobs: int = 1) -> np.ndarray:
    """
    One sliding LDA per band on the channels of that band. X: (n_trials, n_chans, n_bands, n_times), standardized per
    subject first (subject_standardize). Returns the decision function of the test trials, (n_test, n_bands, n_times, n_classes).
    """
    X_train, X_test = subject_standardize(X_train, subjects_train, X_test, subjects_test)
    return np.stack([sliding_lda(X_train[:, :, b], y_train, X_test[:, :, b], n_jobs) for b in range(X_train.shape[2])], axis=1)


def fit_per_subject(decode: Callable[..., np.ndarray], X_train: np.ndarray, y_train: np.ndarray, subjects_train: List[str],
                    X_test: np.ndarray, subjects_test: List[str], *args, **kwargs) -> np.ndarray:
    """
    Runs decode (decode_bands or searchlight, with its further args) separately on every subject's trials: each
    subject's test trials get the decision function of models trained on that subject's training trials only, in the
    original order. The classes are those of all training trials together (the last axis); a class missing from a
    subject's training trials gets -inf, probability 0 after the softmax.
    """
    subjects_train, subjects_test = np.asarray(subjects_train), np.asarray(subjects_test)
    classes = np.unique(y_train)
    outputs = None
    for subject in np.unique(subjects_test):
        train, test = subjects_train == subject, subjects_test == subject
        subject_classes = np.unique(y_train[train])
        decision = decode(X_train[train], y_train[train], subjects_train[train], X_test[test], subjects_test[test], *args, **kwargs)
        if len(subject_classes) == 2:  # binary LDA: one column, the score of the second class
            decision = np.stack([-decision, decision], axis=-1)
        if outputs is None: # fill all classes with -inf (in case a class is missing from training trials)
            outputs = np.full((len(subjects_test), *decision.shape[1:-1], len(classes)), -np.inf, dtype=decision.dtype)
        subject_outputs = np.full((test.sum(), *decision.shape[1:-1], len(classes)), -np.inf, dtype=decision.dtype)
        subject_outputs[..., np.searchsorted(classes, subject_classes)] = decision
        outputs[test] = subject_outputs
    return outputs


def searchlight(X_train: np.ndarray, y_train: np.ndarray, subjects_train: List[str], X_test: np.ndarray,
                subjects_test: List[str], clusters: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """
    Searchlight decoding: one sliding LDA per channel cluster, on the channels of that cluster only.
    X: (n_trials, n_chans, n_times) of one band, standardized per subject first (subject_standardize).
    clusters: (n_clusters, n_chans) bool, row c the channels of cluster c (e.g. core.utils.build_neighbor_mask: a channel
    and its nearest electrodes). Returns the decision function of the test trials, (n_test, n_clusters, n_times, n_classes).
    """
    X_train, X_test = subject_standardize(X_train, subjects_train, X_test, subjects_test)
    return np.stack([sliding_lda(X_train[:, cluster], y_train, X_test[:, cluster], n_jobs) for cluster in clusters], axis=1)
