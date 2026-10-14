"""
Ridge-regression check of masked-channel prediction, on the input epochs written by make_dataset.py
(<data_root>/pre_patching and <data_root>/post_patching: input_epochs/ and epochs_manifest.csv).

Questions:
    1. Leakage: how much of a masked channel's EEG signal can a linear combination of the visible channels predict?
       Average reference, ICA and interpolation mix channels over the 65-channel cap. If the 23 model channels inherited an
       exact linear dependency, a masked channel could be rebuilt from the others regardless of brain state (R² ~ 1).
    2. Preprocessing (--compare_root): the same analysis on a second build, e.g. a raw build (make_dataset.py --no_ica --no_auto_bad_chans --no_interpolation --no_autoreject),
       and the per subject and channel difference of R² between the builds.
    3. Sensitivity (--inject): does the masked prediction error change when a visual-cortex source gets stronger?

Masked prediction, per build, session and subject:
    - trials of all stimulus types are pooled and split in half (seeded): ridge models are fitted on one half and tested
      on the other
    - the whole input epoch (settings.EPOCH_WINDOW) is band-passed per band; every time sample is an observation
    - one model per band, target channel and masking scheme predicts the target channel's sample from the visible
      channels' samples at the same time. Only the target channel is predicted and scored:
          single:  the target channel is hidden, the other 22 are visible
          cluster: the target and its 4 nearest electrodes (HydroCel-65 positions) are hidden, the other 18 are visible
    - models are fitted on the training trials in which the target channel is not interpolated (recording- or
      autoreject-level, from the manifest); the visible channels are used as they are
    - test trials are scored separately for an intact and an interpolated target channel (R² over the pooled test samples,
      at least MIN_TEST_TRIALS trials). Interpolated channels are rebuilt from neighbouring electrodes of the full cap, so
      the difference shows how much interpolation makes a channel predictable
    - ridge penalty: RIDGE_ALPHA times the mean variance of the visible channels (training data), i.e. nearly unregularised,
      so an exact linear dependency would show up as R² ~ 1
    - eigenvalue check: smallest / largest eigenvalue of the 23-channel covariance of the 5-60 Hz signal (all trials);
      an exact linear dependency gives ~0 (~1e-12 or less in double precision)

Gain injection (--inject GAIN [GAIN ...], baseline session of the main build only):
    - source (--head_model fsaverage, default): every source of the --inject_labels aparc labels (default pericalcarine,
      primary visual cortex) of both hemispheres on the MNE fsaverage template, oriented normal to the cortex with equal
      amplitude, through the 3-layer fsaverage BEM; the HydroCel-65 electrodes are aligned with the template by their
      fiducials. fsaverage (~200 MB) is downloaded into the MNE data folder on first use.
      --head_model sphere: radial dipole at SOURCE_DEPTH of the head radius towards the midpoint of O1 and O2, 3-shell sphere
      model fitted to the electrodes, no download
    - the topography is average-referenced over the 65 channels (as in the preprocessing), then the 23 model channels are
      taken; the table reports the N_SOURCE_ROI channels where it is strongest
    - every subject's baseline trials are split in half into a pseudo-baseline A and a pseudo-post session B; the models are
      fitted on all of A and scored on B only, so no scored trial was seen in training
    - in B, the source's activity s(t) is estimated with the least-squares spatial filter of its topography g,
      s = g.x / g.g, and gain * g * s(t) is added after stimulus onset (raised-cosine ramp over ONSET_RAMP s): the visual
      source is (1 + gain) times stronger after onset, the rest of the signal is unchanged
    - every model is scored on the same B trials with and without the injection, on the samples of
      settings.POST_TARGET_WINDOW: error change = MSE(injected) / MSE(original) and signal change = variance of the target
      channel (injected) / (original). Masking is blind to the gain change where the error stays flat while the signal grows

Outputs (outputs/ridge_leakage/<label>/ and outputs/plots/ridge_leakage/<label>/):
    r2_subject_channel.csv  one row per session, subject, band, scheme, channel and test set
    summary.csv             per session, band and scheme: median and quartiles over subjects of the channel-mean R²
                            (intact target channels)
    interpolation.csv       per session and band: R² of interpolated minus intact target channel (single masking),
                            over subject x channel pairs with both test sets
    per_channel.csv         per channel: interpolation rate per session, median R² (5-60 Hz, subject-sessions pooled)
    eigenvalues.csv         per session and subject
    comparison_<compare_label>.csv  (--compare_root) per session, subject, band, scheme and channel: R² of both builds
                                    and their difference (main - compare); the compare build gets its own full output folder
    injection.csv           (--inject) per gain, subject, band, scheme and channel: MSE and target variance with and without
    report.md               method and tables formatted for the report
    topomaps_r2_{scheme}_{session}.png, topomaps_interpolation_{session}.png,
    topomaps_r2_difference_{compare_label}_{scheme}_{session}.png, injected_source_topography.png,
    topomaps_injection_{error_single|error_cluster|signal}_gain{gain}.png

Run from the repository root after make_dataset.py, e.g.
    python ridge_leakage.py --compare_root data/processed_raw --compare_label raw --inject 0.1 0.25 0.5
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from core import settings
from core.io import EpochIO
from core.plot_data_processing import create_topomap_info

OUTPUT_ROOT = settings.BASE_DIR / "outputs" / "ridge_leakage"
PLOT_ROOT = settings.PLOT_DIR / "ridge_leakage"
SESSIONS = {"baseline": settings.FINETUNE_DIR.name, "post-patching": settings.PATCH_DIR.name}
N_NEIGHBOURS = 4
RIDGE_ALPHA = 1e-3
MIN_TEST_TRIALS = 10  # R² of fewer pooled test trials is too noisy, especially for the interpolated test sets
BANDS = {"5-60 Hz": (5, 60), **settings.FREQ_BANDS}
CHANNEL_NAMES = [c.replace("EEG ", "").replace("-REF", "") for c in settings.CHORDER_STANDARD]
N_CH = len(CHANNEL_NAMES)
SOURCE_DEPTH = 0.7  # dipole position as a fraction of the head radius, towards the midpoint of O1 and O2
ONSET_RAMP = 0.02  # s, raised-cosine ramp of the injected gain after stimulus onset
N_SOURCE_ROI = 4  # the injection table reports the channels where the injected source is strongest (largest |g|)
FSAVERAGE_LABELS = ["pericalcarine"]  # aparc labels of the injected fsaverage source (pericalcarine = primary visual cortex)
QUARTILES = dict(median="median", q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75))


# ---------------------------------------------------------------- data
def nearest_electrodes(info, n_neighbours):
    """{channel index: indices of its n_neighbours nearest electrodes}, Euclidean distance of the electrode positions in info."""
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return {c: [j for j in np.argsort(dist[c]) if j != c][:n_neighbours] for c in range(len(pos))}


def interpolated_channels(manifest_path):
    """{(subject, stimulus, id): set of interpolated channel indices} from an epochs manifest."""
    manifest = pd.read_csv(manifest_path, dtype={"rec_interpolated": str, "ar_interpolated": str}).fillna("")
    return {(r.subject, r.stimulus, int(r.id)): {int(c) for c in f"{r.rec_interpolated};{r.ar_interpolated}".split(";") if c}
            for r in manifest.itertuples()}


def interpolation_rates(interp):
    """Share of the trials in which each channel is interpolated."""
    counts = np.zeros(N_CH)
    for channels in interp.values():
        counts[list(channels)] += 1
    return counts / max(len(interp), 1)


def subject_trials(inputs, subject, interp):
    """The subject's input epochs (n, 23, n_times) float64 uV and an (n, 23) bool matrix, True where the channel is interpolated."""
    X, bad = [], []
    for cond, epoch_list in sorted(inputs[subject].items()):
        for idx, epoch in epoch_list:
            X.append(epoch.numpy().astype(np.float64))
            row = np.zeros(N_CH, dtype=bool)
            row[list(interp.get((subject, cond, int(idx)), set()))] = True
            bad.append(row)
    return np.stack(X), np.stack(bad)


def build_description(root):
    """The preprocessing steps of a make_dataset.py build, from its build_info.json (written since the step flags exist)."""
    path = Path(root) / "build_info.json"
    if not path.exists():
        return "preprocessing steps not recorded: built before make_dataset.py wrote build_info.json"
    steps = json.loads(path.read_text(encoding="utf-8"))
    names = {"ica": "ICA", "auto_bad_channels": "automatic bad channel detection",
             "bad_channel_interpolation": "bad channel interpolation", "autoreject": "autoreject"}
    return ", ".join(f"{names.get(step, step)} {'on' if on else 'off'}" for step, on in steps.items())


def subject_number(subject):
    return int(subject.removeprefix("STMD"))


# ---------------------------------------------------------------- ridge
def band_filter(X, edges):
    return sosfiltfilt(butter(4, edges, btype="band", fs=settings.RSFREQ, output="sos"), X, axis=-1)


def trial_stats(X):
    """Per-trial sums over the time samples of X (n, 23, n_times): sum of x x^T, sum of x, sample count."""
    return np.einsum("nct,ndt->ncd", X, X), X.sum(axis=-1), np.full(len(X), X.shape[-1], dtype=float)


def fit_ridge(stats, sel, target, visible):
    """
    Fits a ridge model of the target channel on the visible channels from the summed statistics of the selected trials.
    Returns (a, mu): the residual of an observation x is a . (x - mu).
    """
    S, m, n = (s[sel].sum(axis=0) for s in stats)
    mu = m / n
    cov = S - n * np.outer(mu, mu)
    G = cov[np.ix_(visible, visible)]
    w = np.linalg.solve(G + RIDGE_ALPHA * np.trace(G) / len(visible) * np.eye(len(visible)), cov[visible, target])
    a = np.zeros(N_CH)
    a[target] = 1.0
    a[visible] = -w
    return a, mu


def evaluate(stats, sel, target, a, mu):
    """(MSE, R², target variance) of a fitted model over the pooled samples of the selected trials."""
    S, m, n = (s[sel].sum(axis=0) for s in stats)
    centred = S - np.outer(m, mu) - np.outer(mu, m) + n * np.outer(mu, mu)
    mse = (a @ centred @ a) / n
    var = S[target, target] / n - (m[target] / n) ** 2
    r2 = 1 - mse / var if var > 1e-12 else np.nan  # a flat channel (e.g. disconnected, in a build without interpolation) has no R²
    return mse, r2, var


def masking_schemes(c, neighbours):
    """{scheme: visible channel indices} for target channel c."""
    return {scheme: [j for j in range(N_CH) if j not in hidden] for scheme, hidden in
            [("single", {c}), ("cluster", {c, *neighbours[c]})]}


def analyse_subject(session, subject, X, bad, neighbours, rng):
    """R² rows and the eigenvalue ratio of one subject's session."""
    n = len(X)
    is_train = np.zeros(n, dtype=bool)
    is_train[rng.permutation(n)[: n // 2]] = True
    rows, eig_ratio = [], np.nan
    for band, edges in BANDS.items():
        Xf = band_filter(X, edges)
        if band == "5-60 Hz":
            ev = np.linalg.eigvalsh(np.cov(Xf.transpose(1, 0, 2).reshape(N_CH, -1)))
            eig_ratio = ev[0] / ev[-1]
        stats = trial_stats(Xf)
        for c in range(N_CH):
            train = is_train & ~bad[:, c]
            tests = {"intact": ~is_train & ~bad[:, c], "interpolated": ~is_train & bad[:, c]}
            tests = {k: v for k, v in tests.items() if v.sum() >= MIN_TEST_TRIALS}
            if train.sum() < MIN_TEST_TRIALS or not tests:
                continue
            for scheme, visible in masking_schemes(c, neighbours).items():
                a, mu = fit_ridge(stats, train, c, visible)
                for test_set, sel in tests.items():
                    rows.append({"session": session, "subject": subject, "band": band, "scheme": scheme,
                                 "channel": CHANNEL_NAMES[c], "test_set": test_set, "n_test_trials": int(sel.sum()),
                                 "r2": evaluate(stats, sel, c, a, mu)[1]})
    return rows, eig_ratio


# ---------------------------------------------------------------- injection
def hydrocel_info():
    """mne.Info of all 65 HydroCel-65 electrodes (E1 ... E64 and the reference Cz) with the standard montage."""
    montage = mne.channels.make_standard_montage("GSN-HydroCel-65_1.0")
    info = mne.create_info(montage.ch_names, settings.RSFREQ, "eeg")
    info.set_montage(montage)
    return info, montage


def model_channel_topography(gain, info):
    """
    Average-references a 65-channel topography (as in the preprocessing), picks the 23 model channels in CHORDER_STANDARD
    order and scales it to a maximum absolute value of 1.
    """
    gain = gain - gain.mean()
    egi = {v: k for k, v in settings.EGI_TO_1020_MAP.items()}
    montage_name = lambda ch: "Cz" if egi[ch] == "EEG VREF" else "E" + egi[ch].split(" ")[-1]
    g = np.array([gain[info.ch_names.index(montage_name(ch))] for ch in settings.CHORDER_STANDARD])
    return g / np.abs(g).max()


def fsaverage_topography(labels):
    """
    Scalp topography of visual cortex on the MNE fsaverage template: every source of the given aparc labels (both hemispheres),
    oriented normal to the cortex with equal amplitude, through the 3-layer fsaverage BEM, HydroCel-65 electrodes aligned
    with the template by their fiducials (trans='fsaverage'). Folded cortex partly cancels itself, as in real EEG.
    Downloads fsaverage (~200 MB) into the MNE data folder on the first call.
    """
    fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    subjects_dir = fs_dir.parent
    info, _ = hydrocel_info()
    fwd = mne.make_forward_solution(info, trans="fsaverage", src=fs_dir / "bem" / "fsaverage-ico-5-src.fif",
                                    bem=fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif", meg=False, eeg=True, verbose=False)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    annotation = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=subjects_dir, verbose=False)
    gain, offset, columns = fwd["sol"]["data"], 0, []
    for hemi, src in zip(["lh", "rh"], fwd["src"]):
        vertices = np.concatenate([lab.vertices for lab in annotation if lab.name in {f"{name}-{hemi}" for name in labels}] or [[]])
        columns.append(offset + np.where(np.isin(src["vertno"], vertices))[0])
        offset += len(src["vertno"])
    columns = np.concatenate(columns)
    if len(columns) == 0:
        raise ValueError(f"No fsaverage source in the aparc labels {labels}")
    return model_channel_topography(gain[:, columns].sum(axis=1), info), \
        f"{', '.join(labels)} (aparc, both hemispheres, {len(columns)} sources normal to the cortex), fsaverage BEM"


def sphere_topography():
    """
    Scalp topography of a radial dipole in occipital cortex: at SOURCE_DEPTH of the head radius towards the midpoint of
    O1 (E35) and O2 (E39), 3-shell sphere model fitted to the HydroCel-65 electrodes. Needs no download.
    """
    info, montage = hydrocel_info()
    sphere = mne.make_sphere_model("auto", "auto", info, verbose=False)
    pos = montage.get_positions()["ch_pos"]
    r0 = np.asarray(sphere["r0"])
    direction = (pos["E35"] + pos["E39"]) / 2 - r0
    direction /= np.linalg.norm(direction)
    dipole = mne.Dipole(times=[0.0], pos=[r0 + SOURCE_DEPTH * sphere.radius * direction], amplitude=[1e-8],
                        ori=[direction], gof=[100.0])
    fwd, _ = mne.make_forward_dipole(dipole, sphere, info, trans=None, verbose=False)
    return model_channel_topography(fwd["sol"]["data"][:, 0], info), \
        f"radial dipole at {SOURCE_DEPTH:g} of the head radius towards O1/O2, 3-shell sphere model"


def visual_source_topography(head_model, labels):
    """Topography g of the injected source (23 model channels, max |g| = 1) and a description of it for the report."""
    return fsaverage_topography(labels) if head_model == "fsaverage" else sphere_topography()


def inject_subject(subject, X, bad, g, gains, neighbours, rng):
    """Injection rows of one subject's baseline: error and signal change per gain, band, scheme and target channel."""
    n = len(X)
    perm = rng.permutation(n)
    pseudo_baseline, pseudo_post = perm[: n // 2], perm[n // 2:]
    train = np.zeros(n, dtype=bool)
    train[pseudo_baseline] = True  # the models only need to be tested on the pseudo-post trials
    in_post = np.zeros(n, dtype=bool)
    in_post[pseudo_post] = True

    times = settings.EPOCH_WINDOW[0] + np.arange(X.shape[-1]) / settings.RSFREQ
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(times / ONSET_RAMP, 0, 1))  # 0 before onset, 1 from ONSET_RAMP s on
    window = (times >= settings.POST_TARGET_WINDOW[0] - 1e-9) & (times <= settings.POST_TARGET_WINDOW[1] + 1e-9)
    source = np.einsum("c,nct->nt", g, X) / (g @ g)  # least-squares estimate of the source's activity per trial

    rows = []
    for band, edges in BANDS.items():
        Xf = band_filter(X, edges)
        train_stats, orig_stats = trial_stats(Xf), trial_stats(Xf[:, :, window])
        injected_stats = {}
        for gain in gains:
            Xi = X.copy()
            Xi[in_post] += gain * g[None, :, None] * (source[in_post] * ramp)[:, None, :]
            injected_stats[gain] = trial_stats(band_filter(Xi, edges)[:, :, window])
        for c in range(N_CH):
            fit_sel, test_sel = train & ~bad[:, c], in_post & ~bad[:, c]
            if fit_sel.sum() < MIN_TEST_TRIALS or test_sel.sum() < MIN_TEST_TRIALS:
                continue
            for scheme, visible in masking_schemes(c, neighbours).items():
                a, mu = fit_ridge(train_stats, fit_sel, c, visible)
                mse_o, r2_o, var_o = evaluate(orig_stats, test_sel, c, a, mu)
                for gain in gains:
                    mse_i, r2_i, var_i = evaluate(injected_stats[gain], test_sel, c, a, mu)
                    rows.append({"gain": gain, "subject": subject, "band": band, "scheme": scheme, "channel": CHANNEL_NAMES[c],
                                 "n_test_trials": int(test_sel.sum()), "mse_original": mse_o, "mse_injected": mse_i,
                                 "var_original": var_o, "var_injected": var_i, "r2_original": r2_o, "r2_injected": r2_i})
    return rows


# ---------------------------------------------------------------- output
def fmt(median, q1, q3, scale=1.0, unit=""):
    return f"{scale * median:.2f}{unit} [{scale * q1:.2f}, {scale * q3:.2f}]"


def markdown_table(df, index=True):
    """Markdown table of a DataFrame (floats with 2 decimals, missing values as '–')."""
    if index:
        df = df.reset_index()
    cell = lambda v: "–" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def band_order(df, col="band"):
    return df.sort_values(col, key=lambda s: s.map(list(BANDS).index), kind="stable")


def plot_topomap_row(values_by_band, info, vlim, cmap, label, title, save_path):
    """
    One topomap per band in a row, one colour scale shared by all panels (layout of the plasticity_plotting.ipynb topomaps).
    values_by_band: {band: Series indexed by channel name}; channels without a value are left out of that panel.
    """
    fig, axes = plt.subplots(1, len(values_by_band), figsize=(24, 4))
    axes = np.atleast_1d(axes)
    im = None
    for ax, (band, values) in zip(axes, values_by_band.items()):
        vals = values.reindex(CHANNEL_NAMES).to_numpy(dtype=float)
        keep = np.where(~np.isnan(vals))[0]
        if len(keep) < 4:
            ax.set_axis_off()
            ax.set_title(f"{band}\n(too few channels)")
            continue
        im, _ = mne.viz.plot_topomap(vals[keep], mne.pick_info(info, keep), axes=ax, show=False, vlim=vlim, cmap=cmap)
        ax.set_title(band)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label=label)
    plt.suptitle(title, y=1.02)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def symmetric_limit(values_by_band, default=0.1):
    return max([np.nanmax(np.abs(v.to_numpy())) for v in values_by_band.values() if len(v)] or [default])


def run_build(root, label, neighbours, info):
    """Masked-prediction analysis of one build: writes its tables, report and topomaps, returns its R² table."""
    out_dir, plot_dir = OUTPUT_ROOT / label, PLOT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows, eig_rows, rates = [], [], {}
    io_handler = EpochIO(ext=".pt")
    for session, folder in SESSIONS.items():
        directory = Path(root) / folder
        interp = interpolated_channels(directory / "epochs_manifest.csv")
        rates[session] = interpolation_rates(interp)
        inputs = io_handler.load_all_epochs(directory / settings.INPUT_EPOCHS_DIR_NAME, marker="_input")
        for subject in sorted(inputs, key=subject_number):
            X, bad = subject_trials(inputs, subject, interp)
            rng = np.random.default_rng([settings.SEED, subject_number(subject), int(session == "baseline")])
            subject_rows, eig_ratio = analyse_subject(session, subject, X, bad, neighbours, rng)
            rows += subject_rows
            eig_rows.append({"session": session, "subject": subject, "n_trials": len(X), "ratio": eig_ratio})
            print(f"[{label}] {session} {subject}: {len(X)} trials, eigenvalue ratio {eig_ratio:.1e}")

    r2 = pd.DataFrame(rows)
    r2.to_csv(out_dir / "r2_subject_channel.csv", index=False)
    eig = pd.DataFrame(eig_rows)
    eig.to_csv(out_dir / "eigenvalues.csv", index=False)

    intact = r2[r2.test_set == "intact"]
    subj_means = intact.groupby(["session", "band", "scheme", "subject"])["r2"].mean().reset_index()
    summary = subj_means.groupby(["session", "band", "scheme"])["r2"].agg(**QUARTILES, n_subjects="count").reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    summary["text"] = [fmt(m, a, b) for m, a, b in zip(summary["median"], summary.q1, summary.q3)]
    table = summary.pivot_table(index="band", columns=["session", "scheme"], values="text", aggfunc="first")
    table = table.reindex(index=[b for b in BANDS if b in table.index])
    table.columns = [f"{session}, {scheme}" for session, scheme in table.columns]

    single = r2[r2.scheme == "single"].pivot_table(index=["session", "band", "subject", "channel"], columns="test_set", values="r2")
    single = single.dropna(subset=["intact", "interpolated"]) if "interpolated" in single else single.iloc[0:0]
    single["diff"] = single.get("interpolated", np.nan) - single["intact"]
    interp_summary = band_order(single.groupby(["session", "band"])["diff"].agg(**QUARTILES, n_pairs="count").reset_index())
    interp_summary.to_csv(out_dir / "interpolation.csv", index=False)
    interp_table = interp_summary.copy()
    interp_table["R² interpolated - intact"] = [fmt(m, a, b) for m, a, b in zip(interp_table["median"], interp_table.q1, interp_table.q3)]
    interp_table = interp_table[["session", "band", "R² interpolated - intact", "n_pairs"]]

    broad = r2[r2.band == "5-60 Hz"]
    med = lambda scheme, test_set: broad[(broad.scheme == scheme) & (broad.test_set == test_set)].groupby("channel")["r2"].median().reindex(CHANNEL_NAMES)
    per_channel = pd.DataFrame({
        "channel": CHANNEL_NAMES,
        "interp. rate baseline": rates["baseline"],
        "interp. rate post": rates["post-patching"],
        "R² single": med("single", "intact").values,
        "R² cluster": med("cluster", "intact").values,
        "R² single, interpolated trials": med("single", "interpolated").values,
    })
    per_channel.to_csv(out_dir / "per_channel.csv", index=False)

    report = [f"# Ridge-regression check of masked-channel prediction ({label})", "",
              f"Build: `{root}` ({build_description(root)}). Subjects: {r2.subject.nunique()}. Input epochs {settings.EPOCH_WINDOW[0]} to {settings.EPOCH_WINDOW[1]} s, "
              "band-passed per band, every time sample one observation, trials of all stimulus types pooled and split in half "
              f"per subject and session (fit / test). single = target channel hidden (22 visible); cluster = target and its "
              f"{N_NEIGHBOURS} nearest electrodes hidden ({N_CH - 1 - N_NEIGHBOURS} visible); only the target channel is predicted. "
              "R² over the pooled test samples of trials with an intact target channel; table values are the median [quartiles] "
              "over subjects of the mean over channels.", "",
              "## Exact linear dependency", "",
              f"Smallest / largest eigenvalue of the 23-channel covariance (5-60 Hz): median {eig['ratio'].median():.1e}, "
              f"minimum {eig['ratio'].min():.1e} over {len(eig)} subject-sessions. An exact linear dependency would give ~1e-12 or less.", "",
              "## R² of the masked channel", "", markdown_table(table), "",
              "## Interpolated vs intact target channel (single masking)", "",
              "R² of the target channel in test trials where it was interpolated minus in trials where it was intact, "
              "per subject x channel with both test sets: median [quartiles], number of pairs.", "",
              markdown_table(interp_table, index=False), "",
              "## Per channel", "",
              "Interpolation rate: share of trials in which the channel is interpolated (recording- or autoreject-level). "
              "R²: 5-60 Hz, median over subject-sessions.", "", markdown_table(per_channel, index=False), ""]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    for session in SESSIONS:
        sess = intact[intact.session == session]
        for scheme in ["single", "cluster"]:
            values = {band: sess[(sess.scheme == scheme) & (sess.band == band)].groupby("channel")["r2"].median() for band in BANDS}
            hidden = "target channel" if scheme == "single" else f"target and its {N_NEIGHBOURS} nearest electrodes"
            plot_topomap_row(values, info, (0, 1), "Reds", "R²",
                             f"Ridge R² of the masked channel, {hidden} hidden, {session}, {label} (median over N={sess.subject.nunique()} subjects)",
                             plot_dir / f"topomaps_r2_{scheme}_{session}.png")
        diffs = single.reset_index()
        values = {band: diffs[(diffs.session == session) & (diffs.band == band)].groupby("channel")["diff"].median() for band in BANDS}
        lim = symmetric_limit(values)
        plot_topomap_row(values, info, (-lim, lim), "RdBu_r", "R² interpolated − intact",
                         f"R² gain of interpolated target channels (single masking), {session}, {label} "
                         f"(median over subjects with both test sets; channels without such subjects left out)",
                         plot_dir / f"topomaps_interpolation_{session}.png")
    return r2, out_dir, plot_dir


def compare_builds(r2_main, r2_compare, label, compare_label, main_root, compare_root, out_dir, plot_dir, info):
    """Per session, subject, band, scheme and channel: R² of both builds (intact target channels) and main - compare."""
    keys = ["session", "subject", "band", "scheme", "channel"]
    main = r2_main[r2_main.test_set == "intact"].set_index(keys)["r2"].rename(f"r2_{label}")
    comp = r2_compare[r2_compare.test_set == "intact"].set_index(keys)["r2"].rename(f"r2_{compare_label}")
    both = pd.concat([main, comp], axis=1).dropna()
    both["difference"] = both[f"r2_{label}"] - both[f"r2_{compare_label}"]
    both.reset_index().to_csv(out_dir / f"comparison_{compare_label}.csv", index=False)

    subj = both.reset_index().groupby(["session", "band", "scheme", "subject"])["difference"].mean().reset_index()
    summary = subj.groupby(["session", "band", "scheme"])["difference"].agg(**QUARTILES, n_subjects="count").reset_index()
    summary["text"] = [fmt(m, a, b) for m, a, b in zip(summary["median"], summary.q1, summary.q3)]
    table = summary.pivot_table(index="band", columns=["session", "scheme"], values="text", aggfunc="first")
    table = table.reindex(index=[b for b in BANDS if b in table.index])
    table.columns = [f"{session}, {scheme}" for session, scheme in table.columns]
    section = [f"## R² of {label} minus {compare_label}", "",
               f"- {label}: `{main_root}` ({build_description(main_root)})",
               f"- {compare_label}: `{compare_root}` ({build_description(compare_root)})", "",
               f"Same analysis on both builds; per subject the mean over channels of R² ({label}) - R² ({compare_label}), "
               "intact target channels, subjects in both builds: median [quartiles]. The trial sets of the builds differ "
               "(e.g. autoreject drops and repairs different epochs, or none), so subjects, not trials, are paired.", "",
               markdown_table(table), ""]

    flat = both.reset_index()
    for session in SESSIONS:
        for scheme in ["single", "cluster"]:
            sel = flat[(flat.session == session) & (flat.scheme == scheme)]
            values = {band: sel[sel.band == band].groupby("channel")["difference"].median() for band in BANDS}
            lim = symmetric_limit(values)
            plot_topomap_row(values, info, (-lim, lim), "RdBu_r", f"R² {label} − {compare_label}",
                             f"R² difference {label} − {compare_label}, {scheme} masking, {session} (median over N={sel.subject.nunique()} subjects)",
                             plot_dir / f"topomaps_r2_difference_{compare_label}_{scheme}_{session}.png")
    return section


def injection_test(root, gains, head_model, labels, neighbours, info, out_dir, plot_dir):
    """Gain injection on the baseline session of the build: writes injection.csv and topomaps, returns the report section."""
    g, source_description = visual_source_topography(head_model, labels)
    roi = [CHANNEL_NAMES[c] for c in np.argsort(-np.abs(g))[:N_SOURCE_ROI]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im, _ = mne.viz.plot_topomap(g, info, axes=ax, show=False, vlim=(-1, 1), cmap="RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8, label="relative potential")
    ax.set_title("Injected visual-cortex source\n" + textwrap.fill(source_description, 45), pad=10)
    fig.savefig(plot_dir / "injected_source_topography.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    directory = Path(root) / SESSIONS["baseline"]
    interp = interpolated_channels(directory / "epochs_manifest.csv")
    inputs = EpochIO(ext=".pt").load_all_epochs(directory / settings.INPUT_EPOCHS_DIR_NAME, marker="_input")
    rows = []
    for subject in sorted(inputs, key=subject_number):
        X, bad = subject_trials(inputs, subject, interp)
        rng = np.random.default_rng([settings.SEED, subject_number(subject), 2])
        rows += inject_subject(subject, X, bad, g, gains, neighbours, rng)
        print(f"[injection] {subject}: done")
    inj = pd.DataFrame(rows)
    # a flat target channel (e.g. disconnected, in a build without interpolation) has no defined change
    flat = (inj.mse_original <= 1e-12) | (inj.var_original <= 1e-12)
    inj["error_change"] = np.where(flat, np.nan, np.log(inj.mse_injected / inj.mse_original.where(~flat, 1.0)))
    inj["signal_change"] = np.where(flat, np.nan, np.log(inj.var_injected / inj.var_original.where(~flat, 1.0)))
    inj.to_csv(out_dir / "injection.csv", index=False)

    pct = lambda log_ratio: 100 * (np.exp(log_ratio) - 1)
    table_rows = []
    for gain in gains:
        for band in BANDS:
            sel = inj[(inj.gain == gain) & (inj.band == band) & inj.channel.isin(roi)]
            row = {"gain": f"{100 * gain:+g} %", "band": band}
            signal = sel[sel.scheme == "single"].groupby("subject")["signal_change"].mean().agg(**QUARTILES)
            row[f"signal power change ({', '.join(roi)})"] = fmt(pct(signal["median"]), pct(signal.q1), pct(signal.q3), unit=" %")
            for scheme in ["single", "cluster"]:
                err = sel[sel.scheme == scheme].groupby("subject")["error_change"].mean().agg(**QUARTILES)
                row[f"error change, {scheme}"] = fmt(pct(err["median"]), pct(err.q1), pct(err.q3), unit=" %")
            table_rows.append(row)

    for gain in gains:
        sel = inj[inj.gain == gain]
        n_subjects = sel.subject.nunique()
        for kind, scheme in [("signal", "single"), ("error", "single"), ("error", "cluster")]:
            part = sel[sel.scheme == scheme]
            values = {band: pct(part[part.band == band].groupby("channel")[f"{kind}_change"].median()) for band in BANDS}
            lim = symmetric_limit(values, default=1.0)
            what = "Target channel power change" if kind == "signal" else f"Masked prediction error change ({scheme} masking)"
            plot_topomap_row(values, info, (-lim, lim), "RdBu_r", "change (%)",
                             f"{what} after a {100 * gain:+g} % visual-cortex source gain (median over N={n_subjects} subjects)",
                             plot_dir / f"topomaps_injection_{kind}{'' if kind == 'signal' else '_' + scheme}_gain{gain:g}.png")

    return ["## Gain injection", "",
            f"Pseudo-sessions from the baseline: in half of every subject's trials, the activity of a visual-cortex source "
            f"({source_description}; average reference over the 65 HydroCel electrodes, then the 23 model channels) "
            f"was made 1 + gain times stronger after stimulus onset. The models (fitted on the other half, the pseudo-baseline) were scored on the same "
            f"trials with and without the injection, {settings.POST_TARGET_WINDOW[0]} to {settings.POST_TARGET_WINDOW[1]} s. "
            f"Values: per subject the mean over {', '.join(roi)} (where the source is strongest), median [quartiles] over "
            "subjects. Signal power change = variance of the target channel; error change = MSE of its masked prediction. "
            "An error change much smaller than the signal change means the masked prediction absorbs the gain change.", "",
            markdown_table(pd.DataFrame(table_rows), index=False), ""]


def main(opts):
    info = create_topomap_info()  # 10-20 labels at the positions of the recording HydroCel electrodes
    neighbours = nearest_electrodes(info, N_NEIGHBOURS)
    print("Nearest electrodes: " + "; ".join(f"{CHANNEL_NAMES[c]}: {', '.join(CHANNEL_NAMES[j] for j in nb)}" for c, nb in neighbours.items()))

    r2, out_dir, plot_dir = run_build(opts.data_root, opts.label, neighbours, info)
    extra = []
    if opts.compare_root:
        r2_compare, _, _ = run_build(opts.compare_root, opts.compare_label, neighbours, info)
        extra += compare_builds(r2, r2_compare, opts.label, opts.compare_label, opts.data_root, opts.compare_root, out_dir, plot_dir, info)
    if opts.inject:
        extra += injection_test(opts.data_root, opts.inject, opts.head_model, opts.inject_labels, neighbours, info, out_dir, plot_dir)
    if extra:
        with open(out_dir / "report.md", "a", encoding="utf-8") as f:
            f.write("\n".join(extra))

    print(f"\nTables in {out_dir}, topomaps in {plot_dir}")
    print((out_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ridge-regression check of masked-channel prediction")
    parser.add_argument("--data_root", default=str(settings.PROCESS_DIR), help="make_dataset.py build: folder with pre_patching/ and post_patching/")
    parser.add_argument("--label", default="dataset", help="output subfolder of the build")
    parser.add_argument("--compare_root", default=None, help="second build analysed the same way and compared, e.g. data/processed_raw")
    parser.add_argument("--compare_label", default="raw", help="output subfolder of the second build")
    parser.add_argument("--inject", type=float, nargs="+", default=None,
                        help="gain injection test on the baseline, e.g. --inject 0.1 0.25 0.5 (visual source 10/25/50 %% stronger after onset)")
    parser.add_argument("--head_model", choices=["fsaverage", "sphere"], default="fsaverage",
                        help="head model of the injected source: fsaverage template (downloads ~200 MB on first use) or an offline 3-shell sphere")
    parser.add_argument("--inject_labels", nargs="+", default=FSAVERAGE_LABELS,
                        help="aparc labels of the fsaverage source, e.g. pericalcarine lateraloccipital")
    main(parser.parse_args())
