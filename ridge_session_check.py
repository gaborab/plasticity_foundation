"""
Session difference of the masked-channel R² from ridge_leakage.py, and whether it follows the extra cleaning of a session.
Reads existing outputs only, nothing is recomputed:
    outputs/ridge_leakage/<label>/r2_subject_channel.csv   (ridge_leakage.py)
    <data_root>/{pre,post}_patching/epochs_manifest.csv     (make_dataset.py: interpolated channels per trial)
    <qc_dir>/ica_report.csv                                 (make_dataset.py: ICA components removed per recording; skipped if missing)

Per band and masking scheme, with the subject as the unit (intact target channels, mean over channels per subject and session):
    1. paired t-test of the session difference: dR² = R²(post-patching) - R²(baseline); mean, d_z
    2. dose-response: Spearman correlation over subjects of dR² with
           d interpolation = interpolated share of trial x channel pairs (23 model channels), post-patching - baseline
           d ICA           = ICA components removed per recording, post-patching - baseline
       If the extra cleaning of a session raised its R², subjects with more extra cleaning would show a larger dR².

Outputs (outputs/ridge_leakage/<label>/): session_check.csv (one row per band, scheme and test), session_check.md.

Run from the repository root, e.g.
    python ridge_session_check.py --label dataset
    python ridge_session_check.py --label raw --data_root data/processed_raw
"""
import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from scipy import stats

from core import settings

OUTPUT_ROOT = settings.BASE_DIR / "outputs" / "ridge_leakage"
SESSIONS = {"baseline": settings.FINETUNE_DIR.name, "post-patching": settings.PATCH_DIR.name}
BANDS = ["5-60 Hz", *settings.FREQ_BANDS]
N_CH = len(settings.CHORDER_STANDARD)


def subject_session_r2(r2_path):
    """Mean R² over channels per session, subject, band and scheme (intact target channels)."""
    r2 = pd.read_csv(r2_path)
    r2 = r2[r2.test_set == "intact"]
    return r2.groupby(["band", "scheme", "subject", "session"])["r2"].mean().unstack("session")


def interpolation_share(data_root):
    """Interpolated share of trial x channel pairs per subject and session (recording- or autoreject-level)."""
    rows = []
    for session, folder in SESSIONS.items():
        manifest = pd.read_csv(Path(data_root) / folder / "epochs_manifest.csv",
                               dtype={"rec_interpolated": str, "ar_interpolated": str}).fillna("")
        n_interp = [len({c for c in f"{rec};{ar}".split(";") if c}) for rec, ar in zip(manifest.rec_interpolated, manifest.ar_interpolated)]
        rows.append(pd.DataFrame({"subject": manifest.subject, "session": session, "share": np.array(n_interp) / N_CH}))
    return pd.concat(rows).groupby(["subject", "session"])["share"].mean().unstack("session")


def ica_removals(qc_dir):
    """ICA components removed per recording, averaged per subject and session; None if the build has no ICA report."""
    path = Path(qc_dir) / "ica_report.csv"
    if not path.exists():
        return None
    ica = pd.read_csv(path)
    # data\raw\STMD_data\<subject>\<session>\<recording folder>\<file>.edf
    parts = ica["recording"].map(lambda p: PureWindowsPath(p).parts)
    ica["subject"], ica["session"] = parts.str[-4], parts.str[-3].str.lower()
    per_recording = ica.groupby(["subject", "session", "recording"])["excluded"].sum().reset_index()
    return per_recording.groupby(["subject", "session"])["excluded"].mean().unstack("session")


def markdown_table(df):
    cell = lambda v: "–" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(opts):
    out_dir = OUTPUT_ROOT / opts.label
    qc_dir = Path(opts.qc_dir) if opts.qc_dir else (settings.QC_DIR if Path(opts.data_root).resolve() == settings.PROCESS_DIR.resolve()
                                                    else Path(opts.data_root) / "qc")
    r2 = subject_session_r2(out_dir / "r2_subject_channel.csv")
    covariates = {"interpolation": interpolation_share(opts.data_root)}
    ica = ica_removals(qc_dir)
    if ica is not None:
        covariates["ICA"] = ica
    else:
        print(f"No ica_report.csv in {qc_dir}: the ICA dose-response is skipped")
    deltas = {name: (cov["post-patching"] - cov["baseline"]).rename(f"d_{name}") for name, cov in covariates.items()}

    rows = []
    for band in BANDS:
        for scheme in ["single", "cluster"]:
            if (band, scheme) not in r2.index.droplevel("subject").unique():
                continue
            d_r2 = (r2.loc[(band, scheme), "post-patching"] - r2.loc[(band, scheme), "baseline"]).dropna()
            t, p_t = stats.ttest_1samp(d_r2, 0)
            row = {"band": band, "scheme": scheme, "n_subjects": len(d_r2),
                   "R² baseline": r2.loc[(band, scheme), "baseline"].median(), "R² post": r2.loc[(band, scheme), "post-patching"].median(),
                   "mean dR²": d_r2.mean(), "d_z": d_r2.mean() / d_r2.std(ddof=1), "t": t, "p t-test": p_t}
            for name, delta in deltas.items():
                both = pd.concat([d_r2.rename("d_r2"), delta], axis=1).dropna()
                if len(both) < len(d_r2):
                    print(f"WARNING: {band}, {scheme}: only {len(both)} of {len(d_r2)} subjects have the {name} difference")
                rho, p_rho = stats.spearmanr(both["d_r2"], both[delta.name])
                row[f"rho dR² vs d {name}"], row[f"p rho {name}"] = rho, p_rho
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "session_check.csv", index=False)
    summary = {name: f"median {delta.median():+.3f} (post - baseline, {len(delta.dropna())} subjects; "
                     f"{(delta > 0).sum()} with more in post-patching)" for name, delta in deltas.items()}
    report = [f"# Session difference of the masked-channel R² ({opts.label})", "",
              f"dR² = R²(post-patching) - R²(baseline) per subject (mean over channels, intact target channels). "
              "Test: paired t-test on dR², d_z = mean / SD. Dose-response: Spearman rank correlation (rho) "
              "over subjects of dR² with the session difference of the cleaning.", "",
              *[f"- d {name}: {text}" for name, text in summary.items()],
              "  (interpolation = interpolated share of trial x channel pairs; ICA = components removed per recording)", "",
              markdown_table(result), ""]
    (out_dir / "session_check.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Session difference of the ridge R² and its dose-response to the extra cleaning")
    parser.add_argument("--label", default="dataset", help="output subfolder of the ridge_leakage.py run")
    parser.add_argument("--data_root", default=str(settings.PROCESS_DIR), help="make_dataset.py build the ridge run analysed")
    parser.add_argument("--qc_dir", default=None, help="QC folder of the build (default: outputs/qc for data/processed, else <data_root>/qc)")
    main(parser.parse_args())
