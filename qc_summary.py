"""
Short summary of the preprocessing QC: automatic bad channel detection, ICA (ICLabel) and autoreject.

Reads the reports in outputs/qc/ and the epochs_manifest.csv files, and writes
    outputs/qc/qc_summary.csv          one row per metric, columns Baseline / Post-patching / All
    outputs/qc/qc_worst_channels.csv   per model channel: how often it was interpolated, and by which step
    outputs/qc/qc_subjects.csv         per subject and session (+ 'All'): dropped epochs, interpolated channels, ICA removals,
                                       and for 'All' a rank score and the metrics in which the subject is an outlier
and prints them.
"""
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from core import settings

ICLABEL_CLASSES = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']
SESSIONS = ['Baseline', 'Post-patching']


def split_recording_path(df: pd.DataFrame) -> pd.DataFrame:
    """Adds subject / session / rec columns from 'data\\raw\\STMD_data\\<subject>\\<session>\\<rec>\\<file>.edf'."""
    parts = df['recording'].map(lambda p: PureWindowsPath(p).parts)
    return df.assign(subject=parts.str[-4], session=parts.str[-3], rec=parts.str[-2])


def channel_label(egi: str, egi_to_1020: dict) -> str:
    """'EEG 52' -> 'T4 (E52)', non-model channels -> 'E61'."""
    short = 'Cz' if egi == 'EEG VREF' else 'E' + egi.split(' ')[1]
    return f"{egi_to_1020[egi].split(' ')[1].replace('-REF', '')} ({short})" if egi in egi_to_1020 else short


def per_session(rows: dict, frame: pd.DataFrame, fn) -> dict:
    """Applies fn to the Baseline, Post-patching and full frame."""
    return {**{s: fn(frame[frame['session'] == s]) for s in SESSIONS}, 'All': fn(frame), **rows}


def top_items(series: pd.Series, n: int = 5) -> str:
    return ', '.join(f'{k} {v}' for k, v in series.head(n).items())


def summarize_bad_channels(path: Path, egi_to_1020: dict):
    bc = split_recording_path(pd.read_csv(path))
    bc['flagged'] = bc['reasons'].fillna('') != ''
    bc['model'] = bc['channel'].isin(egi_to_1020)
    per_rec = bc.groupby(['subject', 'session', 'rec']).agg(n=('flagged', 'sum'), n_model=('flagged', lambda f: (f & bc.loc[f.index, 'model']).sum()))
    per_rec = per_rec.reset_index()
    flagged = bc[bc['flagged']]
    worst = flagged['channel'].map(lambda c: channel_label(c, egi_to_1020)).value_counts()
    rows = [
        ('Bad channels', 'recordings', lambda d: d['rec'].size),
        ('Bad channels', 'channel checks flagged (%)', None),
        ('Bad channels', 'recordings with >= 1 flagged channel (%)', lambda d: 100 * (d['n'] > 0).mean()),
        ('Bad channels', 'flagged channels per recording (mean / max)', lambda d: f"{d['n'].mean():.2f} / {d['n'].max()}"),
        ('Bad channels', 'flagged model channels per recording (mean)', lambda d: d['n_model'].mean()),
    ]
    out = []
    for section, metric, fn in rows:
        if fn is None:
            out.append(per_session({'section': section, 'metric': metric}, bc, lambda d: 100 * d['flagged'].mean()))
        else:
            out.append(per_session({'section': section, 'metric': metric}, per_rec, fn))
    out.append(per_session({'section': 'Bad channels', 'metric': 'flags by reason'}, flagged,
                           lambda d: top_items(d['reasons'].value_counts())))
    out.append(per_session({'section': 'Bad channels', 'metric': 'most often flagged channels'}, flagged,
                           lambda d: top_items(d['channel'].map(lambda c: channel_label(c, egi_to_1020)).value_counts())))
    return out, bc


def summarize_ica(path: Path):
    ica = split_recording_path(pd.read_csv(path))
    ica['label'] = ica[ICLABEL_CLASSES].idxmax(axis=1)
    per_rec = ica.groupby(['subject', 'session', 'rec'])['excluded'].sum().rename('n').reset_index()
    removed = ica[ica['excluded']]
    rows = [
        per_session({'section': 'ICA', 'metric': 'recordings'}, per_rec, lambda d: len(d)),
        per_session({'section': 'ICA', 'metric': 'components removed per recording (mean / median / max)'}, per_rec,
                    lambda d: f"{d['n'].mean():.2f} / {d['n'].median():.0f} / {d['n'].max()}"),
        per_session({'section': 'ICA', 'metric': 'recordings with 0 components removed (%)'}, per_rec, lambda d: 100 * (d['n'] == 0).mean()),
        per_session({'section': 'ICA', 'metric': 'recordings with >= 4 components removed (%)'}, per_rec, lambda d: 100 * (d['n'] >= 4).mean()),
        per_session({'section': 'ICA', 'metric': 'removed components by class'}, removed, lambda d: top_items(d['label'].value_counts(), 7)),
    ]
    return rows


def model_channel_name(idx: str, chorder: list) -> str:
    return chorder[int(idx)].split(' ')[1].replace('-REF', '')


def summarize_autoreject(report_path: Path, manifests: dict, chorder: list):
    ar = pd.read_csv(report_path).rename(columns={'subject_name': 'subject', 'conditions': 'session'})
    ar['pct_dropped'] = 100 * ar['n_dropped'] / ar['n_epochs']
    worst_session = lambda d: f"{d.loc[d['pct_dropped'].idxmax(), 'subject']} ({d['pct_dropped'].max():.1f}%)" if len(d) else ''
    rows = [
        per_session({'section': 'Autoreject', 'metric': 'sessions'}, ar, lambda d: len(d)),
        per_session({'section': 'Autoreject', 'metric': 'epochs dropped (% of all)'}, ar, lambda d: 100 * d['n_dropped'].sum() / d['n_epochs'].sum()),
        per_session({'section': 'Autoreject', 'metric': 'epochs dropped per session (median / max %)'}, ar,
                    lambda d: f"{d['pct_dropped'].median():.1f} / {d['pct_dropped'].max():.1f}"),
        per_session({'section': 'Autoreject', 'metric': 'session with most drops'}, ar, worst_session),
        per_session({'section': 'Autoreject', 'metric': 'sessions with > 20% dropped'}, ar, lambda d: int((d['pct_dropped'] > 20).sum())),
        per_session({'section': 'Autoreject', 'metric': 'interpolated channels per kept epoch, all channels (mean)'}, ar,
                    lambda d: (d['mean_interp_per_kept_epoch'] * (d['n_epochs'] - d['n_dropped'])).sum() / (d['n_epochs'] - d['n_dropped']).sum()),
    ]

    frames = []
    for session, path in manifests.items():
        if path.exists():
            frames.append(pd.read_csv(path, dtype={'rec_interpolated': str, 'ar_interpolated': str}).assign(session=session))
        else:
            print(f'WARNING: {path} not found, skipping model-channel autoreject metrics for {session}')
    if not frames:
        return rows, pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    split = lambda s: [c for c in str(s).split(';') if c not in ('', 'nan')]
    m['ar_list'] = m['ar_interpolated'].map(split)
    m['masked_list'] = [sorted(set(split(r)) | set(split(a))) for r, a in zip(m['rec_interpolated'], m['ar_interpolated'])]
    rows += [
        per_session({'section': 'Autoreject', 'metric': 'kept epochs (saved to disk)'}, m, lambda d: len(d)),
        per_session({'section': 'Autoreject', 'metric': 'model channels interpolated per kept epoch (mean)'}, m, lambda d: d['ar_list'].map(len).mean()),
        per_session({'section': 'Autoreject', 'metric': 'kept epochs with >= 1 model channel interpolated (%)'}, m, lambda d: 100 * (d['ar_list'].map(len) > 0).mean()),
        per_session({'section': 'Masking', 'metric': 'model channels masked per epoch, any reason (mean / max)'}, m,
                    lambda d: f"{d['masked_list'].map(len).mean():.2f} / {d['masked_list'].map(len).max()}"),
        per_session({'section': 'Autoreject', 'metric': 'most often interpolated model channels (% of kept epochs)'}, m,
                    lambda d: ', '.join(f'{model_channel_name(c, chorder)} {100 * n / len(d):.0f}%'
                                        for c, n in d['ar_list'].explode().dropna().value_counts().head(5).items())),
    ]
    return rows, m


def worst_channels_table(bc: pd.DataFrame, m: pd.DataFrame, egi_to_1020: dict, chorder: list) -> pd.DataFrame:
    """Per model channel: share of recordings flagged by the detector, share of kept epochs interpolated by autoreject."""
    name_of = {egi: v.split(' ')[1].replace('-REF', '') for egi, v in egi_to_1020.items()}
    det = (bc[bc['model']].assign(name=lambda d: d['channel'].map(name_of))
           .groupby(['name', 'session'])['flagged'].mean().mul(100).unstack('session'))
    det.columns = [f'detector_recordings_flagged_%_{c}' for c in det.columns]
    table = det
    if not m.empty:
        ar = (m.explode('ar_list').dropna(subset=['ar_list']).assign(name=lambda d: d['ar_list'].map(lambda c: model_channel_name(c, chorder)))
              .groupby(['name', 'session']).size().unstack('session'))
        ar = ar.div(m.groupby('session').size(), axis=1).mul(100)
        ar.columns = [f'autoreject_epochs_interpolated_%_{c}' for c in ar.columns]
        table = det.join(ar, how='outer')
    table = table.reindex([n.split(' ')[1].replace('-REF', '') for n in chorder]).fillna(0)
    sort_col = [c for c in table.columns if c.startswith('autoreject')] or list(table.columns)
    return table.assign(_sort=table[sort_col].mean(axis=1)).sort_values('_sort', ascending=False).drop(columns='_sort').round(1)


SUBJECT_RANK_METRICS = ['autoreject dropped (%)', 'masked trial x model channel pairs (%)', 'ICA components removed per recording']


def tukey_outliers(values: pd.Series) -> pd.Series:
    """True where a value lies above Q3 + 1.5 IQR of the series (high = worse for every subject metric)."""
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    return values > q3 + 1.5 * (q3 - q1)


def subject_table(qc: Path, manifest: pd.DataFrame, bc: pd.DataFrame) -> pd.DataFrame:
    """
    One row per subject and session plus an 'All' row per subject: epochs dropped by autoreject (and by the fast-press exclusion
    if fast_press_report.csv exists), interpolated model channels (whole recording, autoreject per epoch, any), automatic
    detector flags, ICA components removed per recording and their ICLabel classes.
    The 'All' rows also get a rank score (mean percentile over subjects of SUBJECT_RANK_METRICS, 1 = worst) and the metrics in
    which the subject is an outlier (above Q3 + 1.5 IQR of all subjects).
    manifest: the concatenated epochs manifests with 'session', 'ar_list' and 'masked_list' (see summarize_autoreject).
    bc: the automatic bad channel report with subject / session / rec / flagged (see summarize_bad_channels).
    """
    ar = pd.read_csv(qc / 'autoreject_report.csv').rename(columns={'subject_name': 'subject', 'conditions': 'session'})
    ica = split_recording_path(pd.read_csv(qc / 'ica_report.csv'))
    ica['label'] = ica[ICLABEL_CLASSES].idxmax(axis=1)
    fast_press_path = qc / 'fast_press_report.csv'
    fast_press = pd.read_csv(fast_press_path) if fast_press_path.exists() else None

    def metrics(subject, session):
        sel = lambda d: d[(d['subject'] == subject) & (d['session'] == session if session != 'All' else True)]
        a, m, i, b = sel(ar), sel(manifest), sel(ica), sel(bc)
        removed = i[i['excluded']]
        n_recordings = i['rec'].nunique()
        row = {
            'subject': subject, 'session': session,
            'epochs before autoreject': a['n_epochs'].sum(),
            'autoreject dropped': a['n_dropped'].sum(),
            'autoreject dropped (%)': 100 * a['n_dropped'].sum() / a['n_epochs'].sum() if a['n_epochs'].sum() else np.nan,
        }
        if fast_press is not None:
            row['fast-press dropped'] = sel(fast_press)['n_dropped'].sum()
        rec_interp = m.drop_duplicates('recording')['rec_interpolated'].map(lambda s: len([c for c in str(s).split(';') if c not in ('', 'nan')]))
        row.update({
            'kept epochs': len(m),
            'model channels interpolated per recording': rec_interp.mean(),
            'detector-flagged channels per recording (all 65)': b.groupby('rec')['flagged'].sum().mean(),
            'autoreject repaired model channels per kept epoch': m['ar_list'].map(len).mean(),
            'autoreject repaired channels per kept epoch (all 65)': ((a['mean_interp_per_kept_epoch'] * (a['n_epochs'] - a['n_dropped'])).sum()
                                                                     / (a['n_epochs'] - a['n_dropped']).sum()),
            'masked trial x model channel pairs (%)': 100 * m['masked_list'].map(len).mean() / len(settings.CHORDER_STANDARD),
            'ICA components removed per recording': len(removed) / n_recordings if n_recordings else np.nan,
            'ICA removed: eye': (removed['label'] == 'eye blink').sum(),
            'ICA removed: muscle': (removed['label'] == 'muscle artifact').sum(),
            'ICA removed: other classes': (~removed['label'].isin(['eye blink', 'muscle artifact'])).sum(),
        })
        return row

    subjects = sorted(manifest['subject'].unique(), key=lambda s: int(s.removeprefix('STMD')))
    table = pd.DataFrame([metrics(s, session) for s in subjects for session in SESSIONS + ['All']])

    # the reports and manifests must come from the same make_dataset.py build: kept = before autoreject - dropped
    sessions = table[table['session'] != 'All']
    expected = sessions['epochs before autoreject'] - sessions['autoreject dropped'] - sessions.get('fast-press dropped', 0)
    mismatch = sessions[(sessions['epochs before autoreject'] > 0) & (expected != sessions['kept epochs'])]
    missing = sessions[sessions['epochs before autoreject'] == 0]['subject'].unique()
    if len(missing):
        print(f'WARNING: {len(missing)} subjects of the manifests are missing from autoreject_report.csv '
              f'(reports from another make_dataset.py build?): {", ".join(missing)}')
    if len(mismatch):
        print(f'WARNING: kept epochs != epochs before autoreject - dropped for {mismatch["subject"].nunique()} subjects '
              f'(e.g. {", ".join(mismatch["subject"].unique()[:5])}): the reports and the manifests come from different builds')

    overall = table[table['session'] == 'All'].set_index('subject')
    complete = overall[SUBJECT_RANK_METRICS].notna().all(axis=1)
    score = overall.loc[complete, SUBJECT_RANK_METRICS].rank(pct=True).mean(axis=1).reindex(overall.index)
    outliers = pd.concat({metric: tukey_outliers(overall.loc[complete, metric]) for metric in SUBJECT_RANK_METRICS}, axis=1)
    outlier_text = outliers.apply(lambda row: '; '.join(m for m, out in row.items() if out), axis=1).reindex(overall.index)
    outlier_text[~complete] = 'missing QC data: ' + overall.loc[~complete, SUBJECT_RANK_METRICS].isna().apply(
        lambda row: ', '.join(m for m, na in row.items() if na), axis=1)
    overall_info = pd.DataFrame({'rank score (1 = worst)': score, 'outlier in': outlier_text.fillna('')})
    table = table.merge(overall_info, left_on='subject', right_index=True, how='left')
    table.loc[table['session'] != 'All', ['rank score (1 = worst)', 'outlier in']] = [np.nan, '']
    return table


def main():
    qc = settings.QC_DIR
    bc_rows, bc = summarize_bad_channels(qc / 'auto_bad_chans.csv', settings.EGI_TO_1020_MAP)
    ica_rows = summarize_ica(qc / 'ica_report.csv')
    ar_rows, manifest = summarize_autoreject(qc / 'autoreject_report.csv',
                                             {'Baseline': settings.FINETUNE_DIR / 'epochs_manifest.csv',
                                              'Post-patching': settings.PATCH_DIR / 'epochs_manifest.csv'},
                                             settings.CHORDER_STANDARD)

    summary = pd.DataFrame(bc_rows + ica_rows + ar_rows)[['section', 'metric'] + SESSIONS + ['All']]
    summary[SESSIONS + ['All']] = summary[SESSIONS + ['All']].applymap(  # pandas 1.5: applymap (renamed to map in 2.1)
        lambda v: round(v, 2) if isinstance(v, (float, np.floating)) else v)
    worst = worst_channels_table(bc, manifest, settings.EGI_TO_1020_MAP, settings.CHORDER_STANDARD)

    subjects = subject_table(qc, manifest, bc)

    summary.to_csv(qc / 'qc_summary.csv', index=False)
    worst.to_csv(qc / 'qc_worst_channels.csv')
    subjects.round(2).to_csv(qc / 'qc_subjects.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 90, 'display.max_columns', 30):
        print(summary.to_string(index=False))
        print('\nModel channels, most often interpolated first:')
        print(worst.head(10).to_string())
        print('\nSubjects, worst overall first (session "All"):')
        overall = subjects[subjects['session'] == 'All'].sort_values('rank score (1 = worst)', ascending=False, na_position='last')
        print(overall[['subject', 'kept epochs', *SUBJECT_RANK_METRICS, 'rank score (1 = worst)', 'outlier in']].head(10).round(2).to_string(index=False))
    print(f'\nSaved {qc / "qc_summary.csv"}, {qc / "qc_worst_channels.csv"} and {qc / "qc_subjects.csv"}')


if __name__ == '__main__':
    main()
