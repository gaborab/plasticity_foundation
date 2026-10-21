import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mne
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact
from pathlib import Path
from typing import List, Dict, Any, Optional
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

import core.io
from core import settings, io_utils
from core.io_models import Trial
from core.plot_data_processing import order_to_channel_viewer_format
from core.utils import cut_to_timewindow


def correlation_plot(x: Dict[str, List[float]], y: Dict[str, List[float]], xlabel: str, ylabel: str, title: str,
                     save_dir: Path, save_name: str):
    if not len(x) == len(y):
        raise ValueError("x and y must have the same length.")
    n_bands = len(x)

    fig, ax = plt.subplots(n_bands, 1, figsize=(10, 5 * n_bands), sharex=True, sharey=True)
    ax = np.atleast_1d(ax)  # Ensure ax is always an array for consistent indexing

    for i, band in enumerate(x.keys()):
        ax[i].scatter(x[band], y[band], alpha=0.6, color='steelblue', edgecolor='black', zorder=2)
        ax[i].set_xlabel(xlabel, fontsize=12)
        ax[i].set_ylabel(ylabel, fontsize=12)
        ax[i].yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax[i].set_axisbelow(True)
        ax[i].set_title(f'Correlation for {band}', fontsize=14, fontweight='bold')
        m, b = np.polyfit(x[band], y[band], 1)
        ax[i].plot(x[band], m * np.array(x[band]) + b, color='black', linestyle='--')
        corr, p_val = stats.pearsonr(x[band], y[band])
        ax[i].text(0.95, 0.05, f"r: {corr:.4f}, m: {m:.4f}", transform=ax[i].transAxes, ha='right', va='top',
                   fontsize=10, )

    plt.suptitle(title, y=1.02)
    save_button(fig, save_dir=save_dir,
                default_name=save_name)
    plt.tight_layout()
    plt.show()


def session_correlation_plot(x_by_session: Dict[str, Dict[str, List[float]]], y_by_metric: Dict[str, Dict[str, Dict[str, List[float]]]],
                             xlabel: str, title: str, save_dir: Path, save_name: str, sharey: bool | str = 'row', point_size: float = 8, point_alpha: float = 0.35):
    """
    Scatter plots with one row per band and one column per metric, e.g. MAE before and corrected MAE after the volatility correction.
    Every session gets its own color and its own fitted line, the legend shows the slope (m) and Pearson r per session.
    x_by_session: {session: {band: x values}}
    y_by_metric: {metric label: {session: {band: y values}}}, with the same sessions, bands and value order as x_by_session.
    sharey: passed to plt.subplots, use False when the metrics have different units.
    point_size, point_alpha: marker size and opacity, e.g. larger and more opaque for one point per subject than for one point per trial.
    """
    sessions = list(x_by_session.keys())
    bands = list(x_by_session[sessions[0]].keys())
    metrics = list(y_by_metric.keys())
    colors = ['steelblue', 'salmon', 'lightgreen', 'plum']

    fig, axes = plt.subplots(len(bands), len(metrics), figsize=(6 * len(metrics), 3.6 * len(bands)), sharex='row', sharey=sharey, squeeze=False)
    for row, band in enumerate(bands):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for i, session in enumerate(sessions):
                x = np.array(x_by_session[session][band])
                y = np.array(y_by_metric[metric][session][band])
                if len(x) != len(y):
                    raise ValueError(f"x and y of {session}, {metric}, {band} have different lengths.")
                m, b = np.polyfit(x, y, 1)
                r = stats.pearsonr(x, y)[0]
                ax.scatter(x, y, s=point_size, alpha=point_alpha, color=colors[i], zorder=2)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, m * xs + b, color=colors[i], linewidth=2, label=f"{session}: m={m:.3f}, r={r:.2f}", zorder=3)
            ax.set_title(f"{metric}: {band}", fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
            ax.legend(fontsize=9, loc='upper left')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save_button(fig, save_dir=save_dir, default_name=save_name)
    plt.show()


def distribution_viewer(pre_nmaes: Dict[str, List[float]], post_nmaes: Dict[str, List[float]], title: str,
                        save_dir: Path, save_name: str):
    if not len(pre_nmaes) == len(post_nmaes):
        raise ValueError("pre_nmaes and post_nmaes must have the same length.")
    n_bands = len(pre_nmaes)

    fig, ax = plt.subplots(n_bands, 1, figsize=(10, 5 * n_bands), sharex=True)
    ax = np.atleast_1d(ax)  # Ensure ax is always an array for consistent indexing

    bins = 40

    for i, band in enumerate(pre_nmaes.keys()):
        mean_pre = np.mean(pre_nmaes[band])
        mean_post = np.mean(post_nmaes[band])
        band_pre = pre_nmaes[band]
        band_post = post_nmaes[band]

        ax[i].hist(band_pre, bins=bins, alpha=0.6, color='steelblue',
                   edgecolor='black', label='Baseline', zorder=2)
        ax[i].hist(band_post, bins=bins, alpha=0.6, color='darkorange',
                   edgecolor='black', label='Post-Patch', zorder=2)

        ax[i].axvline(mean_pre, color='red', linestyle='--', linewidth=2.5,
                      label=f'Baseline Mean: {mean_pre:.4f}', zorder=3)
        ax[i].axvline(mean_post, color='darkred', linestyle='-', linewidth=2.5,
                      label=f'Post-Patch Mean: {mean_post:.4f}', zorder=3)

        ax[i].set_xlabel('Normalized MAE (NMAE)', fontsize=12)
        ax[i].set_ylabel('# Trials', fontsize=12)
        ax[i].yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax[i].set_axisbelow(True)
        ax[i].legend(fontsize=10, loc='upper right')
        ax[i].set_title(f'Distribution of NMAE for {band}', fontsize=14, fontweight='bold')

    plt.suptitle(title, y=1.02)
    save_button(fig, save_dir=save_dir,
                default_name=save_name)
    plt.tight_layout()
    plt.show()


def single_channel_viewer(dataset_data: List[List[Trial]], dataset_metrics: List[List[Dict[str, Any]]], n_chans,
                          styling_params, save_dir: Path, freqs, an_suffix: str, excl_suffix: str, bands=None):
    """
    Interactive viewer for visualizing single-channel reconstructions across multiple datasets.
    Allows users to select a sample and channel, and view the corresponding reconstruction and metrics.
    Provides a button to save the current figure to a specified directory.
    Params:
        - dataset_data: List of the datasets (usually pre and post patch) containing Trial objects.
        - dataset_metrics: List of the datasets containing metrics for each Trial.
    """

    if len(dataset_data) != len(dataset_metrics):
        raise ValueError("dataset_data and dataset_metrics must have the same length.")
    # 1. Setup UI Elements
    sample_slider = widgets.IntSlider(
        min=0, max=min(len(ds) for ds in dataset_data) - 1, step=1, value=0,
        description="Sample: ", layout=widgets.Layout(width="100%")
    )
    channel_slider = widgets.IntSlider(
        min=0, max=n_chans - 1, step=1, value=0,
        description="Channel: ", layout=widgets.Layout(width="100%")
    )
    save_button = widgets.Button(description="Save Figure", button_style='success', layout=widgets.Layout(width="100%"))
    save_message = widgets.Output()

    # 2. The Plotting Callback
    def update_plot(sample_idx, channel_idx):
        try:

            trial_data_dict, ds_metrics = order_to_channel_viewer_format(dataset_data, dataset_metrics, sample_idx,
                                                                         channel_idx)

            show_reconstructions(
                data=trial_data_dict,
                metrics_data=ds_metrics,
                plot_title=f"Channel {channel_idx}, Sample {sample_idx}",
                bands=bands,
                freqs=freqs,
                rows=len(dataset_metrics),

                **styling_params
            )
        except Exception as e:
            # If a TypeError or IndexError happens, print it directly to the screen!
            import traceback
            print(f"ERROR: {traceback.format_exc()}")

    # 3. Create a dedicated "Screen" for the plot output
    plot_output = widgets.interactive_output(
        update_plot,
        {'sample_idx': sample_slider, 'channel_idx': channel_slider}
    )

    # 4. The Save Callback
    def on_save_clicked(b):
        trial_data_dict, ds_metrics = order_to_channel_viewer_format(dataset_data, dataset_metrics, sample_slider.value,
                                                                     channel_slider.value)
        filepath = save_dir / f"sample_{sample_slider.value}_ch_{channel_slider.value}_{an_suffix}_{excl_suffix}.png"
        with save_message:
            # Clear previous save messages
            save_message.clear_output()
            print(f"Saving figure to: {filepath} ...")
            show_reconstructions(
                data=trial_data_dict,
                metrics_data=ds_metrics,
                save=True,
                show=False,
                save_filename=filepath,
                plot_title=f"Channel {channel_slider.value}, Sample {sample_slider.value}",
                bands=bands,
                freqs=freqs,
                rows=len(dataset_metrics),
                **styling_params
            )
            print("Save complete!")

    save_button.on_click(on_save_clicked)

    # 5. Build the Master Layout and Display
    ui = widgets.VBox([
        sample_slider,
        channel_slider,
        save_button,
        save_message,
        plot_output  # The plot is now locked firmly beneath the buttons
    ])
    display(ui)


def set_heatmap_formatting(ax, target_mode, im=None, cbar_label="Power (dB)"):
    if target_mode == "autoencoder":
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude [uV]")
        ax.legend()
    elif target_mode in ["wavelet", "superlet"]:
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")

        # --- THE INSET AXES FIX ---
        if im is not None:
            # Create a child axis tied to the parent's exact dimensions.
            # Coordinates are [X_start, Y_start, Width, Height] as percentages.
            # 1.04 = Start 4% to the right of the heatmap's right edge
            # 0.00 = Start exactly at the bottom (0%)
            # 0.05 = Make it 5% wide
            # 1.00 = Make it exactly 100% the height of the heatmap
            cax = ax.inset_axes([1.04, 0, 0.05, 1.0])

            plt.colorbar(im, cax=cax, label=cbar_label)

        my_ticks = [5, 8, 12, 20, 30, 45]  # the frequency window of the targets
        ax.set_yticks(my_ticks)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_locator(ticker.NullLocator())


def show_reconstructions(data: List[Dict[str, Any]], metrics_data: List[Dict[str, Any]], suptitle, plot_title,
                         target_mode, freqs,
                         save_filename=None, save=False, show=True, rows=2, apply_log=False, bands=None, param_text="",
                         time_window=None, analysis_window=None, **kwargs):
    """
    Plots the target, reconstruction and error (all in dB) of one channel for every dataset (one row each), and the
    band-wise MAE of the shown part.
    time_window: (start, end) in s of the stored patch, e.g. settings.PATCH_WINDOWS["post"], so the time axis is relative to
    stimulus onset; None starts the axis at 0.
    analysis_window: (start, end) in s inside time_window, e.g. settings.POST_TARGET_WINDOW; only this part is plotted and
    the errors are computed from it. Needs time_window. None plots the whole patch.
    bands: {band name: frequency bin indices}, e.g. settings.FREQ_BAND_INDICES, for the band-wise MAE.
    metrics_data: one {"metadata": {subject, condition, id}} dict per dataset, see get_channel_viewer_metrics.
    """
    # 1. Height Adjustment & Constrained Layout (Fixes stretching and overlapping)
    total_rows = rows + 1 if target_mode in ["wavelet", "superlet"] else rows
    fig = plt.figure(figsize=(16, 4 * total_rows), layout="constrained")

    # 2. Unified GridSpec (3 columns for Wavelets, 1 column for Autoencoder)
    ncols = 3 if target_mode in ["wavelet", "superlet"] else 1
    gs = fig.add_gridspec(nrows=total_rows, ncols=ncols)

    fs = settings.RSFREQ if target_mode == "autoencoder" else settings.TARGET_FS  # time bins per second of the targets
    time_start = time_window[0] if time_window is not None else 0.0
    if analysis_window is not None:
        data = [{key: cut_to_timewindow(values, time_window, analysis_window, fs) for key, values in channel_data.items()}
                for channel_data in data]
        time_start = analysis_window[0]
    if apply_log:
        data = [{**channel_data, "output": 20 * np.log10(channel_data["output"] + 1e-7)} for channel_data in data]

    # one power scale for all rows, from the 1st to the 99th percentile so single very low bins do not squash it
    finite = np.concatenate([values[np.isfinite(values)] for channel_data in data for values in channel_data.values()])
    v_min, v_max = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    errors = [channel_data["target"] - channel_data["output"] for channel_data in data]
    finite_errors = np.concatenate([np.abs(error[np.isfinite(error)]) for error in errors])
    err_max = np.percentile(finite_errors, 99) if finite_errors.size else 1.0

    all_bandwise_data = []
    for i in range(len(data)):
        try:
            channel_recon = data[i]["output"]
            channel_true = data[i]["target"]
            meta = metrics_data[i]["metadata"]
            fig.suptitle(suptitle + meta["condition"] + f"\n{param_text}", fontweight="bold", fontsize=16)

            subj = meta.get("subject", "unknown")
            idx = meta.get("id", "unknown")
            row_label = f"({subj}, {idx})"

            # errors of the shown part only, NaN for a channel interpolated in this trial (no output)
            abs_error = np.abs(errors[i])
            interpolated = not np.isfinite(abs_error).any()
            mae = np.nan if interpolated else np.nanmean(abs_error)
            if bands is not None:
                all_bandwise_data.append([np.nan if interpolated else np.nanmean(abs_error[idxs]) for idxs in bands.values()])
            mae_text = "interpolated, no output" if interpolated else f"MAE: {mae:.2f} dB"

            time = (time_start + np.arange(channel_true.shape[-1]) / fs) * 1000  # ms relative to stimulus onset, start of every bin

            if target_mode == "autoencoder":
                # Maps exactly to row 'i', column '0'
                ax1 = fig.add_subplot(gs[i, 0])
                ax1.plot(time, channel_true, label='True', alpha=0.7)
                ax1.plot(time, channel_recon, label='Reconstructed', linestyle='--', alpha=0.9)
                ax1.set_title(f"{plot_title} - {row_label}", loc='left')
                ax1.text(0.95, 0.05, mae_text, transform=ax1.transAxes, ha='right', va='top', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                set_heatmap_formatting(ax1, target_mode)

            elif target_mode in ["wavelet", "superlet"]:
                # Column 1: True
                ax1 = fig.add_subplot(gs[i, 0])
                im1 = ax1.pcolormesh(time, freqs, channel_true, shading="gouraud", vmin=v_min, vmax=v_max)
                ax1.set_title(f"True | {row_label}", pad=10)
                set_heatmap_formatting(ax1, target_mode, im1)

                # Column 2: Recon
                ax2 = fig.add_subplot(gs[i, 1])
                im2 = ax2.pcolormesh(time, freqs, channel_recon, shading="gouraud", vmin=v_min, vmax=v_max)
                ax2.set_title(f"Reconstructed | {row_label}", pad=10)
                set_heatmap_formatting(ax2, target_mode, im2)

                # Column 3: Error, its own symmetric scale
                ax3 = fig.add_subplot(gs[i, 2])
                im3 = ax3.pcolormesh(time, freqs, errors[i], shading="gouraud", vmin=-err_max, vmax=err_max, cmap="RdBu_r")
                ax3.set_title(f"Error (True - Recon) | {row_label}", pad=10)
                set_heatmap_formatting(ax3, target_mode, im3, cbar_label="Error (dB)")

                ax3.text(0.95, 0.05, mae_text, transform=ax3.transAxes, ha='right', va='top',
                         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        except Exception as e:
            # Print full stack trace to Jupyter if a subplot fails!
            import traceback
            print(f"Error plotting reconstruction for channel {i}:\n{traceback.format_exc()}")

    if target_mode in ["wavelet", "superlet"] and all_bandwise_data:
        # gs[rows, :] means "put this in the last row, and span ALL columns"
        comp_ax = fig.add_subplot(gs[rows, :])

        # Give them clean labels (Baseline vs PostPatch)
        dataset_labels = ["Baseline", "Post-Patch"] if len(data) == 2 else [f"Source {i}" for i in
                                                                            range(len(data))]

        plot_grouped_bar(
            x_text=list(bands.keys()),
            width=0.8 / len(data),
            means=all_bandwise_data,
            labels=dataset_labels,
            title="Comparison of Band-wise MAE",
            single_channel=True,
            ax=comp_ax,
            y_label="MAE (dB)"
        )

    # Note: No plt.tight_layout() here because layout="constrained" handles it automatically!
    if save and save_filename:
        save_filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print("Saved figure to:", save_filename)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_grouped_bar(x_text, width, means: List[List[Any]], labels: List[str], title: str, single_channel=False,
                     ax=None, fig=None, sems: Optional[List[List[Any]]] = None, y_label="Normalized MAE (NMAE)",
                     ylim=None, n_text: str=None):
    x = np.arange(len(x_text))
    is_standalone = ax is None
    if is_standalone:
        fig, ax = plt.subplots(figsize=(10, 6))
    n = len(means)
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum', 'lightcoral']  # Add more colors if needed
    for i in range(n):
        group_means = means[i]
        offset = x + (i - (n - 1) / 2) * width  # bars of one group side by side, centered on the tick
        if not single_channel:
            group_sems = sems[i]
            label = labels[i]

            ax.bar(offset, group_means, width=width, yerr=group_sems, label=label, color=colors[i],
                   ecolor='black', capsize=6)
        else:
            ax.bar(offset, group_means, width=width, label=labels[i], color=colors[i])
            for j in range(len(x)):
                ax.text(offset[j], group_means[j], f"{group_means[j]:.3f}", ha='center', va='bottom', fontsize=9,
                        color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(x_text)

    # Formatting
    ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim(ylim)

    if n_text is not None:
        title+= f" ({n_text})"
    ax.set_title(title)
    ax.legend()

    if is_standalone:
        plt.tight_layout()
        plt.show()
    return fig, ax


def plot_time_series(times, means: Dict[str, np.ndarray], title: str, sems: Optional[Dict[str, np.ndarray]] = None,
                     x_label="Time (ms)", y_label="MAE", vlines: Optional[List[float]] = None,
                     hlines: Optional[List[float]] = None, ax=None, fig=None, n_text: str = None):
    """
    Plots one line per key of means against times, with a shaded ±SEM band if sems are given (same keys as means).
    vlines: x positions of dashed vertical lines, e.g. [0] for the stimulus onset. hlines: y positions of solid horizontal lines, e.g. [0] for differences.
    """
    is_standalone = ax is None
    if is_standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    for label, mean in means.items():
        line, = ax.plot(times, mean, label=label)
        if sems is not None:
            ax.fill_between(times, mean - sems[label], mean + sems[label], color=line.get_color(), alpha=0.25, linewidth=0)
    for x in vlines or []:
        ax.axvline(x, color='gray', linestyle='--', linewidth=1)
    for y in hlines or []:
        ax.axhline(y, color='black', linewidth=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if n_text is not None:
        title += f" ({n_text})"
    ax.set_title(title)
    ax.legend()

    if is_standalone:
        plt.tight_layout()
        plt.show()
    return fig, ax


def plot_time_series_grid(times, means: Dict[str, Dict[str, np.ndarray]], suptitle: str,
                          sems: Optional[Dict[str, Dict[str, np.ndarray]]] = None, y_label="MAE",
                          vlines: Optional[List[float]] = None, hlines: Optional[List[float]] = None, n_cols: int = 3):
    """
    Plots one plot_time_series subplot per panel, e.g. per band. Expects means (and sems) as {panel title: {line label: series}}.
    The y-axes are not shared, so panels with different scales stay readable.
    """
    n_rows = int(np.ceil(len(means) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for ax, (panel, panel_means) in zip(axes.flat, means.items()):
        plot_time_series(times, panel_means, title=panel, sems=sems[panel] if sems is not None else None,
                         y_label=y_label, vlines=vlines, hlines=hlines, ax=ax, fig=fig)
    for ax in axes.flat[len(means):]:
        ax.set_visible(False)

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
    return fig, axes


def save_button(fig, save_dir: Path, default_name="untitled"):
    name_input = widgets.Text(value=default_name, description='Filename:', placeholder='untitled')

    save_button_comp = widgets.Button(description="Save Figure")
    output_comp = widgets.Output()

    def on_save_clicked(b):
        with output_comp:
            output_comp.clear_output()
            core.io.save_figure(name_input=name_input.value, save_dir=save_dir, fig=fig, default_name=default_name)

    save_button_comp.on_click(on_save_clicked)
    display(widgets.VBox([widgets.HBox([name_input]), save_button_comp, output_comp]))

# functions for saving signal waveforms (for debugging purposes)
def save_heatmap(channel_data, save_dir: Path, title="Wavelet transform heatmap", xlabel="Time (ms)", ylabel="Frequency (bins)"):
    plt.figure(figsize=(10, 6))
    plt.imshow(channel_data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    plt.close()


def save_line_plot(channel_data, save_dir: Path, title="Epoch waveform", xlabel="Time (samples)", ylabel="Amplitude [uV]"):
    plt.figure(figsize=(12, 6))
    plt.plot(channel_data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    plt.close()


def plot_topomap_row(values_by_band, info, channel_names, vlim, cmap, label, title, save_path):
    """
    One topomap per band in a row, one colour scale shared by all panels (layout of the plasticity_plotting.ipynb topomaps).
    values_by_band: {band: Series indexed by channel name}; channels without a value are left out of that panel.
    info: mne.Info of the channels in channel_names order (plot_data_processing.create_topomap_info).
    """
    fig, axes = plt.subplots(1, len(values_by_band), figsize=(24, 4))
    axes = np.atleast_1d(axes)
    im = None
    for ax, (band, values) in zip(axes, values_by_band.items()):
        vals = values.reindex(channel_names).to_numpy(dtype=float)
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
    """Returns the largest absolute value of all bands' Series, for a colour scale symmetric around 0 (default if all are empty)."""
    return max([np.nanmax(np.abs(v.to_numpy())) for v in values_by_band.values() if len(v)] or [default])
