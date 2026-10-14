#!/usr/bin/env python3
"""Utility script for analyzing uncertainty evaluation CSV outputs.

Supports:
- Sparsification plots (mean +/- standard error over removed fractions)
- Coverage gap plots across interval levels (global and per-channel)
- Mean-vs-refined quality gap extraction
- Calibration extremes extraction
- Calibration metric distribution comparison across experiments
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from matplotlib.gridspec import GridSpecFromSubplotSpec

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
    }
)


LOWER_IS_BETTER = {"rmse", "mae", "mean_nll"}
HIGHER_IS_BETTER = {"psnr", "pearson_uncertainty_error", "spearman_uncertainty_error"}


@dataclass
class ExperimentFrame:
    label: str
    path: Path
    frame: pd.DataFrame


def _default_labels(paths: List[Path]) -> List[str]:
    labels = []
    for p in paths:
        parent = p.parent.name
        stem = p.stem
        labels.append(f"{parent}/{stem}")
    return labels


def _load_experiments(csvs: Iterable[str], labels: Iterable[str] | None = None) -> List[ExperimentFrame]:
    paths = [Path(p).expanduser().resolve() for p in csvs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    labels_list = list(labels) if labels is not None else _default_labels(paths)
    if len(labels_list) != len(paths):
        raise ValueError("Number of labels must match number of CSV files.")

    out = []
    for label, path in zip(labels_list, paths):
        out.append(ExperimentFrame(label=label, path=path, frame=pd.read_csv(path)))
    return out


def _channel_colors(channel_names: Sequence[str]) -> Dict[str, str]:
    # Okabe-Ito palette for colorblind-safe categorical lines.
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ]
    colors = {}
    for idx, channel in enumerate(channel_names):
        colors[channel] = palette[idx % len(palette)]
    return colors


def _save_or_show(fig: plt.Figure, output: str | None):
    if output:
        out_path = Path(output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
        print(f"Saved plot: {out_path}")
    else:
        plt.show()


def _experiment_colors(labels: List[str]) -> dict:
    if len(labels) == 1:
        return {labels[0]: "black"}

    palette = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]
    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


def _apply_boxplot_style(bp, labels, colors):
    for i, box in enumerate(bp["boxes"]):
        label = labels[i]
        box.set(facecolor="white", edgecolor=colors[label], linewidth=1.6)
    for i, whisker in enumerate(bp["whiskers"]):
        label = labels[i // 2]
        whisker.set(color=colors[label], linewidth=1.4)
    for i, cap in enumerate(bp["caps"]):
        label = labels[i // 2]
        cap.set(color=colors[label], linewidth=1.4)
    for i, median in enumerate(bp["medians"]):
        label = labels[i]
        median.set(color=colors[label], linewidth=1.8)
    for i, mean in enumerate(bp["means"]):
        label = labels[i]
        mean.set(marker="o", markerfacecolor=colors[label], markeredgecolor=colors[label], markersize=4)
    for i, flier in enumerate(bp["fliers"]):
        label = labels[i]
        flier.set(marker="o", markerfacecolor=colors[label], markeredgecolor=colors[label], alpha=0.5, markersize=3)


def _add_broken_axis_marks(ax_top, ax_bottom):
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-0.015, +0.015), (-0.015, +0.015), **kwargs)
    ax_top.plot((0.985, 1.015), (-0.015, +0.015), **kwargs)
    kwargs = dict(transform=ax_bottom.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_bottom.plot((-0.015, +0.015), (0.985, 1.015), **kwargs)
    ax_bottom.plot((0.985, 1.015), (0.985, 1.015), **kwargs)


def cmd_plot_sparsity(args: argparse.Namespace):
    exps = _load_experiments(args.csvs, args.labels)
    metric = args.metric.lower()
    if metric not in {"rmse", "mae"}:
        raise ValueError("--metric must be one of: rmse, mae")

    if args.xlim is not None and args.xlim[0] >= args.xlim[1]:
        raise ValueError("--xlim requires XMIN < XMAX")
    if args.ylim is not None and args.ylim[0] >= args.ylim[1]:
        raise ValueError("--ylim requires YMIN < YMAX")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = _experiment_colors([e.label for e in exps])
    use_log = not args.linear_y

    if use_log and args.ylim is not None and args.ylim[0] <= 0:
        raise ValueError("--ylim lower bound must be > 0 when using logarithmic y-axis")

    for exp in exps:
        df = exp.frame
        needed = {"remove_fraction", metric}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {exp.path}: {sorted(missing)}")

        grouped = (
            df.groupby("remove_fraction")[metric]
            .agg(["mean", "std", "count"])  # type: ignore[arg-type]
            .reset_index()
            .sort_values("remove_fraction")
        )
        grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1.0))

        x = grouped["remove_fraction"].to_numpy()
        y = grouped["mean"].to_numpy()
        yerr = grouped["sem"].fillna(0.0).to_numpy()
        if use_log:
            # Log scale cannot render intervals crossing zero; clip lower error
            # so that mean points and lower bars remain visible.
            min_positive = np.maximum(y * 1e-3, 1e-12)
            lower_err = np.minimum(yerr, np.maximum(y - min_positive, 0.0))
            yerr_plot = np.vstack([lower_err, yerr])
        else:
            yerr_plot = yerr

        ax.errorbar(
            x,
            y,
            yerr=yerr_plot,
            marker="o",
            markersize=4,
            capsize=3,
            linewidth=1.6,
            color=colors[exp.label],
            ecolor=colors[exp.label],
            label=exp.label,
        )

    ax.set_xlabel("Removed Fraction")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Sparsification: {metric.upper()} vs Removed Fraction")
    if use_log:
        ax.set_yscale("log", nonpositive="clip")
    if args.xlim is not None:
        ax.set_xlim(args.xlim)
    if args.ylim is not None:
        ax.set_ylim(args.ylim)
    ax.grid(alpha=0.3)
    ax.legend()

    _save_or_show(fig, args.output)


def cmd_plot_coverage(args: argparse.Namespace):
    exps = _load_experiments(args.csvs, args.labels)
    show_legend = not args.no_legend
    global_linestyle = "-" if args.solid_lines else "--"
    labels = [e.label for e in exps]
    colors = _experiment_colors(labels)

    summary_frames = []
    channel_frames = []

    for exp in exps:
        df = exp.frame
        needed = {"interval", "gap"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {exp.path}: {sorted(missing)}")

        grouped = (
            df.groupby("interval", as_index=False)["gap"]
            .agg(["mean", "std", "count"])  # type: ignore[arg-type]
            .reset_index()
            .sort_values("interval")
            .rename(columns={"mean": "gap_mean", "std": "gap_std", "count": "gap_count"})
        )
        grouped["gap_sem"] = grouped["gap_std"] / np.sqrt(grouped["gap_count"].clip(lower=1.0))
        grouped["label"] = exp.label
        grouped["gap_sem"] = grouped["gap_sem"].fillna(0.0)
        summary_frames.append(grouped)

        ch_cols = [c for c in df.columns if c.startswith("channel")]
        for ch in ch_cols:
            ch_gap = df.assign(channel_gap=df[ch] - df["interval"])
            ch_grouped = (
                ch_gap.groupby("interval", as_index=False)["channel_gap"]
                .agg(["mean", "std", "count"])  # type: ignore[arg-type]
                .reset_index()
                .sort_values("interval")
                .rename(columns={"mean": "gap_mean", "std": "gap_std", "count": "gap_count"})
            )
            ch_grouped["gap_sem"] = ch_grouped["gap_std"] / np.sqrt(ch_grouped["gap_count"].clip(lower=1.0))
            ch_grouped["label"] = exp.label
            ch_grouped["channel"] = ch
            ch_grouped["gap_sem"] = ch_grouped["gap_sem"].fillna(0.0)
            channel_frames.append(ch_grouped)

    summary = pd.concat(summary_frames, ignore_index=True)
    all_intervals = np.sort(summary["interval"].unique())
    xpos = np.arange(len(all_intervals))
    interval_to_x = {val: idx for idx, val in enumerate(all_intervals)}
    tick_labels = [f"{v:g}" for v in all_intervals]

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, sdf in summary.groupby("label", sort=False):
        sdf = sdf.sort_values("interval")
        x = np.array([interval_to_x[v] for v in sdf["interval"].to_numpy()])
        ax.errorbar(
            x,
            sdf["gap_mean"].to_numpy(),
            yerr=sdf["gap_sem"].to_numpy(),
            marker="s",
            linestyle=global_linestyle,
            capsize=3,
            linewidth=1.6,
            color=colors[label],
            ecolor=colors[label],
            alpha=0.8,
            label=f"{label} (global)",
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Interval")
    ax.set_xticks(xpos)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Gap (empirical - interval)")
    ax.set_title("Coverage Gap vs Interval")
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend()

    if not args.channels:
        _save_or_show(fig, args.output)
        return

    if not channel_frames:
        print("No channel columns found; skipping channel plot.")
        _save_or_show(fig, args.output)
        return

    ch_df = pd.concat(channel_frames, ignore_index=True)
    channel_names = sorted(ch_df["channel"].unique().tolist())
    channel_color = _channel_colors(channel_names)
    channel_display = {channel: channel for channel in channel_names}
    if args.channel_labels is not None:
        if len(args.channel_labels) != len(channel_names):
            raise ValueError(
                "Number of --channel-labels entries must match number of detected channels "
                f"({len(channel_names)}): {channel_names}"
            )
        channel_display = {
            channel: display
            for channel, display in zip(channel_names, args.channel_labels)
        }
    overlay = bool(args.channels_overlay)

    if overlay:
        for (label, channel), sdf in ch_df.groupby(["label", "channel"], sort=False):
            sdf = sdf.sort_values("interval")
            x = np.array([interval_to_x[v] for v in sdf["interval"].to_numpy()])
            display = channel_display[channel]
            line_label = f"{label}:{display}" if len(exps) > 1 else display
            ax.errorbar(
                x,
                sdf["gap_mean"].to_numpy(),
                yerr=sdf["gap_sem"].to_numpy(),
                marker="o",
                capsize=3,
                linewidth=1.2,
                linestyle="-",
                color=channel_color[channel],
                ecolor=channel_color[channel],
                alpha=0.8,
                label=line_label,
            )

        ax.set_title("Coverage Gap vs Interval (global + channels)")
        if show_legend:
            ax.legend()
        out_path = args.output if args.output else args.channels_output
        _save_or_show(fig, out_path)
        return

    _save_or_show(fig, args.output)

    fig_ch, ax_ch = plt.subplots(figsize=(10, 5))

    for (label, channel), sdf in ch_df.groupby(["label", "channel"], sort=False):
        sdf = sdf.sort_values("interval")
        x = np.array([interval_to_x[v] for v in sdf["interval"].to_numpy()])
        display = channel_display[channel]
        line_label = f"{label}:{display}" if len(exps) > 1 else display
        ax_ch.errorbar(
            x,
            sdf["gap_mean"].to_numpy(),
            yerr=sdf["gap_sem"].to_numpy(),
            marker="o",
            capsize=3,
            linewidth=1.4,
            linestyle="-",
            color=channel_color[channel],
            ecolor=channel_color[channel],
            label=line_label,
        )

    ax_ch.axhline(0.0, linestyle="--", linewidth=1.0)
    ax_ch.set_xlabel("Interval")
    ax_ch.set_xticks(xpos)
    ax_ch.set_xticklabels(tick_labels)
    ax_ch.set_ylabel("Channel Gap (empirical - interval)")
    ax_ch.set_title("Per-Channel Coverage Gaps vs Interval")
    ax_ch.grid(alpha=0.3)
    if show_legend:
        ax_ch.legend()

    out_channels = args.channels_output
    _save_or_show(fig_ch, out_channels)


def _quality_gap_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    needed = {"filename", "variant", metric}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    pivot = (
        df[df["variant"].isin(["mean", "refined"])]
        .pivot_table(index="filename", columns="variant", values=metric, aggfunc="mean")
        .dropna(subset=["mean", "refined"])
        .reset_index()
    )

    if metric in LOWER_IS_BETTER:
        pivot["delta"] = pivot["mean"] - pivot["refined"]
    elif metric in HIGHER_IS_BETTER:
        pivot["delta"] = pivot["refined"] - pivot["mean"]
    else:
        raise ValueError(f"Unsupported quality metric: {metric}")

    return pivot


def cmd_extract_quality_gaps(args: argparse.Namespace):
    exps = _load_experiments(args.csvs, args.labels)
    metric = args.metric.lower()
    if metric not in {"rmse", "mae", "psnr"}:
        raise ValueError("--metric must be one of: rmse, mae, psnr")
    top_k = int(args.top_k)
    if top_k <= 0:
        raise ValueError("--top-k must be a positive integer")

    records = []
    for exp in exps:
        gap_df = _quality_gap_by_metric(exp.frame, metric)
        if gap_df.empty:
            raise ValueError(f"No comparable mean/refined rows in {exp.path}")

        best_rows = gap_df.sort_values("delta", ascending=False).head(top_k).reset_index(drop=True)
        worst_rows = gap_df.sort_values("delta", ascending=True).head(top_k).reset_index(drop=True)

        for rank, row in best_rows.iterrows():
            records.append(
                {
                    "experiment": exp.label,
                    "metric": metric,
                    "group": "best_refined",
                    "rank": int(rank + 1),
                    "filename": row["filename"],
                    "delta": float(row["delta"]),
                }
            )

        for rank, row in worst_rows.iterrows():
            records.append(
                {
                    "experiment": exp.label,
                    "metric": metric,
                    "group": "worst_refined",
                    "rank": int(rank + 1),
                    "filename": row["filename"],
                    "delta": float(row["delta"]),
                }
            )

    out_df = pd.DataFrame(records)
    print(out_df.to_string(index=False))

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved table: {out_path}")


def _best_worst(df: pd.DataFrame, col: str, higher_is_better: bool) -> Tuple[pd.Series, pd.Series]:
    clean = df[["filename", col]].dropna()
    if clean.empty:
        raise ValueError(f"No valid rows for {col}")

    if higher_is_better:
        best = clean.sort_values(col, ascending=False).iloc[0]
        worst = clean.sort_values(col, ascending=True).iloc[0]
    else:
        best = clean.sort_values(col, ascending=True).iloc[0]
        worst = clean.sort_values(col, ascending=False).iloc[0]
    return best, worst


def cmd_extract_calibration_extremes(args: argparse.Namespace):
    exps = _load_experiments(args.csvs, args.labels)

    metrics = [
        ("pearson_uncertainty_error", True),
        ("spearman_uncertainty_error", True),
        ("mean_nll", False),
    ]

    rows = []
    for exp in exps:
        df = exp.frame
        needed = {"filename", "pearson_uncertainty_error", "spearman_uncertainty_error", "mean_nll"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {exp.path}: {sorted(missing)}")

        for metric, higher_is_better in metrics:
            best, worst = _best_worst(df, metric, higher_is_better)
            rows.append(
                {
                    "experiment": exp.label,
                    "metric": metric,
                    "best_filename": best["filename"],
                    "best_value": float(best[metric]),
                    "worst_filename": worst["filename"],
                    "worst_value": float(worst[metric]),
                }
            )

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved table: {out_path}")


def cmd_plot_calibration_compare(args: argparse.Namespace):
    exps = _load_experiments(args.csvs, args.labels)
    labels = [e.label for e in exps]
    colors = {label: "black" for label in labels}

    needed = {"pearson_uncertainty_error", "spearman_uncertainty_error", "mean_nll"}
    per_exp_values = []

    for exp in exps:
        df = exp.frame
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {exp.path}: {sorted(missing)}")

        per_exp_values.append(
            {
                "label": exp.label,
                "pearson_uncertainty_error": df["pearson_uncertainty_error"].dropna().to_numpy(),
                "spearman_uncertainty_error": df["spearman_uncertainty_error"].dropna().to_numpy(),
                "mean_nll": df["mean_nll"].dropna().to_numpy(),
            }
        )

    cols = ["pearson_uncertainty_error", "spearman_uncertainty_error", "mean_nll"]
    title_map = {
        "pearson_uncertainty_error": "Pearson",
        "spearman_uncertainty_error": "Spearman",
    }

    if args.broken_y_mean_nll:
        fig = plt.figure(figsize=(14, 4))
        outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.35)
        axes = [
            fig.add_subplot(outer[0]),
            fig.add_subplot(outer[1]),
        ]
        mean_nll_spec = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], height_ratios=[1, 3], hspace=0.05)
        mean_nll_top = fig.add_subplot(mean_nll_spec[0])
        mean_nll_bottom = fig.add_subplot(mean_nll_spec[1], sharex=mean_nll_top)
    else:
        fig, axes_arr = plt.subplots(1, 3, figsize=(14, 4))
        axes = [axes_arr[0], axes_arr[1]]
        mean_nll_top = None
        mean_nll_bottom = axes_arr[2]

    for ax, col in zip(axes, cols[:2]):
        data = [entry[col] for entry in per_exp_values]
        labels = [entry["label"] for entry in per_exp_values]
        bp = ax.boxplot(
            data,
            labels=labels,
            whis=(2.5, 97.5),
            showfliers=True,
            showmeans=True,
            meanline=False,
            patch_artist=True,
        )
        _apply_boxplot_style(bp, labels, colors)

        ax.set_title(title_map[col])
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

    data = [entry["mean_nll"] for entry in per_exp_values]
    labels = [entry["label"] for entry in per_exp_values]
    if args.broken_y_mean_nll:
        flat = np.concatenate([np.asarray(v) for v in data if len(v) > 0])
        if flat.size == 0:
            raise ValueError("No valid mean_nll values found for broken-axis plot.")

        bp_top = mean_nll_top.boxplot(
            data,
            labels=labels,
            whis=(2.5, 97.5),
            showfliers=True,
            showmeans=True,
            meanline=False,
            patch_artist=True,
        )
        bp_bottom = mean_nll_bottom.boxplot(
            data,
            labels=labels,
            whis=(2.5, 97.5),
            showfliers=True,
            showmeans=True,
            meanline=False,
            patch_artist=True,
        )
        _apply_boxplot_style(bp_top, labels, colors)
        _apply_boxplot_style(bp_bottom, labels, colors)

        upper_whisker_max = max(
            np.max(np.asarray(line.get_ydata()))
            for line in bp_top["whiskers"][1::2]
        )
        break_margin = float(args.mean_nll_break_margin)
        break_gap = float(args.mean_nll_break_gap)
        if break_margin <= 0 or break_gap <= 0:
            raise ValueError("mean_nll break margin and gap must be > 0.")

        lower_top = upper_whisker_max * (1.0 + break_margin)
        upper_bottom = upper_whisker_max * (1.0 + break_gap)
        if upper_bottom <= lower_top:
            upper_bottom = lower_top * (1.0 + max(break_gap, 0.02))

        mean_nll_top.set_ylim(upper_bottom, flat.max() * 1.03 if flat.max() > upper_bottom else upper_bottom * 1.05)
        lower_min = flat.min()
        mean_nll_bottom.set_ylim(lower_min * 0.97 if lower_min < 0 else max(lower_min * 0.97, 1e-8), lower_top)
        mean_nll_top.spines["bottom"].set_visible(False)
        mean_nll_bottom.spines["top"].set_visible(False)
        mean_nll_top.tick_params(labelbottom=False)
        mean_nll_top.set_title("mean_nll")
        mean_nll_bottom.tick_params(axis="x", rotation=25)
        mean_nll_top.grid(axis="y", alpha=0.3)
        mean_nll_bottom.grid(axis="y", alpha=0.3)
        _add_broken_axis_marks(mean_nll_top, mean_nll_bottom)

        if args.log_y_mean_nll:
            print("--log-y-mean-nll is ignored when --broken-y-mean-nll is enabled.")
    else:
        bp = mean_nll_bottom.boxplot(
            data,
            labels=labels,
            whis=(2.5, 97.5),
            showfliers=True,
            showmeans=True,
            meanline=False,
            patch_artist=True,
        )
        _apply_boxplot_style(bp, labels, colors)

        if args.log_y_mean_nll:
            flat = np.concatenate([np.asarray(v) for v in data if len(v) > 0])
            if np.any(flat <= 0):
                print(
                    "mean_nll contains non-positive values; using symlog instead of log scale."
                )
                mean_nll_bottom.set_yscale("symlog", linthresh=args.symlog_linthresh)
            else:
                mean_nll_bottom.set_yscale("log")

        mean_nll_bottom.set_title("mean_nll")
        mean_nll_bottom.tick_params(axis="x", rotation=25)
        mean_nll_bottom.grid(axis="y", alpha=0.3)

    fig.suptitle("Calibration Metric Distributions by Experiment")
    fig.tight_layout()
    _save_or_show(fig, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze uncertainty evaluation CSV files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sparse = sub.add_parser("plot-sparsity", help="Plot MAE/RMSE over removed fractions.")
    p_sparse.add_argument("--csvs", nargs="+", required=True, help="Input sparsification CSV files.")
    p_sparse.add_argument("--labels", nargs="*", help="Optional labels matching --csvs order.")
    p_sparse.add_argument("--metric", default="rmse", choices=["rmse", "mae"], help="Metric to plot.")
    p_sparse.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear y-axis. Default is logarithmic y-axis.",
    )
    p_sparse.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        help="Optional x-axis limits for removed fraction.",
    )
    p_sparse.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits for the selected metric.",
    )
    p_sparse.add_argument("--output", help="Optional output image path.")
    p_sparse.set_defaults(func=cmd_plot_sparsity)

    p_cov = sub.add_parser("plot-coverage", help="Plot coverage gap across intervals.")
    p_cov.add_argument("--csvs", nargs="+", required=True, help="Input coverage CSV files.")
    p_cov.add_argument("--labels", nargs="*", help="Optional labels matching --csvs order.")
    p_cov.add_argument(
        "--channel-labels",
        nargs="*",
        help=(
            "Optional display labels for detected channel columns in sorted order "
            "(for example channel0 channel1 ...). Length must match number of channels."
        ),
    )
    p_cov.add_argument("--channels", action="store_true", help="Also plot per-channel gaps on one graph.")
    p_cov.add_argument(
        "--channels-overlay",
        action="store_true",
        help="Overlay channel curves onto the main coverage plot instead of creating a separate channel figure.",
    )
    p_cov.add_argument(
        "--solid-lines",
        action="store_true",
        help="Use solid lines for global coverage curves (default is dashed).",
    )
    p_cov.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide legends in coverage plots.",
    )
    p_cov.add_argument("--output", help="Optional output image path for global gap plot.")
    p_cov.add_argument(
        "--channels-output",
        help="Optional output image path for channel plot. If omitted and --channels is set, shows interactively.",
    )
    p_cov.set_defaults(func=cmd_plot_coverage)

    p_q = sub.add_parser(
        "extract-quality-gaps",
        help="Extract top-K filenames with largest/smallest refined-vs-mean gap.",
    )
    p_q.add_argument("--csvs", nargs="+", required=True, help="Input prediction_quality CSV files.")
    p_q.add_argument("--labels", nargs="*", help="Optional labels matching --csvs order.")
    p_q.add_argument("--metric", default="rmse", choices=["rmse", "mae", "psnr"], help="Quality metric.")
    p_q.add_argument("--top-k", type=int, default=5, help="How many best/worst candidates to extract per experiment.")
    p_q.add_argument("--output", help="Optional CSV output path.")
    p_q.set_defaults(func=cmd_extract_quality_gaps)

    p_c = sub.add_parser(
        "extract-calibration-extremes",
        help="Extract best/worst filenames by Pearson, Spearman, and mean_nll.",
    )
    p_c.add_argument("--csvs", nargs="+", required=True, help="Input calibration_summary CSV files.")
    p_c.add_argument("--labels", nargs="*", help="Optional labels matching --csvs order.")
    p_c.add_argument("--output", help="Optional CSV output path.")
    p_c.set_defaults(func=cmd_extract_calibration_extremes)

    p_cc = sub.add_parser(
        "plot-calibration-compare",
        help="Plot calibration distributions with boxplots for multiple experiments.",
    )
    p_cc.add_argument("--csvs", nargs="+", required=True, help="Input calibration_summary CSV files.")
    p_cc.add_argument("--labels", nargs="*", help="Optional labels matching --csvs order.")
    p_cc.add_argument(
        "--log-y-mean-nll",
        action="store_true",
        help="Use log scale for the mean_nll panel when values are positive; falls back to symlog otherwise.",
    )
    p_cc.add_argument(
        "--broken-y-mean-nll",
        action="store_true",
        help="Use a broken y-axis for the mean_nll panel to separate main values from extreme outliers.",
    )
    p_cc.add_argument(
        "--mean-nll-break-margin",
        type=float,
        default=0.02,
        help="Fraction above the upper whisker where the lower panel should end.",
    )
    p_cc.add_argument(
        "--mean-nll-break-gap",
        type=float,
        default=0.08,
        help="Fraction above the upper whisker where the upper panel should start.",
    )
    p_cc.add_argument(
        "--symlog-linthresh",
        type=float,
        default=1e-2,
        help="Linear threshold used when mean_nll falls back to symlog scaling.",
    )
    p_cc.add_argument("--output", help="Optional output image path.")
    p_cc.set_defaults(func=cmd_plot_calibration_compare)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
