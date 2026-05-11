"""
Visualization utilities for EDA, cleaning diagnostics, training, and evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise KeyError when required dataframe columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required dataframe column(s): {missing}")
    

def _read_rgb_image(path: str | Path) -> Image.Image | None:
    """Read a path as an RGB PIL image, returning None when unreadable."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _resolve_axes_array(axes: Any) -> np.ndarray:
    """Return axes as a flat numpy array."""
    return np.asarray(axes, dtype=object).reshape(-1)


def _save_figure(fig: plt.Figure, save_path: str | Path | None, dpi: int = 160) -> Path | None:
    """Save a figure if save_path is provided."""
    if save_path is None:
        return None

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return output


def _finalize_figure(
    fig: plt.Figure,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Save/show a figure consistently and return it."""
    _save_figure(fig, save_path)

    if show:
        plt.show()

    return fig


# -----------------------------------------------------------------------------
# Generic image-grid plotting
# -----------------------------------------------------------------------------


def plot_image_grid_from_df(
    df: pd.DataFrame,
    n: int = 12,
    path_col: str = "path",
    title_col: str | None = None,
    subtitle_cols: Sequence[str] | None = None,
    filter_col: str | None = None,
    filter_value: Any | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    random_sample: bool = False,
    seed: int = 42,
    n_cols: int = 4,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a generic image grid from a dataframe using a path column.

    This replaces semantic duplicates such as ``plot_removed_examples`` and
    ``plot_wrong_predictions``. The same generic function can plot removed
    images, wrong predictions, extreme samples, or arbitrary dataframe samples.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n_cols <= 0:
        raise ValueError("n_cols must be positive.")

    _ensure_columns(df, [path_col])
    view_df = df.copy()

    if filter_col is not None:
        _ensure_columns(view_df, [filter_col])
        if filter_value is not None:
            view_df = view_df[
                view_df[filter_col].astype(str).str.contains(str(filter_value), regex=False, na=False)
            ]

    if sort_by is not None:
        _ensure_columns(view_df, [sort_by])
        view_df = view_df.sort_values(sort_by, ascending=ascending)
    elif random_sample:
        view_df = view_df.sample(frac=1.0, random_state=seed)

    view_df = view_df.head(n).copy()

    if view_df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No images to display", ha="center", va="center")
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        _finalize_figure(fig, save_path=save_path, show=show)
        return fig, np.asarray([ax], dtype=object)

    n_cols = min(n_cols, len(view_df))
    n_rows = int(np.ceil(len(view_df) / n_cols))

    if figsize is None:
        figsize = (n_cols * 3.1, n_rows * 3.4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = _resolve_axes_array(axes)

    subtitle_cols = list(subtitle_cols or [])

    for ax, (_, row) in zip(axes_flat, view_df.iterrows()):
        image = _read_rgb_image(row[path_col])
        if image is None:
            ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")
        else:
            ax.imshow(image)

        ax.axis("off")

        title_parts: list[str] = []
        if title_col is not None and title_col in row:
            title_parts.append(str(row[title_col]))

        for col in subtitle_cols:
            if col in row:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.4g}"
                title_parts.append(f"{col}: {value}")

        if title_parts:
            ax.set_title("\n".join(title_parts), fontsize=8)

    for ax in axes_flat[len(view_df):]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, axes_flat


# -----------------------------------------------------------------------------
# EDA plots
# -----------------------------------------------------------------------------


def plot_sample_grid(
    df: pd.DataFrame,
    n_per_class: int = 5,
    path_col: str = "path",
    label_col: str = "label_name",
    class_order: Sequence[Any] | None = None,
    seed: int = 42,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot sampled images by class."""
    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive.")

    _ensure_columns(df, [path_col, label_col])

    if class_order is None:
        class_order = sorted(df[label_col].dropna().unique().tolist())

    if not class_order:
        raise ValueError("No classes found to plot.")

    if figsize is None:
        figsize = (n_per_class * 2.4, len(class_order) * 2.4)

    fig, axes = plt.subplots(len(class_order), n_per_class, figsize=figsize)
    axes_2d = np.asarray(axes, dtype=object)

    if len(class_order) == 1:
        axes_2d = axes_2d.reshape(1, -1)

    for row_idx, class_name in enumerate(class_order):
        group = df[df[label_col] == class_name]
        sample = group.sample(n=min(n_per_class, len(group)), random_state=seed) if len(group) else group

        for col_idx in range(n_per_class):
            ax = axes_2d[row_idx, col_idx]

            if col_idx >= len(sample):
                ax.axis("off")
                continue

            image = _read_rgb_image(sample.iloc[col_idx][path_col])
            if image is None:
                ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")
            else:
                ax.imshow(image)

            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(str(class_name), fontsize=11, fontweight="bold")

    fig.suptitle("Sample Images by Class", fontsize=14, fontweight="bold")
    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, axes_2d


def plot_pie_chart(
    df: pd.DataFrame,
    category_col: str,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot distribution of any categorical column as a pie chart."""
    _ensure_columns(df, [category_col])
    counts = df[category_col].value_counts(dropna=False).sort_index()

    if counts.empty:
        raise ValueError(f"No values to plot for column {category_col}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
    ax.set_title(title or f"Distribution of {category_col}", fontweight="bold")
    ax.axis("equal")

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_bar_chart(
    df: pd.DataFrame,
    category_col = str,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str = "Count",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot distribution of any categorical column as a bar chart."""
    _ensure_columns(df, [category_col])
    counts = df[category_col].value_counts(dropna=False).sort_index()

    if counts.empty:
        raise ValueError(f"No values to plot for column {category_col}")

    total = counts.sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values)

    for bar, count in zip(bars, counts.values):
        pct = (count / total) * 100.0 if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title or f"Count by {category_col}", fontweight="bold")
    ax.set_xlabel(x_label or category_col)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_scatter_distribution(
    df: pd.DataFrame,
    x_col: str,       
    y_col: str,        
    hue_col: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot scatter distribution for any two numeric columns."""
    _ensure_columns(df, [x_col, y_col])
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    if plot_df.empty:
        raise ValueError(f"No valid numeric rows to plot for {x_col} and {y_col}.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col if hue_col in plot_df.columns else None,
        alpha=0.45,
        s=28,
        ax=ax,
    )
    ax.set_title(title or f"Scatter: {x_col} vs {y_col}", fontweight="bold")
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    ax.grid(axis="both", linestyle="--", alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_rgb_channel_kde(
    df: pd.DataFrame,
    sample_per_class: int = 300,
    path_col: str = "path",
    label_col: str = "label_name",
    class_order: Sequence[Any] | None = None,
    seed: int = 42,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot KDE of per-image mean RGB channels by class."""
    if sample_per_class <= 0:
        raise ValueError("sample_per_class must be positive.")

    _ensure_columns(df, [path_col, label_col])

    if class_order is None:
        class_order = sorted(df[label_col].dropna().unique().tolist())

    if not class_order:
        raise ValueError("No classes found to plot.")

    sampled_parts = []
    for class_name in class_order:
        group = df[df[label_col] == class_name]
        if len(group):
            sampled_parts.append(group.sample(n=min(sample_per_class, len(group)), random_state=seed))

    if not sampled_parts:
        raise ValueError("No samples available for RGB KDE.")

    sampled = pd.concat(sampled_parts, ignore_index=True)

    def mean_rgb(path: str | Path) -> tuple[float, float, float]:
        image = _read_rgb_image(path)
        if image is None:
            return np.nan, np.nan, np.nan
        arr = np.asarray(image.resize((64, 64)))
        return (
            float(arr[:, :, 0].mean()),
            float(arr[:, :, 1].mean()),
            float(arr[:, :, 2].mean()),
        )

    sampled[["R_mean", "G_mean", "B_mean"]] = pd.DataFrame(
        sampled[path_col].apply(mean_rgb).tolist(),
        index=sampled.index,
    )

    fig, axes = plt.subplots(1, len(class_order), figsize=(6 * len(class_order), 4.5))
    axes_flat = _resolve_axes_array(axes)

    for ax, class_name in zip(axes_flat, class_order):
        part = sampled[sampled[label_col] == class_name].dropna(subset=["R_mean", "G_mean", "B_mean"])
        if part.empty:
            ax.axis("off")
            continue

        for col, color in [("R_mean", "red"), ("G_mean", "green"), ("B_mean", "blue")]:
            sns.kdeplot(data=part, x=col, color=color, fill=True, alpha=0.18, ax=ax, label=color.upper())

        ax.set_title(str(class_name), fontweight="bold")
        ax.set_xlabel("Mean Intensity")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("RGB Channel KDE by Class", fontsize=14, fontweight="bold")
    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, axes_flat


# -----------------------------------------------------------------------------
# Audit / cleaning diagnostic plots
# -----------------------------------------------------------------------------


def plot_metric_distribution(
    audit_df: pd.DataFrame,
    metric: str,
    label_col: str = "label_name",
    thresholds: Sequence[float] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one audit-metric distribution, optionally with threshold lines."""
    _ensure_columns(audit_df, [metric])
    plot_df = audit_df.dropna(subset=[metric]).copy()

    if plot_df.empty:
        raise ValueError(f"No valid rows to plot for metric: {metric}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        data=plot_df,
        x=metric,
        hue=label_col if label_col in plot_df.columns else None,
        fill=True,
        common_norm=False,
        alpha=0.2,
        ax=ax,
    )

    if thresholds is not None:
        for value in thresholds:
            ax.axvline(value, linestyle="--", linewidth=1.2)

    ax.set_title(title or f"Distribution of {metric}", fontweight="bold")
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_metric_correlation_heatmap(
    audit_df: pd.DataFrame,
    metrics: Sequence[str],
    title: str = "Correlation Matrix of Image Quality Metrics",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot correlation heatmap of selected audit metrics."""
    cols = [col for col in metrics if col in audit_df.columns]
    if not cols:
        raise ValueError("No valid metric columns were provided.")

    corr = audit_df[cols].apply(pd.to_numeric, errors="coerce").corr()

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.75), max(6, len(cols) * 0.65)))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontweight="bold")

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_before_after_cleaning(
    summary_df: pd.DataFrame,
    label_col: str = "label_name",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot class counts before/after cleaning from summarize_cleaning output."""
    required = [label_col, "count_before", "count_after", "count_removed"]
    _ensure_columns(summary_df, required)

    plot_df = summary_df[summary_df[label_col].astype(str) != "TOTAL"].copy()
    if plot_df.empty:
        plot_df = summary_df.copy()

    melted = plot_df.melt(
        id_vars=label_col,
        value_vars=["count_before", "count_after", "count_removed"],
        var_name="stage",
        value_name="count",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x=label_col, y="count", hue="stage", ax=ax)
    ax.set_title("Class Counts Before and After Cleaning", fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_threshold_sweep_results(
    sweep_df: pd.DataFrame,
    metric_col: str = "f1_macro",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Sweet Spot threshold sweep results."""
    if sweep_df.empty:
        raise ValueError("sweep_df is empty.")
    _ensure_columns(sweep_df, ["train_retention_pct", metric_col])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sweep_df,
        x="train_retention_pct",
        y=metric_col,
        hue="filter" if "filter" in sweep_df.columns else None,
        s=80,
        ax=ax,
    )
    ax.set_title("Threshold Sweep Results", fontweight="bold")
    ax.set_xlabel("Training Retention (%)")
    ax.set_ylabel(metric_col)
    ax.grid(axis="both", linestyle="--", alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


# -----------------------------------------------------------------------------
# Split / transform / evaluation plots
# -----------------------------------------------------------------------------


def plot_split_distribution(
    splits: Mapping[str, pd.DataFrame] | None = None,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    label_col: str = "label_name",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot class distribution across dataframe splits."""
    if splits is None:
        splits = {"train": train_df, "val": val_df, "test": test_df}

    records: list[dict[str, Any]] = []

    for split_name, split_df in splits.items():
        if split_df is None or split_df.empty:
            continue
        _ensure_columns(split_df, [label_col])
        counts = split_df[label_col].value_counts(dropna=False)
        for class_name, count in counts.items():
            records.append({"split": split_name, "class": class_name, "count": int(count)})

    if not records:
        raise ValueError("No split distribution records to plot.")

    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="split", y="count", hue="class", ax=ax)
    ax.set_title("Class Distribution Across Splits", fontweight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax



def plot_confusion_matrix(
    cm: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a confusion matrix heatmap from a dataframe or numeric matrix."""
    if isinstance(cm, pd.DataFrame):
        matrix = cm.to_numpy()
        xticklabels = cm.columns.tolist()
        yticklabels = cm.index.tolist()
    else:
        matrix = np.asarray(cm)
        xticklabels = list(labels) if labels is not None else "auto"
        yticklabels = list(labels) if labels is not None else "auto"

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_grid_search_results(
    results_df: pd.DataFrame,
    x: str = "feature_extraction.backbone",
    y: str = "f1_macro",
    hue: str = "classifier.name",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot grid-search comparison results."""
    if results_df.empty:
        raise ValueError("results_df is empty.")
    _ensure_columns(results_df, [x, y])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=results_df,
        x=x,
        y=y,
        hue=hue if hue in results_df.columns else None,
        ax=ax,
    )
    ax.set_title("Grid Search Results", fontweight="bold")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    _finalize_figure(fig, save_path=save_path, show=show)
    return fig, ax


__all__ = [
    "plot_image_grid_from_df",
    "plot_sample_grid",
    "plot_pie_chart",
    "plot_bar_chart",
    "plot_scatter_distribution",
    "plot_rgb_channel_kde",
    "plot_metric_distribution",
    "plot_metric_correlation_heatmap",
    "plot_before_after_cleaning",
    "plot_threshold_sweep_results",
    "plot_split_distribution",
    "plot_confusion_matrix",
    "plot_grid_search_results",
]