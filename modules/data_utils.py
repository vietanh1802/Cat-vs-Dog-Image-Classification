"""
Dataset discovery, dataframe creation, stratified splits, and summaries.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

try:
    from sklearn.model_selection import train_test_split
except Exception as exc:  # pragma: no cover - sklearn is expected for this project
    raise ImportError("data_utils.py requires scikit-learn. Install it with `pip install scikit-learn`.") from exc


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _normalize_extensions(extensions: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize image extensions to lowercase dotted suffixes.

    Examples
    --------
    ``["jpg", "*.png", ".jpeg"]`` becomes ``(".jpg", ".png", ".jpeg")``.
    """

    if extensions is None:
        extensions = DEFAULT_EXTENSIONS

    normalized: list[str] = []
    for extension in extensions:
        ext = str(extension).strip().lower()
        if not ext:
            continue
        if ext.startswith("*."):
            ext = ext[1:]
        elif not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)

    # Preserve input order while removing duplicates.
    return tuple(dict.fromkeys(normalized))


def _contains_images(root: str | Path, extensions: Iterable[str] | None = None) -> bool:
    """Return True if ``root`` contains at least one supported image file."""

    root = Path(root)
    if not root.exists() or not root.is_dir():
        return False

    suffixes = set(_normalize_extensions(extensions))
    return any(path.is_file() and path.suffix.lower() in suffixes for path in root.rglob("*"))


def _download_kaggle_dataset(dataset_id: str) -> Path:
    """Download a Kaggle dataset using kagglehub and return the local path."""

    try:
        import kagglehub
    except Exception as exc:
        raise ImportError(
            "Downloading Kaggle datasets requires `kagglehub`. "
            "Install it with `pip install kagglehub` or provide local_root."
        ) from exc

    return Path(kagglehub.dataset_download(dataset_id)).expanduser().resolve()


def _default_class_map() -> dict[str, int]:
    """Return a default token-to-label map for cat/dog datasets."""

    return {
        "cat": 0,
        "cats": 0,
        "kitten": 0,
        "kittens": 0,
        "dog": 1,
        "dogs": 1,
        "puppy": 1,
        "puppies": 1,
    }


def _canonical_label_names(class_map: Mapping[str, int]) -> dict[int, str]:
    """Infer readable label names from a class map."""

    preferred = {0: "cat", 1: "dog"}
    labels = sorted(set(int(value) for value in class_map.values()))
    names: dict[int, str] = {}
    for label in labels:
        if label in preferred:
            names[label] = preferred[label]
            continue
        candidates = [token for token, value in class_map.items() if int(value) == label]
        names[label] = sorted(candidates, key=len)[0] if candidates else str(label)
    return names


def _tokenize_path_part(text: str) -> list[str]:
    """Tokenize a path component into lowercase alphanumeric tokens."""

    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def _candidate_label_parts(path: Path) -> list[str]:
    """Return path parts from most label-informative to least informative."""

    return [path.stem, *[parent.name for parent in path.parents if parent.name]]


def _deduplicate_preserve_order(paths: Sequence[Path]) -> list[Path]:
    """Remove duplicate paths while preserving stable sorted order."""

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _coerce_label_name(label: int | None, label_names: Mapping[int, str]) -> str | None:
    """Return the canonical label name for a numeric label."""

    if label is None:
        return None
    return label_names.get(int(label), str(label))


# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------


def resolve_dataset_root(
    dataset_id: str | None = None,
    local_root: str | Path | None = None,
    kaggle_input_dir: str | Path = "/kaggle/input",
    extensions: Iterable[str] | None = None,
    allow_download: bool = True,
) -> Path:
    """Find a dataset root from a local path, Kaggle input, or kagglehub.

    Parameters
    ----------
    dataset_id:
        Kaggle dataset identifier, e.g. ``"tongpython/cat-and-dog"``.
    local_root:
        Existing local dataset directory. This is checked first.
    kaggle_input_dir:
        Kaggle's mounted input directory. Used when running on Kaggle.
    extensions:
        Supported image file extensions.
    allow_download:
        If True and ``dataset_id`` is provided, use kagglehub as a fallback.

    Returns
    -------
    pathlib.Path
        A directory containing image files somewhere underneath it.
    """

    suffixes = _normalize_extensions(extensions)

    if local_root is not None:
        root = Path(local_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"local_root does not exist: {root}")
        if not _contains_images(root, suffixes):
            raise FileNotFoundError(f"No supported image files found under local_root: {root}")
        return root

    kaggle_dir = Path(kaggle_input_dir)
    if kaggle_dir.exists() and kaggle_dir.is_dir():
        # Prefer a folder whose name resembles the dataset slug when possible.
        candidates: list[Path] = []
        if dataset_id:
            slug = dataset_id.split("/")[-1].lower().replace("_", "-")
            candidates.extend(
                child for child in kaggle_dir.iterdir()
                if child.is_dir() and slug in child.name.lower().replace("_", "-")
            )
        candidates.extend(child for child in kaggle_dir.iterdir() if child.is_dir())
        for candidate in candidates:
            if _contains_images(candidate, suffixes):
                return candidate.resolve()

    if dataset_id and allow_download:
        downloaded_root = _download_kaggle_dataset(dataset_id)
        if not _contains_images(downloaded_root, suffixes):
            raise FileNotFoundError(f"Downloaded dataset has no supported images: {downloaded_root}")
        return downloaded_root

    raise FileNotFoundError(
        "Could not resolve dataset root. Provide local_root, run on Kaggle with /kaggle/input, "
        "or set dataset_id with allow_download=True."
    )


def list_image_paths(
    root: str | Path,
    extensions: Iterable[str] | None = None,
) -> list[Path]:
    """Recursively list supported image files under ``root`` in stable order."""

    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    suffixes = set(_normalize_extensions(extensions))
    paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    paths = sorted(paths, key=lambda path: str(path).lower())
    return _deduplicate_preserve_order(paths)


# -----------------------------------------------------------------------------
# Labels and dataframes
# -----------------------------------------------------------------------------


def infer_label_from_path(
    path: str | Path,
    class_map: Mapping[str, int] | None = None,
) -> tuple[int | None, str | None]:
    """Infer ``(label, label_name)`` from filename and parent-folder tokens.

    The function scans from the most specific path component to the least
    specific one: filename stem, nearest parent, then higher parents. This helps
    avoid accidental labels from broad dataset folders such as ``cat-and-dog``.
    """

    path = Path(path)
    normalized_map = {str(token).lower(): int(label) for token, label in (class_map or _default_class_map()).items()}
    label_names = _canonical_label_names(normalized_map)

    for part in _candidate_label_parts(path):
        tokens = _tokenize_path_part(part)
        matched_labels = [normalized_map[token] for token in tokens if token in normalized_map]
        if not matched_labels:
            continue

        unique_labels = sorted(set(matched_labels))
        if len(unique_labels) == 1:
            label = unique_labels[0]
            return label, _coerce_label_name(label, label_names)

    return None, None


def build_raw_dataframe(
    root: str | Path,
    extensions: Iterable[str] | None = None,
    class_map: Mapping[str, int] | None = None,
    drop_unknown: bool = True,
) -> pd.DataFrame:
    """Build an image-index dataframe with path, label, label_name, and metadata columns."""
    image_paths = list_image_paths(root, extensions)
    class_map = class_map or _default_class_map()

    rows: list[dict[str, Any]] = []
    for path in image_paths:
        label, label_name = infer_label_from_path(path, class_map=class_map)
        rows.append(
            {
                "path": str(path),
                "label": label,
                "label_name": label_name,
                "filename": path.name,
                "extension": path.suffix.lower(),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No image files found under root: {Path(root).resolve()}")

    if drop_unknown:
        df = df[df["label"].notna()].copy()
        if df.empty:
            raise ValueError(
                "Image files were found, but no labels could be inferred. "
                "Check class_map or dataset folder/file names."
            )
        df["label"] = df["label"].astype(int)
    else:
        # Nullable integer allows unknown labels to remain as <NA>.
        df["label"] = df["label"].astype("Int64")

    df = df.sort_values(["label", "path"], na_position="last").reset_index(drop=True)
    df.insert(0, "sample_id", range(len(df)))
    return df


# -----------------------------------------------------------------------------
# Splitting and sampling
# -----------------------------------------------------------------------------
def _validate_min_class_count(df: pd.DataFrame, label_col: str, min_count: int = 2) -> None:
    """Validate that every class has enough samples for stratified splitting."""
    counts = df[label_col].value_counts(dropna=False)
    too_small = counts[counts < min_count]

    if not too_small.empty:
        raise ValueError(
            "Stratified split requires enough samples per class. "
            f"Classes with fewer than {min_count} samples: {too_small.to_dict()}"
        )

def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits with class stratification.

    Returns three dataframes with a new ``split`` column. The original dataframe
    is never mutated.
    """

    if df.empty:
        raise ValueError("Cannot split an empty dataframe")
    if label_col not in df.columns:
        raise KeyError(f"label_col not found in dataframe: {label_col}")

    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"Split ratios must be non-negative, got {ratios}")
    if not abs(sum(ratios) - 1.0) <= 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} with sum={sum(ratios):.6f}")
    if train_ratio <= 0:
        raise ValueError("train_ratio must be positive")

    working_df = df.copy().reset_index(drop=True)
    stratify = working_df[label_col]

    if val_ratio == 0 and test_ratio == 0:
        train_df = working_df.copy()
        train_df["split"] = "train"
        empty = working_df.iloc[0:0].copy()
        return train_df.reset_index(drop=True), empty.copy(), empty.copy()

    holdout_ratio = val_ratio + test_ratio
    _validate_min_class_count(working_df, label_col=label_col, min_count=2)
    train_df, holdout_df = train_test_split(
        working_df,
        test_size=holdout_ratio,
        random_state=seed,
        stratify=stratify,
    )

    train_df = train_df.copy()
    train_df["split"] = "train"

    if val_ratio == 0:
        val_df = working_df.iloc[0:0].copy()
        test_df = holdout_df.copy()
        test_df["split"] = "test"
    elif test_ratio == 0:
        val_df = holdout_df.copy()
        val_df["split"] = "val"
        test_df = working_df.iloc[0:0].copy()
    else:
        relative_test_ratio = test_ratio / holdout_ratio
        val_df, test_df = train_test_split(
            holdout_df,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=holdout_df[label_col],
        )
        val_df = val_df.copy()
        test_df = test_df.copy()
        val_df["split"] = "val"
        test_df["split"] = "test"

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def sample_by_class(
    df: pd.DataFrame,
    n_per_class: int = 5,
    label_col: str = "label_name",
    seed: int = 42,
) -> pd.DataFrame:
    """Return up to ``n_per_class`` rows per class in a stable random sample."""

    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")
    if label_col not in df.columns:
        raise KeyError(f"label_col not found in dataframe: {label_col}")

    samples = []
    for _, group in df.groupby(label_col, sort=True, dropna=False):
        n = min(n_per_class, len(group))
        samples.append(group.sample(n=n, random_state=seed))
    return pd.concat(samples, axis=0).reset_index(drop=True) if samples else df.iloc[0:0].copy()


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summarize_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label_name",
) -> pd.DataFrame:
    """Summarize class counts and percentages."""

    if label_col not in df.columns:
        raise KeyError(f"label_col not found in dataframe: {label_col}")

    counts = df[label_col].value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    summary = counts.rename_axis(label_col).reset_index(name="count")
    summary["percentage"] = (summary["count"] / total * 100).round(2) if total else 0.0
    return summary


def summarize_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label_name",
) -> pd.DataFrame:
    """Summarize class distribution across train/validation/test splits."""
    return summarize_splits_distribution(
        {"train": train_df, "val": val_df, "test": test_df},
        label_col=label_col,
    )


def summarize_splits_distribution(
    splits: Mapping[str, pd.DataFrame],
    label_col: str = "label_name",
) -> pd.DataFrame:
    """Summarize class distribution across an arbitrary mapping of splits."""

    frames = []
    for split_name, split_df in splits.items():
        if split_df.empty:
            continue
        summary = summarize_class_distribution(split_df, label_col=label_col)
        summary.insert(0, "split", split_name)
        summary.insert(1, "split_size", len(split_df))
        frames.append(summary)

    if not frames:
        return pd.DataFrame(columns=["split", "split_size", label_col, "count", "percentage"])

    return pd.concat(frames, axis=0, ignore_index=True)


def read_image_metadata(path: str | Path) -> dict[str, Any]:
    """Read basic image metadata without loading the full dataframe pipeline."""

    try:
        with Image.open(path) as img:
            width, height = img.size
            channels = len(img.getbands())
    except Exception:
        return {
            "width": np.nan,
            "height": np.nan,
            "channels": np.nan,
            "aspect_ratio": np.nan,
            "min_side": np.nan,
            "max_side": np.nan,
        }

    aspect_ratio = width / height if height else np.nan
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "aspect_ratio": float(aspect_ratio),
        "min_side": int(min(width, height)),
        "max_side": int(max(width, height)),
    }


def add_image_metadata(
    df: pd.DataFrame,
    path_col: str = "path",
) -> pd.DataFrame:
    """Add width, height, channels, aspect ratio, min_side, and max_side columns."""

    if path_col not in df.columns:
        raise KeyError(f"path_col not found in dataframe: {path_col}")

    result = df.copy()
    metadata_rows = [read_image_metadata(path) for path in result[path_col]]
    metadata_df = pd.DataFrame(metadata_rows, index=result.index)
    return pd.concat([result, metadata_df], axis=1)


def remove_spatial_outliers(
    df: pd.DataFrame,
    cols: tuple[str, str] = ("width", "height"),
    n_std: float = 3.0,
) -> pd.DataFrame:
    """Remove rows outside mean ± n_std * std for selected spatial columns."""
    if n_std <= 0:
        raise ValueError("n_std must be positive")

    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing spatial column(s): {missing_cols}")

    result = df.copy()
    mask = pd.Series(True, index=result.index)

    for col in cols:
        values = pd.to_numeric(result[col], errors="coerce")
        mean = values.mean()
        std = values.std()

        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue

        mask &= values.between(mean - n_std * std, mean + n_std * std)

    return result.loc[mask].reset_index(drop=True)


__all__ = [
    "DEFAULT_EXTENSIONS",
    "resolve_dataset_root",
    "list_image_paths",
    "infer_label_from_path",
    "build_raw_dataframe",
    "stratified_split",
    "sample_by_class",
    "summarize_class_distribution",
    "summarize_split_distribution",
    "summarize_splits_distribution",
    "read_image_metadata",
    "add_image_metadata",
    "remove_spatial_outliers",
]