"""
Frozen-backbone feature extraction utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules.artifacts import feature_files_exist, load_feature_split, save_feature_split
from modules.backbones import get_backbone, get_feature_dim
from modules.datasets import ImagePathDataset


def _as_device(device: str | torch.device | None = None) -> torch.device:
    """Return a torch.device from a string/device/None value."""
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_positive_int(value: int, name: str, allow_zero: bool = False) -> int:
    """Validate and return a positive integer hyperparameter."""
    int_value = int(value)
    if allow_zero:
        if int_value < 0:
            raise ValueError(f"{name} must be non-negative, got {int_value}.")
    elif int_value <= 0:
        raise ValueError(f"{name} must be positive, got {int_value}.")
    return int_value


def _validate_feature_dataframe(
    df: pd.DataFrame,
    path_col: str,
    label_col: str,
) -> None:
    """Validate dataframe columns required for feature extraction."""
    if path_col not in df.columns:
        raise KeyError(f"path_col not found in dataframe: {path_col}")
    if label_col not in df.columns:
        raise KeyError(f"label_col not found in dataframe: {label_col}")


def _feature_cache_manifest_path(output_dir: str | Path) -> Path:
    """Return the metadata path for cached feature splits."""
    return Path(output_dir) / "feature_manifest.json"


def _build_feature_manifest(
    backbone_name: str,
    pretrained: bool,
    split_sizes: Mapping[str, int],
) -> dict[str, Any]:
    """Build lightweight metadata for cached feature splits."""
    return {
        "backbone_name": str(backbone_name),
        "pretrained": bool(pretrained),
        "feature_dim": int(get_feature_dim(backbone_name)),
        "split_sizes": {str(key): int(value) for key, value in split_sizes.items()},
    }


def _write_feature_manifest(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    """Write feature-cache metadata."""
    path = _feature_cache_manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(dict(manifest), file, ensure_ascii=False, indent=2)


def _read_feature_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    """Read feature-cache metadata when available."""
    path = _feature_cache_manifest_path(output_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _cache_matches_request(
    output_dir: str | Path,
    backbone_name: str,
    pretrained: bool,
    expected_split_sizes: Mapping[str, int],
) -> bool:
    """Return True when cache metadata matches the requested extraction setup.

    If no manifest exists, this returns True for backward compatibility with
    older caches that only contain X_<split>.npy / y_<split>.npy files.
    """
    manifest = _read_feature_manifest(output_dir)
    if manifest is None:
        return True

    expected = _build_feature_manifest(
        backbone_name=backbone_name,
        pretrained=pretrained,
        split_sizes=expected_split_sizes,
    )

    return (
        manifest.get("backbone_name") == expected["backbone_name"]
        and bool(manifest.get("pretrained")) == expected["pretrained"]
        and int(manifest.get("feature_dim", -1)) == expected["feature_dim"]
        and dict(manifest.get("split_sizes", {})) == expected["split_sizes"]
    )


@torch.inference_mode()
def extract_features(
    df: pd.DataFrame,
    transform: Any,
    backbone_name: str,
    batch_size: int = 128,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    path_col: str = "path",
    label_col: str = "label",
    pretrained: bool = True,
    data_parallel: bool = False,
    show_progress: bool = True,
    on_error: str = "raise",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a feature matrix ``X`` and label vector ``y`` from an image dataframe."""
    _validate_feature_dataframe(df, path_col=path_col, label_col=label_col)

    batch_size = _validate_positive_int(batch_size, "batch_size")
    num_workers = _validate_positive_int(num_workers, "num_workers", allow_zero=True)
    device = _as_device(device)

    feature_dim = get_feature_dim(backbone_name)

    if df.empty:
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    if transform is None:
        raise ValueError("transform must not be None for feature extraction.")

    dataset = ImagePathDataset(
        df=df,
        transform=transform,
        path_col=path_col,
        label_col=label_col,
        return_labels=True,
        on_error=on_error,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = get_backbone(
        name=backbone_name,
        device=device,
        pretrained=pretrained,
        freeze=True,
        data_parallel=data_parallel,
    )

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    progress = tqdm(
        loader,
        desc=f"Extracting features ({backbone_name})",
        leave=False,
        disable=not show_progress,
    )

    for images, batch_labels in progress:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        outputs = outputs.flatten(start_dim=1)

        features.append(outputs.detach().cpu().numpy().astype(np.float32, copy=False))
        labels.append(batch_labels.detach().cpu().numpy().astype(np.int64, copy=False))

    if not features:
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X = np.vstack(features)
    y = np.concatenate(labels)

    if X.shape[1] != feature_dim:
        raise ValueError(
            f"Extracted feature dimension mismatch for {backbone_name}: "
            f"expected {feature_dim}, got {X.shape[1]}."
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return X, y


def _load_cached_feature_splits(output_dir: str | Path) -> dict[str, Any]:
    """Load train/val/test feature splits from cache."""
    X_train, y_train = load_feature_split("train", output_dir)
    X_val, y_val = load_feature_split("val", output_dir)
    X_test, y_test = load_feature_split("test", output_dir)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "loaded_from_cache": True,
        "feature_dir": Path(output_dir),
    }


def extract_feature_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform: Any,
    eval_transform: Any,
    backbone_name: str,
    batch_size: int = 128,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    output_dir: str | Path | None = None,
    pretrained: bool = True,
    data_parallel: bool = False,
    force_recompute: bool = False,
    path_col: str = "path",
    label_col: str = "label",
    show_progress: bool = True,
    on_error: str = "raise",
) -> dict[str, Any]:
    """Extract or load cached features for train, validation, and test splits."""
    split_sizes = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)

        if (
            not force_recompute
            and feature_files_exist(output_dir, split_names=("train", "val", "test"))
            and _cache_matches_request(
                output_dir=output_dir,
                backbone_name=backbone_name,
                pretrained=pretrained,
                expected_split_sizes=split_sizes,
            )
        ):
            return _load_cached_feature_splits(output_dir)

    X_train, y_train = extract_features(
        df=train_df,
        transform=train_transform,
        backbone_name=backbone_name,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        path_col=path_col,
        label_col=label_col,
        pretrained=pretrained,
        data_parallel=data_parallel,
        show_progress=show_progress,
        on_error=on_error,
    )

    X_val, y_val = extract_features(
        df=val_df,
        transform=eval_transform,
        backbone_name=backbone_name,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        path_col=path_col,
        label_col=label_col,
        pretrained=pretrained,
        data_parallel=data_parallel,
        show_progress=show_progress,
        on_error=on_error,
    )

    X_test, y_test = extract_features(
        df=test_df,
        transform=eval_transform,
        backbone_name=backbone_name,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        path_col=path_col,
        label_col=label_col,
        pretrained=pretrained,
        data_parallel=data_parallel,
        show_progress=show_progress,
        on_error=on_error,
    )

    if output_dir is not None:
        save_feature_split(X_train, y_train, "train", output_dir)
        save_feature_split(X_val, y_val, "val", output_dir)
        save_feature_split(X_test, y_test, "test", output_dir)
        _write_feature_manifest(
            output_dir,
            _build_feature_manifest(
                backbone_name=backbone_name,
                pretrained=pretrained,
                split_sizes=split_sizes,
            ),
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "loaded_from_cache": False,
        "feature_dir": Path(output_dir) if output_dir is not None else None,
    }


def load_or_extract_feature_splits(
    splits: Mapping[str, pd.DataFrame],
    transforms: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Config-driven wrapper for extracting feature splits.

    Expected ``splits`` keys: ``train``, ``val``, and ``test``.
    Expected ``transforms`` keys: ``train`` and ``eval`` or split-specific keys.
    """
    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            raise KeyError(f"splits must contain '{split_name}'.")

    train_transform = transforms.get("train")
    eval_transform = transforms.get("eval", transforms.get("val", transforms.get("test")))

    if train_transform is None:
        raise KeyError("transforms must contain 'train'.")
    if eval_transform is None:
        raise KeyError("transforms must contain 'eval', 'val', or 'test'.")

    feature_config = dict(config.get("feature_extraction", config))

    return extract_feature_splits(
        train_df=splits["train"],
        val_df=splits["val"],
        test_df=splits["test"],
        train_transform=train_transform,
        eval_transform=eval_transform,
        backbone_name=str(feature_config.get("backbone", "efficientnet_b0")),
        batch_size=int(feature_config.get("batch_size", 128)),
        device=feature_config.get("device"),
        num_workers=int(feature_config.get("num_workers", 0)),
        output_dir=output_dir,
        pretrained=bool(feature_config.get("pretrained", True)),
        data_parallel=bool(feature_config.get("data_parallel", False)),
        force_recompute=bool(feature_config.get("force_recompute", False)),
        show_progress=bool(feature_config.get("show_progress", True)),
        on_error=str(feature_config.get("on_error", "raise")),
    )


__all__ = [
    "extract_features",
    "extract_feature_splits",
    "load_or_extract_feature_splits",
]
