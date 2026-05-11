from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _validate_dataframe_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    """Raise KeyError if required dataframe columns are missing."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required dataframe columns: {missing}")


def _read_pil_image(path: str | Path) -> Image.Image:
    """Read an image as RGB PIL image."""
    return Image.open(path).convert("RGB")


def _validate_batch_size(batch_size: int) -> None:
    """Validate a positive dataloader batch size."""
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")


def _resolve_split_transform(
    split_name: str,
    transforms: Mapping[str, Any],
) -> Any:
    """Resolve transform for a split using train/eval fallback convention."""
    if split_name in transforms:
        return transforms[split_name]

    if split_name == "train" and "train" in transforms:
        return transforms["train"]

    if split_name in {"val", "valid", "validation", "test"}:
        for key in ("eval", "val", "validation", "test"):
            if key in transforms:
                return transforms[key]

    raise KeyError(
        f"No transform found for split '{split_name}'. "
        "Provide transforms per split or use keys {'train', 'eval'}."
    )


class ImagePathDataset(Dataset):
    """Dataset that reads images from paths stored in a dataframe or sequence.

    Parameters
    ----------
    df:
        Optional dataframe containing image paths and labels.
    paths:
        Optional path sequence used when df is not provided.
    labels:
        Optional labels used when df is not provided.
    transform:
        Image transform applied after reading the image.
    return_labels:
        If True, ``__getitem__`` returns ``(image, label)``. If False, returns only image.
    on_error:
        Error policy for unreadable images:
        - ``"raise"``: raise the original exception.
        - ``"fallback"``: return a gray fallback image.
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        paths: Sequence[str | Path] | None = None,
        labels: Sequence[int] | np.ndarray | None = None,
        transform: Any | None = None,
        path_col: str = "path",
        label_col: str = "label",
        return_labels: bool = True,
        fallback_size: int = 224,
        on_error: str = "raise",
    ) -> None:
        if df is not None and paths is not None:
            raise ValueError("Provide either df or paths, not both.")

        if on_error not in {"raise", "fallback"}:
            raise ValueError("on_error must be either 'raise' or 'fallback'.")

        if fallback_size <= 0:
            raise ValueError("fallback_size must be positive.")

        if df is not None:
            required_cols = [path_col]
            if return_labels:
                required_cols.append(label_col)
            _validate_dataframe_columns(df, required_cols)

            self.paths = [str(path) for path in df[path_col].tolist()]
            self.labels = None if not return_labels else df[label_col].to_numpy()
        else:
            if paths is None:
                raise ValueError("Either df or paths must be provided.")

            self.paths = [str(path) for path in paths]
            self.labels = None if not return_labels else np.asarray(labels)

            if return_labels and labels is None:
                raise ValueError("labels must be provided when return_labels=True.")

        if self.labels is not None and len(self.paths) != len(self.labels):
            raise ValueError(
                f"paths and labels must have the same length, got "
                f"{len(self.paths)} paths and {len(self.labels)} labels."
            )

        self.transform = transform
        self.return_labels = return_labels
        self.fallback_size = int(fallback_size)
        self.on_error = on_error

    def __len__(self) -> int:
        return len(self.paths)

    def _read_image(self, path: str) -> Image.Image:
        try:
            return _read_pil_image(path)
        except Exception:
            if self.on_error == "fallback":
                return Image.new(
                    "RGB",
                    (self.fallback_size, self.fallback_size),
                    (128, 128, 128),
                )
            raise

    def __getitem__(self, idx: int):
        image = self._read_image(self.paths[idx])

        if self.transform is not None:
            image = self.transform(image)

        if not self.return_labels:
            return image

        return image, int(self.labels[idx])


class NpyBatchDataset(Dataset):
    """Dataset backed by saved X_<split>.npy and y_<split>.npy feature files."""

    def __init__(
        self,
        batch_dir: str | Path,
        split_name: str,
        mmap_mode: str | None = "r",
        return_tensors: bool = True,
    ) -> None:
        batch_dir = Path(batch_dir)
        x_path = batch_dir / f"X_{split_name}.npy"
        y_path = batch_dir / f"y_{split_name}.npy"

        if not x_path.exists():
            raise FileNotFoundError(f"Missing feature file: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing label file: {y_path}")

        self.x = np.load(x_path, allow_pickle=False, mmap_mode=mmap_mode)
        self.y = np.load(y_path, allow_pickle=False, mmap_mode=mmap_mode)

        if len(self.x) != len(self.y):
            raise ValueError(
                f"Feature and label lengths differ for split '{split_name}': "
                f"len(X)={len(self.x)}, len(y)={len(self.y)}"
            )

        self.return_tensors = return_tensors

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x_item = self.x[idx]
        y_item = self.y[idx]

        if self.return_tensors:
            x_item = torch.as_tensor(x_item, dtype=torch.float32)
            y_item = torch.as_tensor(y_item, dtype=torch.long)

        return x_item, y_item


def create_image_dataloader(
    df: pd.DataFrame,
    transform: Any,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    path_col: str = "path",
    label_col: str = "label",
    return_labels: bool = True,
    on_error: str = "raise",
) -> DataLoader:
    """Create a DataLoader for an image dataframe."""
    _validate_batch_size(batch_size)

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    dataset = ImagePathDataset(
        df=df,
        transform=transform,
        path_col=path_col,
        label_col=label_col,
        return_labels=return_labels,
        on_error=on_error,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_image_dataloaders(
    splits: Mapping[str, pd.DataFrame],
    transforms: Mapping[str, Any],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    path_col: str = "path",
    label_col: str = "label",
    return_labels: bool = True,
    on_error: str = "raise",
) -> dict[str, DataLoader]:
    """Create image dataloaders for named dataframe splits.

    ``transforms`` may contain one transform per split, or the common convention:
    ``{"train": train_transform, "eval": eval_transform}``.
    """
    if not splits:
        raise ValueError("splits must contain at least one dataframe.")

    loaders: dict[str, DataLoader] = {}

    for split_name, split_df in splits.items():
        transform = _resolve_split_transform(split_name, transforms)
        loaders[split_name] = create_image_dataloader(
            df=split_df,
            transform=transform,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            path_col=path_col,
            label_col=label_col,
            return_labels=return_labels,
            on_error=on_error,
        )

    return loaders


def create_npy_batch_dataloader(
    batch_dir: str | Path,
    split_name: str,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    mmap_mode: str | None = "r",
    return_tensors: bool = True,
) -> DataLoader:
    """Create a DataLoader for saved NumPy feature splits."""
    _validate_batch_size(batch_size)

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    dataset = NpyBatchDataset(
        batch_dir=batch_dir,
        split_name=split_name,
        mmap_mode=mmap_mode,
        return_tensors=return_tensors,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


__all__ = [
    "ImagePathDataset",
    "NpyBatchDataset",
    "create_image_dataloader",
    "create_image_dataloaders",
    "create_npy_batch_dataloader",
]