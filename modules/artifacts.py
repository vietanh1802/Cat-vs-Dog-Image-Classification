from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    return obj


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save JSON data, converting common ML types into JSON-safe values."""
    path = Path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(data), file, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Save a dataframe to .csv or .parquet."""
    path = Path(path)
    _ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix == ".parquet":
        df.to_parquet(path, index=index)
    else:
        raise ValueError(f"Unsupported dataframe format: {suffix}")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a dataframe from .csv or .parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataframe format: {suffix}")


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array to .npy, creating parent folders if needed."""
    path = Path(path)
    _ensure_parent(path)
    np.save(path, array)


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from .npy without allowing pickles."""
    return np.load(Path(path), allow_pickle=False)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save a trusted Python object to pickle."""
    path = Path(path)
    _ensure_parent(path)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def save_feature_split(X: np.ndarray, y: np.ndarray, split_name: str, output_dir: str | Path) -> dict[str, Path]:
    """Save a feature split using the X_<split>.npy / y_<split>.npy convention."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_path = output_dir / f"X_{split_name}.npy"
    y_path = output_dir / f"y_{split_name}.npy"
    save_numpy(X, x_path)
    save_numpy(y, y_path)
    return {"X": x_path, "y": y_path}


def load_feature_split(split_name: str, feature_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved feature split."""
    feature_dir = Path(feature_dir)
    x_path = feature_dir / f"X_{split_name}.npy"
    y_path = feature_dir / f"y_{split_name}.npy"
    return load_numpy(x_path), load_numpy(y_path)


def feature_files_exist(feature_dir: str | Path, split_names: tuple[str, ...] = ("train", "val", "test")) -> bool:
    """Check whether cached feature files exist for all requested splits."""
    feature_dir = Path(feature_dir)
    return all(
        (feature_dir / f"X_{split}.npy").exists() and (feature_dir / f"y_{split}.npy").exists()
        for split in split_names
    )


__all__ = [
    "save_json",
    "load_json",
    "save_dataframe",
    "load_dataframe",
    "save_numpy",
    "load_numpy",
    "save_pickle",
    "save_feature_split",
    "load_feature_split",
    "feature_files_exist"
]