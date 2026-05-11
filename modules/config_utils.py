from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from modules.config_types import (
    DEFAULT_SEED,
    PROJECT_NAME,
    SUPPORTED_BACKBONES,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_PREPROCESSING_MODES,
    FullConfig,
)

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def get_device() -> Any:
    if torch is None:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(*dirs: str | Path | None) -> None:
    for directory in dirs:
        if directory is None:
            continue
        Path(directory).expanduser().mkdir(parents=True, exist_ok=True)


def resolve_workspace(
    workspace: str | Path | None = None,
    project_name: str = PROJECT_NAME,
) -> Path:
    if workspace is not None:
        return Path(workspace).expanduser().resolve()

    env_workspace = os.environ.get("ML_PROJECT_WORKSPACE")
    if env_workspace:
        return Path(env_workspace).expanduser().resolve()

    if Path("/content").exists():
        return Path("/content") / project_name

    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / project_name

    return (Path.cwd() / project_name).resolve()


def get_default_config(user_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    workspace = resolve_workspace(project_name=PROJECT_NAME)
    return FullConfig.from_dict(user_config, default_workspace=workspace).to_dict()


def validate_config(config: Mapping[str, Any]) -> None:
    # Re-parse through dataclasses. If invalid, this raises a clear error.
    FullConfig.from_dict(config, default_workspace=config.get("paths", {}).get("workspace", resolve_workspace()))


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(config), file, ensure_ascii=False, indent=2, allow_nan=False)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def config_to_run_name(config: Mapping[str, Any]) -> str:
    parts = [
        str(config.get("preprocessing", {}).get("mode", "preprocess")),
        str(config.get("feature_extraction", {}).get("backbone", "backbone")),
        str(config.get("classifier", {}).get("name", "classifier")),
    ]
    raw_name = "__".join(parts).lower()
    safe_chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in raw_name]
    return "".join(safe_chars).strip("_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def validate_and_normalize(name: str, supported_list: tuple[str, ...], entity_name: str = "option") -> str:
    """Normalize a string and validate it against a supported list."""
    normalized = name.lower().strip()
    if normalized not in supported_list:
        raise ValueError(
            f"Unsupported {entity_name}: '{name}'. "
            f"Supported {entity_name}s: {list(supported_list)}"
        )
    return normalized


__all__ = [
    "PROJECT_NAME",
    "DEFAULT_SEED",
    "SUPPORTED_PREPROCESSING_MODES",
    "SUPPORTED_BACKBONES",
    "SUPPORTED_CLASSIFIERS",
    "set_seed",
    "get_device",
    "ensure_dirs",
    "resolve_workspace",
    "get_default_config",
    "validate_config",
    "config_to_run_name",
    "save_config",
    "load_config",
    "validate_and_normalize",
]