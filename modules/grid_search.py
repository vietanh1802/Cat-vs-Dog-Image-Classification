"""
Grid-search utilities for the classical image-classification pipeline.
"""

from __future__ import annotations

import copy
import re
import time
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from modules.artifacts import save_dataframe
from modules.classical_models import train_classifier
from modules.evaluation import evaluate_estimator
from modules.feature_extraction import extract_feature_splits
from modules.transforms import get_hybrid_transform


def _as_list(value: Any) -> list[Any]:
    """Return value as a list, treating strings/scalars as one item."""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value using a dotted key such as 'classifier.name'."""
    keys = str(dotted_key).split(".")
    cursor = config

    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]

    cursor[keys[-1]] = copy.deepcopy(value)


def _safe_slug(value: Any) -> str:
    """Convert an arbitrary value into a short filesystem-safe slug."""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    return text.strip("-") or "value"


def _experiment_run_name(experiment_config: Mapping[str, Any], index: int) -> str:
    """Build a stable run name from a grid experiment config."""
    parts = [f"{_safe_slug(key)}-{_safe_slug(value)}" for key, value in experiment_config.items()]
    return f"{index:03d}__" + "__".join(parts)


def _feature_cache_key(config: Mapping[str, Any]) -> str:
    """Build a feature-cache key from feature-relevant config values only."""
    pre_cfg = config["preprocessing"]
    feat_cfg = config["feature_extraction"]

    parts = [
        f"mode-{_safe_slug(pre_cfg.get('mode', 'preprocess'))}",
        f"size-{_safe_slug(pre_cfg.get('image_size', 224))}",
        f"norm-{_safe_slug(pre_cfg.get('normalize', 'imagenet'))}",
        f"backbone-{_safe_slug(feat_cfg.get('backbone', 'backbone'))}",
        f"pretrained-{_safe_slug(feat_cfg.get('pretrained', True))}",
    ]

    return "__".join(parts)


def _build_feature_transforms(preprocessing_config: Mapping[str, Any]) -> tuple[Any, Any]:
    """Build deterministic transforms for frozen feature extraction."""
    mode = preprocessing_config.get("mode", "letterbox")
    image_size = preprocessing_config.get("image_size", 224)

    # Frozen feature extraction should be deterministic; augmentation belongs to
    # deep_learning.py, not grid-search over cached classical features.
    train_transform = get_hybrid_transform(mode, image_size=image_size, train=False)
    eval_transform = get_hybrid_transform(mode, image_size=image_size, train=False)
    return train_transform, eval_transform


def _sort_ascending_for_column(column: str, primary_metric: str) -> bool:
    """Return sort direction for ranking columns."""
    if column == primary_metric:
        return False

    lowered = column.lower()
    if any(token in lowered for token in ("time", "seconds", "duration", "error", "loss")):
        return True

    return False


def generate_experiment_grid(search_space: Mapping[str, Sequence[Any] | Any]) -> list[dict[str, Any]]:
    """Generate all experiment override dictionaries from a search space.

    Search-space keys should usually be dotted config keys, for example:
    ``"preprocessing.mode"``, ``"feature_extraction.backbone"``,
    ``"classifier.name"``, or ``"classifier.params.C"``.
    """
    if not search_space:
        raise ValueError("search_space must not be empty.")

    keys = list(search_space.keys())
    values = [_as_list(search_space[key]) for key in keys]

    empty_keys = [key for key, options in zip(keys, values) if len(options) == 0]
    if empty_keys:
        raise ValueError(f"Search-space keys have no candidate values: {empty_keys}")

    return [dict(zip(keys, combo)) for combo in product(*values)]


def merge_experiment_config(
    base_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deep-copied config with experiment overrides applied.

    Dotted keys are applied recursively. Non-dotted keys whose values are dicts
    are shallowly merged into the matching top-level config section.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base_config))

    for key, value in experiment_config.items():
        key = str(key)

        if "." in key:
            _set_by_dotted_key(merged, key, value)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **copy.deepcopy(dict(value))}
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def run_single_classical_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Mapping[str, Any],
    feature_dir: str | Path,
    device: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Run one classical experiment and evaluate on the validation split."""
    start_time = time.perf_counter()

    pre_cfg = config["preprocessing"]
    feat_cfg = config["feature_extraction"]
    clf_cfg = config["classifier"]

    train_transform, eval_transform = _build_feature_transforms(pre_cfg)

    feature_result = extract_feature_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_transform=train_transform,
        eval_transform=eval_transform,
        backbone_name=str(feat_cfg["backbone"]),
        batch_size=int(feat_cfg.get("batch_size", 128)),
        device=device or feat_cfg.get("device"),
        num_workers=int(feat_cfg.get("num_workers", 0)),
        output_dir=feature_dir,
        pretrained=bool(feat_cfg.get("pretrained", True)),
        data_parallel=bool(feat_cfg.get("data_parallel", False)),
        force_recompute=bool(feat_cfg.get("force_recompute", False)),
    )

    model = train_classifier(
        feature_result["X_train"],
        feature_result["y_train"],
        classifier_name=str(clf_cfg["name"]),
        seed=int(config.get("seed", 42)),
        **dict(clf_cfg.get("params", {})),
    )

    validation_metrics = evaluate_estimator(
        model,
        feature_result["X_val"],
        feature_result["y_val"],
        model_name=run_name or str(clf_cfg["name"]),
    )

    fit_time_seconds = time.perf_counter() - start_time

    return {
        "run_name": run_name,
        "model": model,
        "validation_metrics": validation_metrics,
        "feature_result": feature_result,
        "loaded_from_cache": bool(feature_result.get("loaded_from_cache", False)),
        "fit_time_seconds": round(fit_time_seconds, 6),
    }


def run_grid_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_config: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any] | Any],
    output_dir: str | Path,
    device: str | None = None,
    fail_fast: bool = True,
) -> pd.DataFrame:
    """Run a grid search and save ``grid_results.csv`` to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    experiments = generate_experiment_grid(search_space)

    for index, experiment in enumerate(experiments):
        run_name = _experiment_run_name(experiment, index)
        config = merge_experiment_config(base_config, experiment)
        feature_dir = output_dir / "features" / _feature_cache_key(config)

        try:
            result = run_single_classical_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                config=config,
                feature_dir=feature_dir,
                device=device,
                run_name=run_name,
            )

            record = {
                "run_name": run_name,
                "status": "ok",
                **experiment,
                **result["validation_metrics"],
                "loaded_from_cache": result["loaded_from_cache"],
                "fit_time_seconds": result["fit_time_seconds"],
                "feature_dir": str(feature_dir),
            }

        except Exception as exc:
            if fail_fast:
                raise

            record = {
                "run_name": run_name,
                "status": "failed",
                **experiment,
                "error": repr(exc),
            }

        records.append(record)

    results_df = pd.DataFrame(records)
    save_dataframe(results_df, output_dir / "grid_results.csv")
    return results_df


def rank_grid_results(
    results_df: pd.DataFrame,
    primary_metric: str = "f1_macro",
    tie_breakers: list[str] | None = None,
) -> pd.DataFrame:
    """Rank successful grid-search results by metric and optional tie-breakers."""
    if results_df.empty:
        raise ValueError("results_df is empty; cannot rank grid results.")

    ranked = results_df.copy()
    if "status" in ranked.columns:
        ranked = ranked[ranked["status"] == "ok"].copy()

    if ranked.empty:
        raise ValueError("No successful grid-search results to rank.")

    if primary_metric not in ranked.columns:
        raise ValueError(f"primary_metric not found in results_df: {primary_metric}")

    sort_columns = [primary_metric]
    if tie_breakers:
        sort_columns.extend([col for col in tie_breakers if col in ranked.columns and col != primary_metric])

    ascending = [_sort_ascending_for_column(col, primary_metric) for col in sort_columns]
    return ranked.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def select_default_config(
    ranked_results: pd.DataFrame,
    base_config: Mapping[str, Any],
    config_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the default config from the best ranked grid-search row.

    Only experiment override columns are merged back. Metric columns such as
    ``f1_macro`` or ``fit_time_seconds`` are intentionally ignored.
    """
    if ranked_results.empty:
        raise ValueError("ranked_results is empty; cannot select default config.")

    best = ranked_results.iloc[0]

    if config_columns is None:
        config_columns = [col for col in ranked_results.columns if "." in str(col)]

    if not config_columns:
        raise ValueError(
            "Could not infer experiment config columns. "
            "Pass config_columns=list(search_space.keys())."
        )

    experiment_config = {
        col: best[col]
        for col in config_columns
        if col in ranked_results.columns and pd.notna(best[col])
    }

    return merge_experiment_config(base_config, experiment_config)


# Backward-compatible alias with a clearer name available for newer notebooks.
select_default_config_from_grid = select_default_config


__all__ = [
    "generate_experiment_grid",
    "merge_experiment_config",
    "run_single_classical_experiment",
    "run_grid_search",
    "rank_grid_results",
    "select_default_config",
    "select_default_config_from_grid",
]
