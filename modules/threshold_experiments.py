"""
Sweet-spot threshold experiment utilities.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules.cleaning import build_cleaning_mask


def get_default_threshold_specs() -> list[dict[str, Any]]:
    """Return single-metric threshold sweeps used in the Sweet Spot notebook."""
    return [
        {"metric": "blur_laplacian", "direction": "gte", "thresholds": [20, 30, 50, 75, 100]},
        {"metric": "min_side", "direction": "gte", "thresholds": [32, 48, 64, 96, 128]},
        {"metric": "near_mono_ratio", "direction": "lte", "thresholds": [0.90, 0.95, 0.98, 0.99]},
        {"metric": "entropy", "direction": "gte", "thresholds": [2.0, 3.0, 4.0, 5.0, 5.5, 6.0]},
        {"metric": "mean_sat", "direction": "gte", "thresholds": [0, 5, 10, 20, 30, 50]},
        {"metric": "compression_artifact", "direction": "lte", "thresholds": [1.2, 1.5, 2.0, 2.5, 3.0, 5.0]},
    ]


def get_default_cleaning_presets() -> dict[str, dict[str, Any]]:
    """Return cleaning presets using the canonical keys expected by cleaning.py.

    The key names intentionally match ``cleaning.build_cleaning_mask``:
    ``min_blur_laplacian``, ``min_side``, ``max_aspect_extremity``,
    ``max_near_mono_ratio``, etc.
    """
    return {
        "valid_only_no_dedup": {
            "enabled": False,
            "remove_corrupted": True,
            "remove_duplicates": False,
        },
        "minimal_safety": {
            "enabled": False,
            "remove_corrupted": True,
            "remove_duplicates": True,
        },
        "conservative": {
            "enabled": True,
            "remove_corrupted": True,
            "remove_duplicates": True,
            "duplicate_hamming_threshold": 4,
            "min_side": 32,
            "max_aspect_extremity": 8.0,
            "min_blur_laplacian": 20.0,
            "min_entropy": 2.0,
            "max_near_mono_ratio": 0.99,
            "max_dark_ratio": 0.99,
            "max_bright_ratio": 0.99,
            "min_mean_sat": 0.0,
            "min_chroma_mean": 0.0,
            "min_center_saliency_ratio": 0.10,
            "max_compression_artifact": 5.0,
        },
        "balanced": {
            "enabled": True,
            "remove_corrupted": True,
            "remove_duplicates": True,
            "duplicate_hamming_threshold": 4,
            "min_side": 64,
            "max_aspect_extremity": 5.0,
            "min_blur_laplacian": 50.0,
            "min_entropy": 3.0,
            "max_near_mono_ratio": 0.95,
            "max_dark_ratio": 0.95,
            "max_bright_ratio": 0.95,
            "min_mean_sat": 5.0,
            "min_chroma_mean": 2.0,
            "min_center_saliency_ratio": 0.15,
            "max_compression_artifact": 3.0,
        },
        "strict": {
            "enabled": True,
            "remove_corrupted": True,
            "remove_duplicates": True,
            "duplicate_hamming_threshold": 4,
            "min_side": 96,
            "max_aspect_extremity": 4.0,
            "min_blur_laplacian": 75.0,
            "min_entropy": 4.0,
            "max_near_mono_ratio": 0.95,
            "max_dark_ratio": 0.95,
            "max_bright_ratio": 0.95,
            "min_mean_sat": 10.0,
            "min_chroma_mean": 5.0,
            "min_center_saliency_ratio": 0.20,
            "max_compression_artifact": 2.5,
        },
    }


def _normalize_direction(direction: str) -> str:
    """Normalize threshold direction aliases."""
    normalized = str(direction).lower().strip()
    aliases = {
        "min": "gte",
        "gte": "gte",
        ">=": "gte",
        "max": "lte",
        "lte": "lte",
        "<=": "lte",
        "gt": "gt",
        ">": "gt",
        "lt": "lt",
        "<": "lt",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported threshold direction: {direction}")
    return aliases[normalized]


def _validate_arrays(X_all: np.ndarray, y_all: np.ndarray, audit_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return feature/label arrays aligned with audit_df row order."""
    X = np.asarray(X_all)
    y = np.asarray(y_all)

    if len(X) != len(y):
        raise ValueError(f"X_all and y_all length mismatch: {len(X)} vs {len(y)}")
    if len(audit_df) != len(y):
        raise ValueError(
            f"audit_df and arrays must be row-aligned. "
            f"len(audit_df)={len(audit_df)}, len(y_all)={len(y)}"
        )
    if len(y) == 0:
        raise ValueError("X_all/y_all must not be empty.")

    return X, y


def _mask_to_array(mask: Any, audit_df: pd.DataFrame) -> np.ndarray:
    """Convert a mask to a boolean NumPy array aligned by audit_df order."""
    if isinstance(mask, pd.Series):
        mask_series = mask.reindex(audit_df.index).fillna(False).astype(bool)
    else:
        mask_series = pd.Series(mask, index=audit_df.index).fillna(False).astype(bool)

    return mask_series.to_numpy(dtype=bool)


def _class_balance_shift(y_before: np.ndarray, y_after: np.ndarray) -> float:
    """Compute maximum absolute change in class proportions."""
    labels = sorted(set(np.asarray(y_before).tolist()).union(set(np.asarray(y_after).tolist())))
    if not labels:
        return float("nan")

    before_counts = pd.Series(y_before).value_counts(normalize=True, dropna=False)
    after_counts = pd.Series(y_after).value_counts(normalize=True, dropna=False)

    return float(max(abs(float(after_counts.get(label, 0.0)) - float(before_counts.get(label, 0.0))) for label in labels))


def _proxy_classifier(classifier_C: float, seed: int) -> Pipeline:
    """Build the fixed proxy classifier used for threshold evaluation."""
    if classifier_C <= 0:
        raise ValueError("classifier_C must be positive.")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=classifier_C, max_iter=1000, random_state=seed)),
    ])


def _invalid_mask_result(
    mask_name: str,
    threshold: Any,
    seed: int,
    n_train: int,
    n_val: int,
    train_retention_frac: float,
    reason: str,
) -> dict[str, Any]:
    """Return a standard result row for invalid proxy-training cases."""
    return {
        "filter": mask_name,
        "threshold": threshold,
        "seed": seed,
        "status": "invalid",
        "invalid_reason": reason,
        "accuracy": np.nan,
        "f1_macro": np.nan,
        "precision_macro": np.nan,
        "recall_macro": np.nan,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "train_retention_pct": round(100 * train_retention_frac, 4),
        "positive_ratio_after": np.nan,
        "imbalance_shift": np.nan,
        "score": np.nan,
    }


def build_single_metric_mask(
    audit_df: pd.DataFrame,
    metric: str,
    threshold: float,
    direction: str,
    base_mask: pd.Series | Sequence[bool] | np.ndarray | None = None,
) -> pd.Series:
    """Build a keep-mask by applying one metric threshold on top of a base mask."""
    direction = _normalize_direction(direction)

    mask = (
        pd.Series(True, index=audit_df.index, dtype=bool)
        if base_mask is None
        else pd.Series(base_mask, index=audit_df.index).fillna(False).astype(bool)
    )

    if metric not in audit_df.columns:
        raise KeyError(f"Metric column not found in audit_df: {metric}")

    values = pd.to_numeric(audit_df[metric], errors="coerce")

    if direction == "gte":
        metric_keep = values >= threshold
    elif direction == "gt":
        metric_keep = values > threshold
    elif direction == "lte":
        metric_keep = values <= threshold
    else:  # direction == "lt"
        metric_keep = values < threshold

    return (mask & metric_keep.fillna(False)).astype(bool)


def make_proxy_split(
    audit_df: pd.DataFrame,
    valid_mask: pd.Series | Sequence[bool] | np.ndarray,
    val_size: float,
    seed: int,
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray]:
    """Create row-position train/validation indices for the fixed proxy protocol."""
    if label_col not in audit_df.columns:
        raise KeyError(f"label_col not found in audit_df: {label_col}")
    if not 0 < float(val_size) < 1:
        raise ValueError("val_size must be in (0, 1).")

    valid_array = _mask_to_array(valid_mask, audit_df)
    valid_positions = np.flatnonzero(valid_array)

    if len(valid_positions) < 2:
        raise ValueError("Not enough valid rows to create a proxy split.")

    valid_labels = audit_df.iloc[valid_positions][label_col].to_numpy()
    class_counts = pd.Series(valid_labels).value_counts(dropna=False)

    if len(class_counts) < 2:
        raise ValueError("Proxy split requires at least two classes.")
    if class_counts.min() < 2:
        raise ValueError(
            "Proxy split requires at least two samples per class. "
            f"Class counts: {class_counts.to_dict()}"
        )

    return train_test_split(
        valid_positions,
        test_size=val_size,
        stratify=valid_labels,
        random_state=seed,
    )


def evaluate_cleaning_mask(
    mask: pd.Series | Sequence[bool] | np.ndarray,
    X_all: np.ndarray,
    y_all: np.ndarray,
    audit_df: pd.DataFrame,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    mask_name: str = "mask",
    threshold: Any = "threshold",
    seed: int = 42,
    classifier_C: float = 0.1,
    retention_penalty: float = 0.10,
    imbalance_penalty: float = 0.05,
    min_train_samples: int = 100,
) -> dict[str, Any]:
    """Evaluate one cleaning mask with a fixed proxy validation protocol.

    The mask is applied to the proxy training subset only. The validation subset
    remains fixed so that candidate masks are comparable.
    """
    X, y = _validate_arrays(X_all, y_all, audit_df)

    train_indices = np.asarray(train_indices, dtype=int)
    val_indices = np.asarray(val_indices, dtype=int)

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("train_indices and val_indices must not be empty.")

    mask_array = _mask_to_array(mask, audit_df)
    train_keep_idx = train_indices[mask_array[train_indices]]
    train_retention_frac = len(train_keep_idx) / max(len(train_indices), 1)

    if len(train_keep_idx) < min_train_samples:
        return _invalid_mask_result(
            mask_name,
            threshold,
            seed,
            n_train=len(train_keep_idx),
            n_val=len(val_indices),
            train_retention_frac=train_retention_frac,
            reason=f"fewer_than_{min_train_samples}_training_samples",
        )

    if len(np.unique(y[train_keep_idx])) < 2:
        return _invalid_mask_result(
            mask_name,
            threshold,
            seed,
            n_train=len(train_keep_idx),
            n_val=len(val_indices),
            train_retention_frac=train_retention_frac,
            reason="single_class_after_filtering",
        )

    classifier = _proxy_classifier(classifier_C=classifier_C, seed=seed)
    classifier.fit(X[train_keep_idx], y[train_keep_idx])
    predictions = classifier.predict(X[val_indices])

    accuracy = accuracy_score(y[val_indices], predictions)
    f1 = f1_score(y[val_indices], predictions, average="macro", zero_division=0)
    precision = precision_score(y[val_indices], predictions, average="macro", zero_division=0)
    recall = recall_score(y[val_indices], predictions, average="macro", zero_division=0)

    positive_ratio_after = float(np.mean(y[train_keep_idx])) if set(np.unique(y)).issubset({0, 1}) else np.nan
    imbalance_shift = _class_balance_shift(y[train_indices], y[train_keep_idx])
    score = float(f1 - retention_penalty * (1 - train_retention_frac) - imbalance_penalty * imbalance_shift)

    return {
        "filter": mask_name,
        "threshold": threshold,
        "seed": int(seed),
        "status": "ok",
        "invalid_reason": "",
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "n_train": int(len(train_keep_idx)),
        "n_val": int(len(val_indices)),
        "train_retention_pct": round(100 * train_retention_frac, 4),
        "positive_ratio_after": positive_ratio_after,
        "imbalance_shift": round(float(imbalance_shift), 6),
        "score": score,
    }


def run_single_metric_threshold_sweep(
    X: np.ndarray,
    y: np.ndarray,
    audit_df: pd.DataFrame,
    threshold_specs: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    base_mask: pd.Series | Sequence[bool] | np.ndarray,
    proxy_config: Mapping[str, Any] | None = None,
    score_config: Mapping[str, Any] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate multiple one-metric threshold candidates."""
    proxy_config = dict(proxy_config or {})
    score_config = dict(score_config or {})
    records: list[dict[str, Any]] = []

    for spec in threshold_specs:
        metric = str(spec["metric"])
        direction = str(spec["direction"])

        for threshold in spec["thresholds"]:
            mask = build_single_metric_mask(
                audit_df=audit_df,
                metric=metric,
                threshold=threshold,
                direction=direction,
                base_mask=base_mask,
            )
            records.append(
                evaluate_cleaning_mask(
                    mask=mask,
                    X_all=X,
                    y_all=y,
                    audit_df=audit_df,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    mask_name=metric,
                    threshold=threshold,
                    seed=seed,
                    classifier_C=float(proxy_config.get("classifier_C", 0.1)),
                    retention_penalty=float(score_config.get("retention_penalty", 0.10)),
                    imbalance_penalty=float(score_config.get("imbalance_penalty", 0.05)),
                    min_train_samples=int(proxy_config.get("min_train_samples", 100)),
                )
            )

    return pd.DataFrame(records)


def evaluate_cleaning_presets(
    X: np.ndarray,
    y: np.ndarray,
    audit_df: pd.DataFrame,
    presets: Mapping[str, Mapping[str, Any]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    proxy_config: Mapping[str, Any] | None = None,
    score_config: Mapping[str, Any] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate named cleaning presets with the fixed proxy protocol."""
    proxy_config = dict(proxy_config or {})
    score_config = dict(score_config or {})
    records: list[dict[str, Any]] = []

    for preset_name, preset_config in presets.items():
        mask = build_cleaning_mask(audit_df, preset_config)
        records.append(
            evaluate_cleaning_mask(
                mask=mask,
                X_all=X,
                y_all=y,
                audit_df=audit_df,
                train_indices=train_indices,
                val_indices=val_indices,
                mask_name=preset_name,
                threshold="preset",
                seed=seed,
                classifier_C=float(proxy_config.get("classifier_C", 0.1)),
                retention_penalty=float(score_config.get("retention_penalty", 0.10)),
                imbalance_penalty=float(score_config.get("imbalance_penalty", 0.05)),
                min_train_samples=int(proxy_config.get("min_train_samples", 100)),
            )
        )

    return pd.DataFrame(records)


def run_cleaning_stability_check(
    X: np.ndarray,
    y: np.ndarray,
    audit_df: pd.DataFrame,
    presets: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    val_size: float,
    proxy_config: Mapping[str, Any] | None = None,
    score_config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate cleaning presets across multiple random proxy splits."""
    if not seeds:
        raise ValueError("seeds must contain at least one seed.")

    _validate_arrays(X, y, audit_df)
    proxy_config = dict(proxy_config or {})
    score_config = dict(score_config or {})

    if "is_corrupted" in audit_df.columns:
        valid_mask = ~audit_df["is_corrupted"].fillna(False).astype(bool)
    else:
        valid_mask = pd.Series(True, index=audit_df.index, dtype=bool)

    records: list[dict[str, Any]] = []

    for seed in seeds:
        train_indices, val_indices = make_proxy_split(
            audit_df=audit_df,
            valid_mask=valid_mask,
            val_size=val_size,
            seed=int(seed),
            label_col=str(proxy_config.get("label_col", "label")),
        )

        baseline = evaluate_cleaning_mask(
            mask=valid_mask,
            X_all=X,
            y_all=y,
            audit_df=audit_df,
            train_indices=train_indices,
            val_indices=val_indices,
            mask_name="baseline_valid_only",
            threshold="none",
            seed=int(seed),
            classifier_C=float(proxy_config.get("classifier_C", 0.1)),
            retention_penalty=float(score_config.get("retention_penalty", 0.10)),
            imbalance_penalty=float(score_config.get("imbalance_penalty", 0.05)),
            min_train_samples=int(proxy_config.get("min_train_samples", 100)),
        )

        baseline_f1 = baseline.get("f1_macro", np.nan)

        for preset_name, preset_config in presets.items():
            mask = build_cleaning_mask(audit_df, preset_config)
            result = evaluate_cleaning_mask(
                mask=mask,
                X_all=X,
                y_all=y,
                audit_df=audit_df,
                train_indices=train_indices,
                val_indices=val_indices,
                mask_name=preset_name,
                threshold="preset",
                seed=int(seed),
                classifier_C=float(proxy_config.get("classifier_C", 0.1)),
                retention_penalty=float(score_config.get("retention_penalty", 0.10)),
                imbalance_penalty=float(score_config.get("imbalance_penalty", 0.05)),
                min_train_samples=int(proxy_config.get("min_train_samples", 100)),
            )
            result["baseline_f1_for_seed"] = baseline_f1
            result["delta_f1_vs_seed_baseline"] = (
                float(result["f1_macro"] - baseline_f1)
                if pd.notna(result["f1_macro"]) and pd.notna(baseline_f1)
                else np.nan
            )
            records.append(result)

    stability_df = pd.DataFrame(records)
    if stability_df.empty:
        raise ValueError("No stability records were produced.")

    stability_summary = (
        stability_df
        .groupby("filter", dropna=False)
        .agg(
            mean_f1=("f1_macro", "mean"),
            std_f1=("f1_macro", "std"),
            mean_delta_f1=("delta_f1_vs_seed_baseline", "mean"),
            std_delta_f1=("delta_f1_vs_seed_baseline", "std"),
            mean_train_retention_pct=("train_retention_pct", "mean"),
            mean_imbalance_shift=("imbalance_shift", "mean"),
            mean_score=("score", "mean"),
            success_rate=("status", lambda values: float((values == "ok").mean())),
        )
        .reset_index()
    )

    return stability_df, stability_summary


def select_cleaning_policy(
    stability_summary: pd.DataFrame,
    presets: Mapping[str, Mapping[str, Any]],
    min_delta_f1: float,
    min_retention_pct: float,
    max_imbalance_shift: float,
    default_policy: str,
) -> tuple[str, dict[str, Any], pd.Series, str]:
    """Select a cleaning preset based on stability, retention, and balance criteria."""
    if stability_summary.empty:
        raise ValueError("stability_summary is empty; cannot select a cleaning policy.")
    if default_policy not in presets:
        raise KeyError(f"default_policy not found in presets: {default_policy}")

    required_columns = {
        "filter",
        "mean_delta_f1",
        "mean_train_retention_pct",
        "mean_imbalance_shift",
        "mean_score",
    }
    missing = required_columns - set(stability_summary.columns)
    if missing:
        raise KeyError(f"Missing required stability_summary columns: {sorted(missing)}")

    candidate_df = stability_summary.copy()
    candidate_df["meets_practical_gain"] = candidate_df["mean_delta_f1"] >= min_delta_f1
    candidate_df["meets_retention"] = candidate_df["mean_train_retention_pct"] >= min_retention_pct
    candidate_df["meets_balance"] = candidate_df["mean_imbalance_shift"] <= max_imbalance_shift

    eligible_df = candidate_df[
        candidate_df["meets_practical_gain"]
        & candidate_df["meets_retention"]
        & candidate_df["meets_balance"]
    ].copy()

    if not eligible_df.empty:
        selected_row = eligible_df.sort_values("mean_score", ascending=False).iloc[0]
        selection_rule = (
            "Selected the highest-scoring preset among candidates that met "
            "F1-gain, retention, and class-balance criteria."
        )
    else:
        fallback_rows = candidate_df[candidate_df["filter"] == default_policy]
        if fallback_rows.empty:
            raise ValueError(f"default_policy '{default_policy}' was not evaluated.")
        selected_row = fallback_rows.iloc[0]
        selection_rule = (
            "No preset met all predefined criteria, so the default policy was selected."
        )

    selected_preset = str(selected_row["filter"])
    return selected_preset, dict(presets[selected_preset]), selected_row, selection_rule


def build_cleaning_report_payload(
    selected_preset: str,
    selected_config: Mapping[str, Any],
    selected_row: pd.Series | Mapping[str, Any],
    selection_rule: str,
    audit_df: pd.DataFrame,
    final_clean_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a JSON-friendly report payload for the selected cleaning policy."""
    selected_row_dict = selected_row.to_dict() if hasattr(selected_row, "to_dict") else dict(selected_row)

    return {
        "selected_preset": selected_preset,
        "selected_config": dict(selected_config),
        "selected_row": selected_row_dict,
        "selection_rule": selection_rule,
        "audit_rows": int(len(audit_df)),
        "final_clean_rows": int(len(final_clean_df)),
        "removed_rows": int(len(removed_df)),
        "removed_pct": round(len(removed_df) / len(audit_df) * 100, 2) if len(audit_df) else 0.0,
        "config": dict(config),
    }


__all__ = [
    "get_default_threshold_specs",
    "get_default_cleaning_presets",
    "build_single_metric_mask",
    "make_proxy_split",
    "evaluate_cleaning_mask",
    "run_single_metric_threshold_sweep",
    "evaluate_cleaning_presets",
    "run_cleaning_stability_check",
    "select_cleaning_policy",
    "build_cleaning_report_payload",
]
