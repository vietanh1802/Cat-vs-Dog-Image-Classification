"""
Image-cleaning utilities for the cat/dog image-classification project.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_REASON_MAP: dict[str, str] = {
    "flag_corrupted": "corrupted",
    "flag_duplicate": "near_duplicate",
    "flag_too_small": "too_small",
    "flag_extreme_aspect": "extreme_aspect",
    "flag_blurry": "blurry",
    "flag_low_entropy": "low_entropy",
    "flag_near_mono": "near_monochrome",
    "flag_too_dark": "too_dark",
    "flag_too_bright": "too_bright",
    "flag_low_saturation": "low_saturation",
    "flag_low_chroma": "low_chroma",
    "flag_low_center_saliency": "low_center_saliency",
    "flag_compression_artifact": "compression_artifact",
}

_FLAG_COLUMNS = tuple(_REASON_MAP.keys())

_QUALITY_WEIGHTS: dict[str, float] = {
    "min_side": 0.10,
    "blur_laplacian": 0.18,
    "entropy": 0.16,
    "mean_sat": 0.08,
    "chroma_mean": 0.08,
    "center_saliency_ratio": 0.10,
    "aspect_extremity": 0.08,
    "near_mono_ratio": 0.08,
    "dark_ratio": 0.04,
    "bright_ratio": 0.04,
    "compression_artifact": 0.06,
}

_HIGHER_IS_BETTER = (
    "min_side",
    "blur_laplacian",
    "entropy",
    "mean_sat",
    "chroma_mean",
    "center_saliency_ratio",
)

_LOWER_IS_BETTER = (
    "aspect_extremity",
    "near_mono_ratio",
    "dark_ratio",
    "bright_ratio",
    "compression_artifact",
)


# -----------------------------------------------------------------------------
# Small internal helpers
# -----------------------------------------------------------------------------


@dataclass
class _UnionFind:
    """Minimal union-find implementation used for duplicate clustering."""

    parent: dict[Hashable, Hashable]
    rank: dict[Hashable, int]

    @classmethod
    def from_items(cls, items: Iterable[Hashable]) -> "_UnionFind":
        item_list = list(items)
        return cls(
            parent={item: item for item in item_list},
            rank={item: 0 for item in item_list},
        )

    def find(self, item: Hashable) -> Hashable:
        """Return the representative item for a set."""
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: Hashable, right: Hashable) -> None:
        """Merge the two sets containing ``left`` and ``right``."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _boolean_series(index: pd.Index, value: bool = False) -> pd.Series:
    """Create a boolean series aligned with ``index``."""
    return pd.Series(value, index=index, dtype=bool)


def _numeric_series(
    df: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    """Return a numeric column aligned to ``df.index`` or a default series."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    """Read a numeric threshold from a config with a safe default."""
    value = config.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _config_int(config: Mapping[str, Any], key: str, default: int) -> int:
    """Read an integer value from a config with a safe default."""
    value = config.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rank_score(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    """Return percentile rank scores in [0, 1] for a quality metric."""
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)

    if col not in df.columns:
        return pd.Series(0.5, index=df.index, dtype="float64")

    values = pd.to_numeric(df[col], errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.5, index=df.index, dtype="float64")

    values = values.fillna(values.median())
    return values.rank(method="average", pct=True, ascending=ascending).astype(float)


def _valid_hex_hash(value: Any) -> str | None:
    """Normalize a perceptual hash value to a lowercase hexadecimal string."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    try:
        int(text, 16)
    except ValueError:
        return None

    return text


def _hamming_distance_hex(hash_a: Any, hash_b: Any) -> int:
    """Compute Hamming distance between two hexadecimal hash strings."""
    left = _valid_hex_hash(hash_a)
    right = _valid_hex_hash(hash_b)
    if left is None or right is None:
        return 10**9

    width = max(len(left), len(right))
    left_int = int(left.zfill(width), 16)
    right_int = int(right.zfill(width), 16)
    return int((left_int ^ right_int).bit_count())


def _add_false_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all known cleaning flag columns set to False."""
    result = df.copy()
    for column in _FLAG_COLUMNS:
        result[column] = False
    return result


def _ensure_duplicate_columns(
    audit_df: pd.DataFrame,
    cleaning_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Return a dataframe with duplicate-decision columns when requested."""
    required = {
        "is_near_duplicate",
        "is_duplicate_representative",
        "is_duplicate_to_remove",
    }

    if not bool(cleaning_config.get("remove_duplicates", False)):
        return audit_df.copy()

    if required.issubset(audit_df.columns):
        return audit_df.copy()

    threshold = _config_int(cleaning_config, "duplicate_hamming_threshold", 4)
    return mark_near_duplicates(audit_df, hamming_threshold=threshold)


def _select_duplicate_representative(group_frame: pd.DataFrame) -> Hashable:
    """Choose the best representative index for one duplicate group."""
    sort_frame = group_frame.copy()
    by = ["quality_score"]
    ascending = [False]

    if "path" in sort_frame.columns:
        sort_frame["_path_sort_key"] = sort_frame["path"].map(str)
        by.append("_path_sort_key")
        ascending.append(True)

    return sort_frame.sort_values(by=by, ascending=ascending, kind="mergesort").index[0]


def _collect_row_reasons(row: pd.Series) -> list[str]:
    """Collect ordered removal reasons from a row of flag columns."""
    return [reason for flag, reason in _REASON_MAP.items() if bool(row.get(flag, False))]


# -----------------------------------------------------------------------------
# Quality score and duplicate detection
# -----------------------------------------------------------------------------


def compute_quality_score(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Add a deterministic ``quality_score`` column used for duplicate keepers.

    Higher is better. The score is a lightweight heuristic for selecting the
    best representative among near-duplicate images, not a model performance
    metric.
    """
    result = audit_df.copy()
    if result.empty:
        result["quality_score"] = pd.Series(dtype="float64", index=result.index)
        return result

    components = pd.DataFrame(index=result.index)

    for column in _HIGHER_IS_BETTER:
        components[column] = _rank_score(result, column, ascending=True)

    for column in _LOWER_IS_BETTER:
        components[column] = _rank_score(result, column, ascending=False)

    score = pd.Series(0.0, index=result.index, dtype="float64")
    for column, weight in _QUALITY_WEIGHTS.items():
        score = score + components.get(column, pd.Series(0.5, index=result.index)) * weight

    if "is_corrupted" in result.columns:
        corrupted = result["is_corrupted"].fillna(False).astype(bool)
        score = score.mask(corrupted, -1.0)

    result["quality_score"] = score.round(6)
    return result


def mark_near_duplicates(
    audit_df: pd.DataFrame,
    hamming_threshold: int = 4,
    hash_col: str = "phash",
    bands: int = 8,
) -> pd.DataFrame:
    """Detect near-duplicate images using perceptual hashes.

    Returns a copy with ``duplicate_cluster_id``, ``duplicate_group_size``,
    ``is_near_duplicate``, ``is_duplicate_representative``, and
    ``is_duplicate_to_remove``.
    """
    if hamming_threshold < 0:
        raise ValueError("hamming_threshold must be non-negative.")
    if bands <= 0:
        raise ValueError("bands must be positive.")

    result = compute_quality_score(audit_df)

    defaults: dict[str, Any] = {
        "duplicate_cluster_id": -1,
        "duplicate_group_size": 1,
        "is_near_duplicate": False,
        "is_duplicate_representative": False,
        "is_duplicate_to_remove": False,
    }
    for column, default in defaults.items():
        result[column] = default

    if result.empty or hash_col not in result.columns:
        return result

    valid_hashes: dict[Hashable, str] = {}
    for idx, value in result[hash_col].items():
        normalized = _valid_hex_hash(value)
        if normalized is not None:
            valid_hashes[idx] = normalized

    if len(valid_hashes) < 2:
        return result

    uf = _UnionFind.from_items(valid_hashes.keys())
    buckets: dict[tuple[int, str], list[Hashable]] = defaultdict(list)

    for idx, hash_value in valid_hashes.items():
        chunk_size = max(1, int(np.ceil(len(hash_value) / bands)))
        for band_idx in range(bands):
            start = band_idx * chunk_size
            if start >= len(hash_value):
                break
            buckets[(band_idx, hash_value[start : start + chunk_size])].append(idx)

    compared_pairs: set[frozenset[Hashable]] = set()
    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue

        for left_pos, left_idx in enumerate(bucket_indices[:-1]):
            for right_idx in bucket_indices[left_pos + 1 :]:
                pair = frozenset((left_idx, right_idx))
                if pair in compared_pairs:
                    continue
                compared_pairs.add(pair)

                distance = _hamming_distance_hex(valid_hashes[left_idx], valid_hashes[right_idx])
                if distance <= hamming_threshold:
                    uf.union(left_idx, right_idx)

    groups: dict[Hashable, list[Hashable]] = defaultdict(list)
    for idx in valid_hashes:
        groups[uf.find(idx)].append(idx)

    cluster_id = 0
    for group_indices in groups.values():
        if len(group_indices) < 2:
            continue

        cluster_id += 1
        group_frame = result.loc[group_indices]
        representative_idx = _select_duplicate_representative(group_frame)
        duplicate_indices = [idx for idx in group_indices if idx != representative_idx]

        result.loc[group_indices, "duplicate_cluster_id"] = cluster_id
        result.loc[group_indices, "duplicate_group_size"] = len(group_indices)
        result.loc[group_indices, "is_near_duplicate"] = True
        result.loc[representative_idx, "is_duplicate_representative"] = True
        result.loc[duplicate_indices, "is_duplicate_to_remove"] = True

    return result


# -----------------------------------------------------------------------------
# Cleaning flags and decisions
# -----------------------------------------------------------------------------


def compute_soft_flags(
    audit_df: pd.DataFrame,
    cleaning_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Add threshold-based cleaning flags without dropping rows.

    Missing metric columns are treated as non-failing. If
    ``cleaning_config['enabled']`` is False, all flags are set to False.
    """
    result = audit_df.copy()
    if result.empty:
        return _add_false_flags(result)

    enabled = bool(cleaning_config.get("enabled", True))
    if not enabled:
        return _add_false_flags(result)

    index = result.index
    corrupted = result.get("is_corrupted", _boolean_series(index, False)).fillna(False).astype(bool)
    duplicate = result.get("is_duplicate_to_remove", _boolean_series(index, False)).fillna(False).astype(bool)

    result["flag_corrupted"] = corrupted & bool(cleaning_config.get("remove_corrupted", True))
    result["flag_duplicate"] = duplicate & bool(cleaning_config.get("remove_duplicates", False))

    result["flag_too_small"] = _numeric_series(result, "min_side") < _config_float(
        cleaning_config, "min_side", -np.inf
    )
    result["flag_extreme_aspect"] = _numeric_series(result, "aspect_extremity") > _config_float(
        cleaning_config, "max_aspect_extremity", np.inf
    )
    result["flag_blurry"] = _numeric_series(result, "blur_laplacian") < _config_float(
        cleaning_config, "min_blur_laplacian", -np.inf
    )
    result["flag_low_entropy"] = _numeric_series(result, "entropy") < _config_float(
        cleaning_config, "min_entropy", -np.inf
    )
    result["flag_near_mono"] = _numeric_series(result, "near_mono_ratio") > _config_float(
        cleaning_config, "max_near_mono_ratio", np.inf
    )
    result["flag_too_dark"] = _numeric_series(result, "dark_ratio") > _config_float(
        cleaning_config, "max_dark_ratio", np.inf
    )
    result["flag_too_bright"] = _numeric_series(result, "bright_ratio") > _config_float(
        cleaning_config, "max_bright_ratio", np.inf
    )
    result["flag_low_saturation"] = _numeric_series(result, "mean_sat") < _config_float(
        cleaning_config, "min_mean_sat", -np.inf
    )
    result["flag_low_chroma"] = _numeric_series(result, "chroma_mean") < _config_float(
        cleaning_config, "min_chroma_mean", -np.inf
    )
    result["flag_low_center_saliency"] = _numeric_series(
        result, "center_saliency_ratio"
    ) < _config_float(cleaning_config, "min_center_saliency_ratio", -np.inf)
    result["flag_compression_artifact"] = _numeric_series(
        result, "compression_artifact"
    ) > _config_float(cleaning_config, "max_compression_artifact", np.inf)

    result[list(_FLAG_COLUMNS)] = result[list(_FLAG_COLUMNS)].fillna(False).astype(bool)
    return result


def build_cleaning_mask(
    audit_df: pd.DataFrame,
    cleaning_config: Mapping[str, Any],
) -> pd.Series:
    """Return a boolean keep-mask where True means the image is retained."""
    if audit_df.empty:
        return pd.Series(dtype=bool, index=audit_df.index)

    working_df = _ensure_duplicate_columns(audit_df, cleaning_config)
    flagged_df = compute_soft_flags(working_df, cleaning_config)
    remove_mask = flagged_df[list(_FLAG_COLUMNS)].any(axis=1).astype(bool)
    return (~remove_mask).reindex(audit_df.index, fill_value=True)


def assign_removal_reasons(
    audit_df: pd.DataFrame,
    cleaning_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Add ``keep``, ``removal_reason``, and ``removal_reasons`` columns."""
    working_df = _ensure_duplicate_columns(audit_df, cleaning_config)
    result = compute_soft_flags(working_df, cleaning_config)

    row_reasons = result.apply(_collect_row_reasons, axis=1)
    result["keep"] = row_reasons.map(len).eq(0)
    result["removal_reason"] = row_reasons.map(lambda reasons: reasons[0] if reasons else "kept")
    result["removal_reasons"] = row_reasons.map(lambda reasons: "|".join(reasons))
    return result


def apply_cleaning(
    audit_df: pd.DataFrame,
    cleaning_config: Mapping[str, Any],
    reset_index: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a cleaning policy and return ``(clean_df, removed_df)``.

    The input dataframe is never mutated. By default, original indices are
    preserved so cleaned/removed rows remain traceable to the audit dataframe.
    """
    decision_df = assign_removal_reasons(audit_df, cleaning_config)

    clean_df = decision_df[decision_df["keep"]].copy()
    removed_df = decision_df[~decision_df["keep"]].copy()

    if reset_index:
        clean_df = clean_df.reset_index(drop=True)
        removed_df = removed_df.reset_index(drop=True)

    return clean_df, removed_df


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def summarize_cleaning(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    label_col: str = "label_name",
) -> pd.DataFrame:
    """Summarize before/after cleaning counts and removal rates by class."""
    for frame_name, frame in (("raw_df", raw_df), ("clean_df", clean_df), ("removed_df", removed_df)):
        if not frame.empty and label_col not in frame.columns:
            raise KeyError(f"{label_col!r} not found in {frame_name}.")

    labels = sorted(
        set(raw_df[label_col].dropna().unique() if label_col in raw_df.columns else [])
        | set(clean_df[label_col].dropna().unique() if label_col in clean_df.columns else [])
        | set(removed_df[label_col].dropna().unique() if label_col in removed_df.columns else []),
        key=str,
    )

    total_before = int(len(raw_df))
    total_after = int(len(clean_df))
    rows: list[dict[str, Any]] = []

    for label in labels:
        before = int((raw_df[label_col] == label).sum()) if label_col in raw_df.columns else 0
        after = int((clean_df[label_col] == label).sum()) if label_col in clean_df.columns else 0
        removed = int((removed_df[label_col] == label).sum()) if label_col in removed_df.columns else before - after

        rows.append(
            {
                label_col: label,
                "count_before": before,
                "count_after": after,
                "count_removed": removed,
                "pct_before": round(before / total_before * 100, 2) if total_before else 0.0,
                "pct_after": round(after / total_after * 100, 2) if total_after else 0.0,
                "removal_rate_pct": round(removed / before * 100, 2) if before else 0.0,
            }
        )

    rows.append(
        {
            label_col: "TOTAL",
            "count_before": total_before,
            "count_after": total_after,
            "count_removed": int(len(removed_df)),
            "pct_before": 100.0 if total_before else 0.0,
            "pct_after": 100.0 if total_after else 0.0,
            "removal_rate_pct": round(len(removed_df) / total_before * 100, 2) if total_before else 0.0,
        }
    )

    return pd.DataFrame(rows)


def summarize_removal_reasons(
    removed_df: pd.DataFrame,
    reason_col: str | None = None,
) -> pd.DataFrame:
    """Summarize removal reasons by count and percentage of removed rows.

    If a pipe-delimited reason column is used, each reason is counted once, so
    percentages may sum to more than 100%.
    """
    columns = ["removal_reason", "count", "percentage"]
    if removed_df.empty:
        return pd.DataFrame(columns=columns)

    if reason_col is None:
        reason_col = "removal_reasons" if "removal_reasons" in removed_df.columns else "removal_reason"

    if reason_col not in removed_df.columns:
        raise KeyError(f"reason_col not found in removed_df: {reason_col}")

    reasons: list[str] = []
    for value in removed_df[reason_col].fillna(""):
        text = str(value).strip()
        if not text or text == "kept":
            continue
        reasons.extend(reason for reason in text.split("|") if reason)

    if not reasons:
        return pd.DataFrame(columns=columns)

    counts = pd.Series(reasons).value_counts().sort_values(ascending=False)
    summary = counts.rename_axis("removal_reason").reset_index(name="count")
    summary["percentage"] = (summary["count"] / len(removed_df) * 100).round(2)
    return summary


def evaluate_cleaning_retention(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    label_col: str = "label",
) -> dict[str, Any]:
    """Compute retention, removal, and class-balance shift after cleaning."""
    if not raw_df.empty and label_col not in raw_df.columns:
        raise KeyError(f"label_col not found in raw_df: {label_col}")
    if not clean_df.empty and label_col not in clean_df.columns:
        raise KeyError(f"label_col not found in clean_df: {label_col}")

    raw_n = int(len(raw_df))
    clean_n = int(len(clean_df))
    removed_n = raw_n - clean_n

    if raw_n == 0:
        return {
            "n_before": 0,
            "n_after": clean_n,
            "n_removed": 0,
            "retention_rate": 0.0,
            "removal_rate": 0.0,
            "retention_pct": 0.0,
            "removal_pct": 0.0,
            "class_balance_shift": 0.0,
            "class_balance_shift_pct": 0.0,
        }

    raw_props = raw_df[label_col].value_counts(normalize=True, dropna=False)
    clean_props = (
        clean_df[label_col].value_counts(normalize=True, dropna=False)
        if clean_n
        else pd.Series(dtype=float)
    )
    all_labels = sorted(set(raw_props.index).union(set(clean_props.index)), key=str)

    class_balance_shift = max(
        (abs(float(clean_props.get(label, 0.0)) - float(raw_props.get(label, 0.0))) for label in all_labels),
        default=0.0,
    )

    retention_rate = clean_n / raw_n
    removal_rate = removed_n / raw_n

    return {
        "n_before": raw_n,
        "n_after": clean_n,
        "n_removed": removed_n,
        "retention_rate": round(retention_rate, 6),
        "removal_rate": round(removal_rate, 6),
        "retention_pct": round(retention_rate * 100, 2),
        "removal_pct": round(removal_rate * 100, 2),
        "class_balance_shift": round(class_balance_shift, 6),
        "class_balance_shift_pct": round(class_balance_shift * 100, 2),
    }


__all__ = [
    "compute_quality_score",
    "mark_near_duplicates",
    "compute_soft_flags",
    "build_cleaning_mask",
    "assign_removal_reasons",
    "apply_cleaning",
    "summarize_cleaning",
    "summarize_removal_reasons",
    "evaluate_cleaning_retention",
]
