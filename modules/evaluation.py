"""
Evaluation utilities for classification models.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_1d_array(values: Any, name: str) -> np.ndarray:
    """Convert values to a 1D NumPy array."""
    array = np.asarray(values)

    if array.ndim != 1:
        array = array.reshape(-1)

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    return array


def _validate_prediction_lengths(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate that true and predicted labels have the same length."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"but y_pred has {len(y_pred)} samples."
        )


def _resolve_labels_and_names(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    label_names: Sequence[str] | Mapping[Any, str] | None = None,
) -> tuple[list[Any], list[str]]:
    """Resolve stable metric labels and display names."""
    if labels is None:
        labels_array = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        resolved_labels = sorted(labels_array.tolist())
    else:
        resolved_labels = list(labels)

    if label_names is None:
        resolved_names = [str(label) for label in resolved_labels]
    elif isinstance(label_names, Mapping):
        resolved_names = [str(label_names.get(label, label)) for label in resolved_labels]
    else:
        resolved_names = [str(name) for name in label_names]
        if len(resolved_names) != len(resolved_labels):
            raise ValueError(
                "label_names must have the same length as labels. "
                f"Got {len(resolved_names)} names and {len(resolved_labels)} labels."
            )

    return resolved_labels, resolved_names


def _extract_positive_class_scores(y_prob: np.ndarray) -> np.ndarray | None:
    """Return positive-class scores for binary ROC-AUC when possible."""
    if y_prob.ndim == 1:
        return y_prob

    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return y_prob[:, 1]

    return None


def compute_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute numeric classification metrics from true and predicted labels."""
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_prediction_lengths(y_true_array, y_pred_array)

    wrong = int((y_true_array != y_pred_array).sum())
    total = int(len(y_true_array))

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision_macro": float(
            precision_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "wrong_predictions": wrong,
        "total_samples": total,
        "error_rate": float(wrong / total),
    }

    if y_prob is not None:
        y_prob_array = np.asarray(y_prob)
        if len(y_prob_array) != total:
            raise ValueError(
                f"Length mismatch: y_prob has {len(y_prob_array)} samples, "
                f"but y_true has {total} samples."
            )

        try:
            unique_labels = np.unique(y_true_array)
            if len(unique_labels) == 2:
                positive_scores = _extract_positive_class_scores(y_prob_array)
                metrics["roc_auc"] = (
                    float(roc_auc_score(y_true_array, positive_scores))
                    if positive_scores is not None
                    else float("nan")
                )
            elif y_prob_array.ndim == 2 and y_prob_array.shape[1] == len(unique_labels):
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(
                        y_true_array,
                        y_prob_array,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except Exception:
            metrics["roc_auc"] = float("nan")

    return metrics


def classification_report_df(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    label_names: Sequence[str] | Mapping[Any, str] | None = None,
) -> pd.DataFrame:
    """Return sklearn's classification report as a dataframe."""
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_prediction_lengths(y_true_array, y_pred_array)

    resolved_labels, resolved_names = _resolve_labels_and_names(
        y_true_array,
        y_pred_array,
        labels=labels,
        label_names=label_names,
    )

    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=resolved_labels,
        target_names=resolved_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_matrix_df(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    label_names: Sequence[str] | Mapping[Any, str] | None = None,
) -> pd.DataFrame:
    """Return a confusion matrix as a dataframe."""
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_prediction_lengths(y_true_array, y_pred_array)

    resolved_labels, resolved_names = _resolve_labels_and_names(
        y_true_array,
        y_pred_array,
        labels=labels,
        label_names=label_names,
    )

    matrix = confusion_matrix(y_true_array, y_pred_array, labels=resolved_labels)
    row_names = [f"Actual {name}" for name in resolved_names]
    col_names = [f"Predicted {name}" for name in resolved_names]
    return pd.DataFrame(matrix, index=row_names, columns=col_names)


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: Sequence[Any] | np.ndarray | None = None,
    labels: Sequence[Any] | None = None,
    label_names: Sequence[str] | Mapping[Any, str] | None = None,
) -> dict[str, Any]:
    """Package metrics, report, and confusion matrix into one result dictionary."""
    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob)
    report = classification_report_df(
        y_true,
        y_pred,
        labels=labels,
        label_names=label_names,
    )
    matrix = confusion_matrix_df(
        y_true,
        y_pred,
        labels=labels,
        label_names=label_names,
    )
    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": matrix,
    }


def evaluate_estimator(
    model: Any,
    X: Any,
    y: Sequence[Any],
    model_name: str = "model",
) -> dict[str, Any]:
    """Evaluate a fitted estimator on feature data."""
    y_pred = model.predict(X)

    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
        except Exception:
            y_prob = None

    metrics = compute_classification_metrics(y, y_pred, y_prob=y_prob)
    return {
        "model": model_name,
        **metrics,
    }


def find_wrong_predictions(
    df: pd.DataFrame,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: Sequence[Any] | np.ndarray | None = None,
    path_col: str = "path",
    label_map: Mapping[Any, str] | None = None,
) -> pd.DataFrame:
    """Return misclassified rows with true/predicted labels and optional confidence."""
    if path_col not in df.columns:
        raise KeyError(f"path_col not found in dataframe: {path_col}")

    output = df.copy().reset_index(drop=True)
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_prediction_lengths(y_true_array, y_pred_array)

    if len(output) != len(y_true_array):
        raise ValueError(
            f"Length mismatch: dataframe has {len(output)} rows, "
            f"but predictions have {len(y_true_array)} rows."
        )

    output["true_label"] = y_true_array
    output["pred_label"] = y_pred_array

    if label_map is not None:
        output["true_label_name"] = output["true_label"].map(label_map).fillna(output["true_label"].astype(str))
        output["pred_label_name"] = output["pred_label"].map(label_map).fillna(output["pred_label"].astype(str))

    if y_prob is not None:
        y_prob_array = np.asarray(y_prob)
        if len(y_prob_array) != len(output):
            raise ValueError(
                f"Length mismatch: y_prob has {len(y_prob_array)} rows, "
                f"but dataframe has {len(output)} rows."
            )

        if y_prob_array.ndim == 2:
            pred_positions = output["pred_label"].to_numpy()
            confidence = []
            for row_idx, pred_label in enumerate(pred_positions):
                try:
                    confidence.append(float(y_prob_array[row_idx, int(pred_label)]))
                except Exception:
                    confidence.append(float(np.max(y_prob_array[row_idx])))
            output["confidence"] = confidence
        elif y_prob_array.ndim == 1:
            output["confidence"] = y_prob_array

    output["is_correct"] = output["true_label"] == output["pred_label"]

    keep_cols = [
        path_col,
        "true_label",
        "pred_label",
        "true_label_name",
        "pred_label_name",
        "confidence",
        "is_correct",
    ]
    keep_cols = [col for col in keep_cols if col in output.columns]

    return output.loc[~output["is_correct"], keep_cols].reset_index(drop=True)


def format_metrics_table(metrics_dict: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a metrics dictionary into a one-row dataframe for notebook display."""
    return pd.DataFrame([dict(metrics_dict)])


def compare_pipeline_results(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Compare multiple pipeline result dictionaries as a dataframe.

    Accepts either:
    - {"classical": {"metrics": {...}}, "deep_learning": {"metrics": {...}}}
    - {"classical": {...flat metrics...}, "deep_learning": {...flat metrics...}}
    """
    rows: list[dict[str, Any]] = []

    for name, result in results.items():
        metrics = result.get("metrics", result)
        row = {"pipeline": name}
        row.update(dict(metrics))
        rows.append(row)

    return pd.DataFrame(rows)


# Backward-compatible alias.
evaluate_model = evaluate_estimator


__all__ = [
    "compute_classification_metrics",
    "classification_report_df",
    "confusion_matrix_df",
    "evaluate_predictions",
    "evaluate_estimator",
    "evaluate_model",
    "find_wrong_predictions",
    "format_metrics_table",
    "compare_pipeline_results",
]
