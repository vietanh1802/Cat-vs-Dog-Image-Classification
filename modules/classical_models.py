from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from modules.config_utils import SUPPORTED_CLASSIFIERS, validate_and_normalize


def get_param_grid(classifier_name: str, grid_size: str = "small") -> dict[str, Any]:
    """Return a default parameter grid for the requested classifier."""
    classifier_name = validate_and_normalize(classifier_name, SUPPORTED_CLASSIFIERS, "classifier")
    normalized_grid_size = grid_size.lower().strip()

    if normalized_grid_size not in {"small", "large"}:
        raise ValueError("grid_size must be either 'small' or 'large'.")

    if classifier_name == "logistic_regression":
        if normalized_grid_size == "large":
            return {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], "max_iter": [3000]}
        return {"C": [0.1, 1.0, 10.0], "max_iter": [3000]}

    if classifier_name == "svm_linear":
        if normalized_grid_size == "large":
            return {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
        return {"C": [0.1, 1.0, 10.0]}

    if classifier_name == "random_forest":
        if normalized_grid_size == "large":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }
        return {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }

    if classifier_name == "voting_soft":
        if normalized_grid_size == "large":
            return {
                "lr__clf__C": [0.03, 0.1, 0.3, 1.0],
                "svm__clf__C": [0.03, 0.1, 0.3, 1.0],
                "rf__n_estimators": [100, 200],
                "rf__max_depth": [None, 10, 20],
                "weights": [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
            }
        return {
            "lr__clf__C": [0.1, 1.0],
            "svm__clf__C": [0.1, 1.0],
            "rf__n_estimators": [100],
            "rf__max_depth": [None, 10],
            "weights": [(1, 1, 1), (2, 1, 1), (1, 2, 1)],
        }

    if classifier_name == "stacking":
        return {
            "final_estimator__C": [0.1, 1.0, 10.0],
            "rf__max_depth": [None, 10],
        }

    raise ValueError(f"No default grid defined for classifier: {classifer_name}")

def _build_logistic_regression(seed: int, **params: Any) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            class_weight=params.get("class_weight"),
            solver=params.get("solver", "lbfgs"),
            random_state=seed,
        )),
    ])


def _build_svm_linear(seed: int, **params: Any) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="linear",
            C=params.get("C", 1.0),
            probability=params.get("probability", True),
            class_weight=params.get("class_weight"),
            random_state=seed,
        )),
    ])


def _build_random_forest(seed: int, **params: Any) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth"),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        class_weight=params.get("class_weight"),
        random_state=seed,
        n_jobs=params.get("n_jobs", -1),
    )


def get_classifier(
    name: str,
    seed: int = 42,
    params: dict[str, Any] | None = None,
    **overrides: Any,
) -> BaseEstimator:
    """Build a classical classifier from a name and parameter dictionary."""
    name = validate_and_normalize(name, SUPPORTED_CLASSIFIERS, "classifier")

    final_params = dict(params or {})
    final_params.update(overrides)

    if name == "logistic_regression":
        return _build_logistic_regression(seed=seed, **final_params)

    if name == "svm_linear":
        return _build_svm_linear(seed=seed, **final_params)

    if name == "random_forest":
        return _build_random_forest(seed=seed, **final_params)

    if name == "voting_soft":
        voting_kwargs = {
            "estimators": [
                ("lr", _build_logistic_regression(seed=seed, **final_params.get("lr", {}))),
                ("svm", _build_svm_linear(seed=seed, **final_params.get("svm", {}))),
                ("rf", _build_random_forest(seed=seed, **final_params.get("rf", {}))),
            ],
            "voting": final_params.get("voting", "soft"),
            "n_jobs": final_params.get("n_jobs", -1),
        }

        if "weights" in final_params:
            voting_kwargs["weights"] = final_params["weights"]

        return VotingClassifier(**voting_kwargs)

    if name == "stacking":
        final_estimator_params = dict(final_params.get("final_estimator", {}))

        return StackingClassifier(
            estimators=[
                ("lr", _build_logistic_regression(seed=seed, **final_params.get("lr", {}))),
                ("svm", _build_svm_linear(seed=seed, **final_params.get("svm", {}))),
                ("rf", _build_random_forest(seed=seed, **final_params.get("rf", {}))),
            ],
            final_estimator=LogisticRegression(
                C=final_estimator_params.get("C", 1.0),
                max_iter=final_estimator_params.get("max_iter", 3000),
                class_weight=final_estimator_params.get("class_weight"),
                solver=final_estimator_params.get("solver", "lbfgs"),
                random_state=seed,
                n_jobs=final_estimator_params.get("n_jobs"),
            ),
            n_jobs=final_params.get("n_jobs", -1),
        )

    raise ValueError(f"Unsupported classifier: {name}")


def train_classifier(
    X_train,
    y_train,
    classifier_name: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    **overrides: Any,
) -> BaseEstimator:
    """Train a classical classifier on training data."""
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must not be None.")

    model = get_classifier(
        classifier_name,
        seed=seed,
        params=params,
        **overrides,
    )
    model.fit(X_train, y_train)
    return model

def _classifier_run_name(cfg: dict, index: int) -> str:
    """Build a stable display/run name for a classifier config."""
    if "run_name" in cfg:
        return str(cfg["run_name"])

    name = str(cfg["name"])
    params = cfg.get("params", {})

    if not params:
        return f"{index:02d}_{name}"

    short_params = "_".join(f"{key}-{value}" for key, value in sorted(params.items()))
    return f"{index:02d}_{name}_{short_params}"


def benchmark_classifiers(
    X_train,
    y_train,
    X_val,
    y_val,
    classifier_configs: list[dict],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Train multiple classifiers and compute validation metrics."""
    if not classifier_configs:
        raise ValueError("classifier_configs must contain at least one classifier config.")

    rows: list[dict[str, Any]] = []
    trained: dict[str, BaseEstimator] = {}

    for index, cfg in enumerate(classifier_configs):
        if "name" not in cfg:
            raise KeyError(f"Missing 'name' in classifier config at index {index}: {cfg}")

        name = str(cfg["name"])
        params = dict(cfg.get("params", {}))
        run_name = _classifier_run_name(cfg, index)

        model = train_classifier(X_train, y_train, name, seed=seed, **params)
        preds = model.predict(X_val)

        trained[run_name] = model
        rows.append({
            "run_name": run_name,
            "model": name,
            "params": params,
            "accuracy": accuracy_score(y_val, preds),
            "precision_macro": precision_score(y_val, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, preds, average="macro", zero_division=0),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True),
        trained,
    )


def select_best_model(
    results_df: pd.DataFrame,
    trained_models: dict[str, BaseEstimator],
    metric: str = "f1_macro",
    model_key_col: str = "run_name",
) -> tuple[str, BaseEstimator, pd.Series]:
    """Select the best trained model from benchmark results."""
    if results_df.empty:
        raise ValueError("results_df is empty; cannot select best model.")

    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results_df columns: {list(results_df.columns)}")

    if model_key_col not in results_df.columns:
        raise ValueError(f"Model key column '{model_key_col}' not found in results_df.")

    best_row = results_df.sort_values(metric, ascending=False).iloc[0]
    model_key = str(best_row[model_key_col])

    if model_key not in trained_models:
        raise KeyError(f"Model '{model_key}' not found in trained_models.")

    return model_key, trained_models[model_key], best_row


def _normalize_param_grid_for_estimator(
    classifier_name: str,
    param_grid: dict[str, Any],
) -> dict[str, Any]:
    """Map simple classifier params to sklearn Pipeline params when needed."""
    name = validate_and_normalize(classifier_name, SUPPORTED_CLASSIFIERS, "classifier")

    if name in {"logistic_regression", "svm_linear"}:
        return {
            key if "__" in key else f"clf__{key}": value
            for key, value in param_grid.items()
        }

    return param_grid


def _normalize_scoring(
    scoring: str | list[str] | tuple[str, ...],
    refit: str | None = None,
) -> tuple[str | dict[str, str], str]:
    """Normalize scoring input for GridSearchCV."""
    if isinstance(scoring, str):
        scoring_name = scoring.strip()
        if not scoring_name:
            raise ValueError("scoring must not be empty.")

        refit_metric = refit or scoring_name
        if refit_metric != scoring_name:
            raise ValueError(
                f"refit='{refit_metric}' must match scoring='{scoring_name}' when scoring is a string."
            )

        return scoring_name, refit_metric

    scoring_names = [str(metric).strip() for metric in scoring if str(metric).strip()]
    if not scoring_names:
        raise ValueError("scoring must contain at least one metric.")

    scoring_map = {metric: metric for metric in scoring_names}
    refit_metric = refit or scoring_names[0]

    if refit_metric not in scoring_map:
        raise ValueError(
            f"refit='{refit_metric}' must be one of scoring metrics: {scoring_names}"
        )

    return scoring_map, refit_metric


def tune_with_params(
    X_train,
    y_train,
    classifier_name: str,
    param_grid: dict[str, Any] | None = None,
    base_params: dict[str, Any] | None = None,
    grid_size: str = "small",
    cv: int = 3,
    seed: int = 42,
    scoring: str | list[str] | tuple[str, ...] = "f1_macro",
    refit: str | None = None,
    n_jobs: int = -1,
    verbose: int = 1,
    return_train_score: bool = True,
) -> GridSearchCV:
    """Perform grid search tuning with optional default grid lookup."""
    if param_grid is None:
        param_grid = get_param_grid(classifier_name, grid_size)

    if not param_grid:
        raise ValueError("param_grid must not be empty.")

    estimator = get_classifier(
        classifier_name,
        seed=seed,
        params=base_params,
    )

    normalized_grid = _normalize_param_grid_for_estimator(classifier_name, param_grid)
    normalized_scoring, refit_metric = _normalize_scoring(scoring, refit=refit)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=normalized_grid,
        cv=cv,
        scoring=normalized_scoring,
        refit=refit_metric,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=return_train_score,
    )

    search.fit(X_train, y_train)
    return search


def tune_classifier_grid(
    X_train,
    y_train,
    classifier_name: str,
    param_grid: dict[str, Any],
    base_params: dict[str, Any] | None = None,
    cv: int = 3,
    seed: int = 42,
    scoring: str | list[str] | tuple[str, ...] = "f1_macro",
    refit: str | None = None,
    n_jobs: int = -1,
    verbose: int = 2,
    return_train_score: bool = False,
) -> GridSearchCV:
    """Perform grid search hyperparameter tuning on a classifier."""
    return tune_with_params(
        X_train=X_train,
        y_train=y_train,
        classifier_name=classifier_name,
        param_grid=param_grid,
        base_params=base_params,
        cv=cv,
        seed=seed,
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=return_train_score,
    )