from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_NAMES = ["mahalanobis", "logistic", "rbf_svm", "gradient_boosting"]


class RobustMahalanobisClassifier(BaseEstimator, ClassifierMixin):
    """Positive-reference trimmed covariance plus calibrated distance score."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        positives = X[y == 1]
        if len(positives) < X.shape[1] + 2:
            raise ValueError("Too few positive tracks for robust Mahalanobis fitting")
        # Start from a coordinate-wise median/MAD center, then iteratively retain
        # the central 97.5% of positive-reference distances. This gives a stable,
        # auditable robust covariance estimate without relying on platform-specific
        # minimum-covariance-determinant native code.
        median = np.median(positives, axis=0)
        mad = np.median(np.abs(positives - median), axis=0)
        mad = np.where(mad > 1e-12, mad, 1.0)
        initial_distance = np.sum(((positives - median) / mad) ** 2, axis=1)
        retained = positives[initial_distance <= np.quantile(initial_distance, 0.975)]
        for _ in range(4):
            location = retained.mean(axis=0)
            covariance = np.cov(retained, rowvar=False)
            ridge = max(float(np.trace(covariance)) / covariance.shape[0] * 1e-8, 1e-10)
            covariance = covariance + np.eye(covariance.shape[0]) * ridge
            precision = np.linalg.pinv(covariance)
            delta = positives - location
            positive_distance = np.einsum("ij,jk,ik->i", delta, precision, delta)
            retained = positives[positive_distance <= np.quantile(positive_distance, 0.975)]
        self.location_ = retained.mean(axis=0)
        self.covariance_ = np.cov(retained, rowvar=False)
        ridge = max(float(np.trace(self.covariance_)) / self.covariance_.shape[0] * 1e-8, 1e-10)
        self.covariance_ = self.covariance_ + np.eye(self.covariance_.shape[0]) * ridge
        self.precision_ = np.linalg.pinv(self.covariance_)
        self.n_positive_reference_ = len(positives)
        self.n_robust_reference_retained_ = len(retained)
        distance = self.mahalanobis_distance_squared(X)
        self.calibrator_ = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=self.random_state)
        self.calibrator_.fit((-np.log1p(distance)).reshape(-1, 1), y)
        self.classes_ = np.array([0, 1])
        return self

    def mahalanobis_distance_squared(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        delta = matrix - self.location_
        return np.einsum("ij,jk,ik->i", delta, self.precision_, delta)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distance = self.mahalanobis_distance_squared(X)
        return self.calibrator_.predict_proba((-np.log1p(distance)).reshape(-1, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_models(random_seed: int) -> dict[str, BaseEstimator]:
    return {
        "mahalanobis": RobustMahalanobisClassifier(random_state=random_seed),
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=random_seed,
        ),
    }


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, model_name: str) -> dict[str, float | str]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "model": model_name,
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall_sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, prediction, zero_division=0),
        "brier_score": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, np.column_stack([1 - probability, probability]), labels=[0, 1]),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def consensus_labels(probabilities: pd.DataFrame, settings: dict) -> pd.DataFrame:
    probability_columns = [f"probability_{name}" for name in MODEL_NAMES]
    result = probabilities.copy()
    result["ensemble_probability"] = result[probability_columns].mean(axis=1)
    result["positive_model_votes"] = (result[probability_columns] >= 0.5).sum(axis=1)
    positive = (
        (result["ensemble_probability"] >= float(settings["positive_probability_threshold"]))
        & (result["positive_model_votes"] >= int(settings["minimum_positive_votes"]))
    )
    negative = (
        (result["ensemble_probability"] <= float(settings["negative_probability_threshold"]))
        & (result["positive_model_votes"] <= int(settings["maximum_negative_votes"]))
    )
    result["consensus_prediction"] = "uncertain_or_out_of_reference"
    result.loc[positive, "consensus_prediction"] = "chromatin_bound_like"
    result.loc[negative, "consensus_prediction"] = "not_chromatin_bound_like"
    result["model_probability_range"] = result[probability_columns].max(axis=1) - result[probability_columns].min(axis=1)
    return result


@dataclass
class TrainingResult:
    models: dict[str, BaseEstimator]
    out_of_fold_predictions: pd.DataFrame
    metrics: pd.DataFrame


def train_models_with_oof(
    X: np.ndarray,
    y: np.ndarray,
    track_uids: pd.Series,
    settings: dict,
    model_dir: str | Path,
) -> TrainingResult:
    random_seed = int(settings["random_seed"])
    n_splits = int(settings["cross_validation_folds"])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof = pd.DataFrame({"track_uid": track_uids.astype(str).to_numpy(), "reference_label_binary": y})
    oof["cv_fold"] = -1
    for model_name in MODEL_NAMES:
        oof[f"probability_{model_name}"] = np.nan
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        fold_models = make_models(random_seed + fold)
        oof.loc[test_index, "cv_fold"] = fold
        for model_name, model in fold_models.items():
            model.fit(X[train_index], y[train_index])
            oof.loc[test_index, f"probability_{model_name}"] = model.predict_proba(X[test_index])[:, 1]
    oof = consensus_labels(oof, settings)
    metric_rows = [
        binary_metrics(y, oof[f"probability_{name}"].to_numpy(float), name)
        for name in MODEL_NAMES
    ]
    metric_rows.append(binary_metrics(y, oof["ensemble_probability"].to_numpy(float), "ensemble"))
    metrics = pd.DataFrame(metric_rows)
    final_models = make_models(random_seed)
    output_path = Path(model_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for model_name, model in final_models.items():
        model.fit(X, y)
        joblib.dump(model, output_path / f"RRL_{model_name}_model.joblib")
    return TrainingResult(final_models, oof, metrics)


def load_models(model_dir: str | Path) -> dict[str, BaseEstimator]:
    directory = Path(model_dir)
    if (directory / 'PORTABLE_MODELS.json').exists():
        from .portable_models import load_portable_models
        return load_portable_models(directory)
    models = {}
    for name in MODEL_NAMES:
        path = directory / f"RRL_{name}_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing trained model: {path}")
        models[name] = joblib.load(path)
    return models


def predict_models(models: dict[str, BaseEstimator], X: np.ndarray, track_uids: pd.Series, settings: dict) -> pd.DataFrame:
    result = pd.DataFrame({"track_uid": track_uids.astype(str).to_numpy()})
    for name in MODEL_NAMES:
        result[f"probability_{name}"] = models[name].predict_proba(X)[:, 1]
    mahalanobis = models["mahalanobis"]
    result["mahalanobis_distance_squared"] = mahalanobis.mahalanobis_distance_squared(X)
    return consensus_labels(result, settings)
