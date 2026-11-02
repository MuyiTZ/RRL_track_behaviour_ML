from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


IDENTITY_COLUMNS = ["Video #", "Cell", "Track"]
MODEL_FEATURES = [
    "log_Rg_resid",
    "log_convex_hull_area_resid",
    "log_D_resid",
    "alpha_resid",
]
PHYSICAL_FEATURES = ["Rg", "convex_hull_area", "D", "alpha"]


def validate_required_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _identity_text(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def make_track_uid(video: object, cell: object, track: object) -> str:
    return "_".join(_identity_text(value) for value in (video, cell, track))


def make_track_identifier(video: object, cell: object, track: object) -> str:
    return f"V{_identity_text(video)}_C{_identity_text(cell)}_T{_identity_text(track)}"


def add_track_identity(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, IDENTITY_COLUMNS, "track table")
    result = df.copy()
    result["track_uid"] = [make_track_uid(*row) for row in result[IDENTITY_COLUMNS].itertuples(index=False, name=None)]
    result["track_identifier"] = [make_track_identifier(*row) for row in result[IDENTITY_COLUMNS].itertuples(index=False, name=None)]
    return result


def radius_of_gyration(x_um: np.ndarray, y_um: np.ndarray) -> float:
    points = np.column_stack([x_um, y_um])
    center = points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1))))


def convex_hull_area(x_um: np.ndarray, y_um: np.ndarray) -> float:
    points = np.column_stack([x_um, y_um])
    if len(points) < 3:
        return np.nan
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return np.nan


def msd_features(
    x_um: np.ndarray,
    y_um: np.ndarray,
    frame_numbers: np.ndarray,
    frame_interval_seconds: float,
    max_msd_lag_frames: int,
) -> tuple[float, float, int]:
    points = np.column_stack([x_um, y_um])
    frame_to_point = {int(frame): point for frame, point in zip(frame_numbers, points)}
    max_lag = min(max_msd_lag_frames, len(points) - 1)
    lags: list[float] = []
    msds: list[float] = []
    for lag in range(1, max_lag + 1):
        displacements = []
        for frame, point in frame_to_point.items():
            later = frame_to_point.get(frame + lag)
            if later is not None:
                displacements.append(float(np.sum((later - point) ** 2)))
        if not displacements:
            continue
        lags.append(lag * frame_interval_seconds)
        msds.append(float(np.mean(displacements)))
    if len(lags) < 2:
        return np.nan, np.nan, 0
    lag_array = np.asarray(lags, dtype=float)
    msd_array = np.asarray(msds, dtype=float)
    denominator = float(np.dot(lag_array, lag_array))
    diffusion = float(np.dot(lag_array, msd_array) / denominator / 4.0) if denominator > 0 else np.nan
    valid = msd_array > 0
    alpha = np.nan
    if valid.sum() >= 2:
        alpha = float(np.polyfit(np.log(lag_array[valid]), np.log(msd_array[valid]), 1)[0])
    return float(diffusion), alpha, len(lags)


def trajectory_geometry(x_pixels: np.ndarray, y_pixels: np.ndarray) -> dict[str, float]:
    points = np.column_stack([x_pixels, y_pixels]).astype(float)
    n_points = len(points)
    result = {
        "geometry_n_points": int(n_points),
        "pca_linearity": np.nan,
        "line_rmse_norm": np.nan,
        "max_step": np.nan,
        "median_step": np.nan,
    }
    if n_points < 2:
        return result
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    result["max_step"] = float(np.max(steps))
    result["median_step"] = float(np.median(steps))
    centered = points - points.mean(axis=0)
    _, singular_values, vectors_t = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2
    total = float(variances.sum())
    result["pca_linearity"] = float(variances[0] / total) if total > 0 else np.nan
    direction = vectors_t[0]
    projected = np.outer(centered @ direction, direction)
    orthogonal = centered - projected
    rmse = float(np.sqrt(np.mean(np.sum(orthogonal**2, axis=1))))
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    result["line_rmse_norm"] = rmse / extent if extent > 0 else np.nan
    return result


def passes_linear_artifact_rule(metrics: dict[str, float], rule: dict[str, float]) -> bool:
    return bool(
        metrics["geometry_n_points"] >= int(rule["minimum_points"])
        and metrics["pca_linearity"] >= float(rule["pca_linearity_min"])
        and metrics["line_rmse_norm"] <= float(rule["line_rmse_norm_max"])
        and metrics["max_step"] >= float(rule["max_step_min_pixels"])
    )


def compute_track_features(track: pd.DataFrame, settings: dict, linear_artifact_rule: dict) -> dict[str, object]:
    validate_required_columns(track, IDENTITY_COLUMNS + ["Frame", "x", "y"], "coordinate track")
    ordered = track.sort_values("Frame")
    x = pd.to_numeric(ordered["x"], errors="coerce")
    y = pd.to_numeric(ordered["y"], errors="coerce")
    frame_values = pd.to_numeric(ordered["Frame"], errors="coerce")
    finite_coordinates = x.notna() & y.notna()
    valid = finite_coordinates & frame_values.notna() & (x >= 0) & (y >= 0)
    x_geometry = x[finite_coordinates].to_numpy(float)
    y_geometry = y[finite_coordinates].to_numpy(float)
    x_pixels = x[valid].to_numpy(float)
    y_pixels = y[valid].to_numpy(float)
    frame_numbers = frame_values[valid].to_numpy(float)
    first = ordered.iloc[0]
    result: dict[str, object] = {
        "Video #": first["Video #"],
        "Cell": first["Cell"],
        "Track": first["Track"],
        "Video Name": first.get("Video Name", ""),
        "raw_track_length": int(len(ordered)),
        "n_valid_frames": int(valid.sum()),
        "Rg": np.nan,
        "convex_hull_area": np.nan,
        "D": np.nan,
        "alpha": np.nan,
        "feature_warning": "",
    }
    # The established audit includes finite -1 gap placeholders here because
    # erroneous links to those placeholders create the diagnostic long line.
    # Physical features below continue to use valid non-negative positions only.
    geometry = trajectory_geometry(x_geometry, y_geometry)
    result.update(geometry)
    result["linear_artifact_rule_flag"] = passes_linear_artifact_rule(geometry, linear_artifact_rule)
    if len(x_pixels) < int(settings["minimum_valid_points"]):
        result["feature_warning"] = "too_few_valid_points"
        return result
    scale = float(settings["pixel_size_um"])
    x_um = x_pixels * scale
    y_um = y_pixels * scale
    result["Rg"] = radius_of_gyration(x_um, y_um)
    result["convex_hull_area"] = convex_hull_area(x_um, y_um)
    diffusion, alpha, lag_count = msd_features(
        x_um,
        y_um,
        frame_numbers,
        float(settings["frame_interval_seconds"]),
        int(settings["max_msd_lag_frames"]),
    )
    result["D"] = diffusion
    result["alpha"] = alpha
    result["msd_fitted_lag_count"] = lag_count
    if not np.isfinite(result["convex_hull_area"]):
        result["feature_warning"] = "failed_convex_hull"
    elif not np.isfinite(diffusion) or not np.isfinite(alpha):
        result["feature_warning"] = "failed_msd_fit"
    return result


def add_log_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    for column in PHYSICAL_FEATURES + ["raw_track_length"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["log_Rg"] = np.where(result["Rg"] > 0, np.log10(result["Rg"]), np.nan)
    result["log_convex_hull_area"] = np.where(
        result["convex_hull_area"] > 0,
        np.log10(result["convex_hull_area"]),
        np.nan,
    )
    result["log_D"] = np.where(result["D"] > 0, np.log10(result["D"]), np.nan)
    result["alpha_model"] = result["alpha"]
    return result


def fit_length_residualizers(features: pd.DataFrame) -> pd.DataFrame:
    mapping = ["log_Rg", "log_convex_hull_area", "log_D", "alpha_model"]
    rows = []
    for feature in mapping:
        x = pd.to_numeric(features["raw_track_length"], errors="coerce")
        y = pd.to_numeric(features[feature], errors="coerce")
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 3:
            raise ValueError(f"Too few finite rows to fit residualizer for {feature}")
        slope, intercept = np.polyfit(x[valid].to_numpy(float), y[valid].to_numpy(float), 1)
        predicted = intercept + slope * x[valid].to_numpy(float)
        observed = y[valid].to_numpy(float)
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        r_squared = 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator if denominator > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "intercept": float(intercept),
                "track_length_slope": float(slope),
                "r_squared": r_squared,
                "n_tracks_fitted": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def add_residualized_features(features: pd.DataFrame, residualizers: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    mapping = {
        "log_Rg": "log_Rg_resid",
        "log_convex_hull_area": "log_convex_hull_area_resid",
        "log_D": "log_D_resid",
        "alpha_model": "alpha_resid",
    }
    parameter_lookup = residualizers.set_index("feature")
    for source, destination in mapping.items():
        if source not in parameter_lookup.index:
            raise ValueError(f"Residualizer parameters are missing feature {source}")
        params = parameter_lookup.loc[source]
        fitted = float(params["intercept"]) + float(params["track_length_slope"]) * result["raw_track_length"]
        result[destination] = result[source] - fitted
    finite = np.isfinite(result[MODEL_FEATURES].to_numpy(float)).all(axis=1)
    result["model_feature_complete"] = finite
    return result


def add_log_and_residualized_features(features: pd.DataFrame, residualizers: pd.DataFrame) -> pd.DataFrame:
    return add_residualized_features(add_log_features(features), residualizers)


def load_behavior_coordinate_files(input_dir: str | Path, behavior_files: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path = Path(input_dir)
    track_rows: list[pd.DataFrame] = []
    frame_rows: list[pd.DataFrame] = []
    for behavior_class, filename in behavior_files.items():
        path = input_path / filename
        if not path.exists() and behavior_class == "single_constrained":
            alternative = input_path / "single-behaviour-track-constraint.csv"
            if alternative.exists():
                path = alternative
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        validate_required_columns(frame, IDENTITY_COLUMNS + ["Frame", "x", "y"], str(path))
        frame = add_track_identity(frame)
        frame["source_behavior_class"] = behavior_class
        frame["source_coordinate_file"] = str(path)
        frame_rows.append(frame)
        registry = (
            frame.sort_values("Frame")
            .groupby("track_uid", as_index=False)
            .first()[IDENTITY_COLUMNS + ["track_uid", "track_identifier", "source_behavior_class", "source_coordinate_file"]]
        )
        registry["source_frame_rows"] = frame.groupby("track_uid").size().reindex(registry["track_uid"]).to_numpy()
        track_rows.append(registry)
    if not track_rows:
        raise FileNotFoundError(f"No configured behavior-coordinate CSVs were found in {input_path}")
    tracks = pd.concat(track_rows, ignore_index=True)
    frames = pd.concat(frame_rows, ignore_index=True)
    duplicate_tracks = tracks[tracks.duplicated("track_uid", keep=False)]
    if not duplicate_tracks.empty:
        names = duplicate_tracks["track_uid"].drop_duplicates().head(20).tolist()
        raise ValueError(f"Tracks occur in more than one behavior file: {names}")
    return tracks, frames
