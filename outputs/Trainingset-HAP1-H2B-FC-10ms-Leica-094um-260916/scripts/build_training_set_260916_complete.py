"""Build the replacement HAP1-H2B operational training-set release.

The build reads historical inputs but never edits them. It applies the existing
linear-mislink protocol, replaces pooled length residualization with fixed-window
geometry and localization-calibrated apparent mobility, rebuilds the operational
anchor, trains grouped models, and performs paired truncation and self audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import wilcoxon
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


RELEASE_NAME = "Trainingset-HAP1-H2B-FC-10ms-Leica-094um-260916"
HAP1_ROOT = Path(__file__).resolve().parents[3] / "training_inputs" / "HAP1"
UNIFIED_ROOT = HAP1_ROOT / "unified_behavior_feature_space_H2B"
MISLINK_ROOT = UNIFIED_ROOT / "chromatin_bound_like_anchor_overlap_mahalanobis" / "linear_mislink_removed_anchor_core2"
MISLINK_REPORT = MISLINK_ROOT / "anchor_core2_linear_mislink_removal_report_20260818_134133.md"
MISLINK_LIST = MISLINK_ROOT / "linear_mislink_removed_track_list_20260818_134133.csv"
REGROUPED_ASSIGNMENTS = HAP1_ROOT / "track_length_bias_diagnostics_4component_GMM" / "cluster_regrouping_after_length_control" / "regrouped_track_assignments_after_length_control.csv"
LOCALIZATION_ROOT = Path(__file__).resolve().parents[3] / "training_inputs" / "localization"
LOCALIZATION_RUN = LOCALIZATION_ROOT / "outputs" / "localization_error_fixed_cell_1129_gaussian_center90_20260721_135351"
LOCALIZATION_SUMMARY = LOCALIZATION_RUN / "tables" / "1129_fixed_cell_localization_precision_gaussian_center90_summary.csv"
LOCALIZATION_METHOD_SCRIPT = LOCALIZATION_RUN / "scripts" / "calculate_localization_error_fixed_cell_1129_gaussian_center90.py"
LOCALIZATION_METHOD_DECK = LOCALIZATION_ROOT / "outputs" / "localization_error_rms_precision_slides_20260723_115838" / "fixed_cell_localization_precision_rms_method.pptx"
RRL_ROOT = Path(__file__).resolve().parents[3]
OLD_REFERENCE = RRL_ROOT / "outputs" / "H2B_reference_pixel094"
OLD_SELF_TEST = RRL_ROOT / "outputs" / "H2B_self_test_application"

PIXEL_SIZE_UM = 0.094
FRAME_INTERVAL_SECONDS = 0.010
FRAME_INTERVAL_MS = 10.0
EXPOSURE_SECONDS = 0.010
MOTION_BLUR_COEFFICIENT = 1.0 / 6.0
WINDOW_FRAMES = 10
WINDOW_STRIDE = 5
MAX_MSD_LAG = 4
RANDOM_SEED = 20260916
SIMULATION_TRACKS_PER_SCENARIO_LENGTH = 80

BEHAVIOR_FILES = {
    "single_bound": "single-behaviour-track-bound.csv",
    "single_constrained": "single-behaviour-track-constraint-only.csv",
    "single_diffusive": "single-behaviour-track-diffusive.csv",
    "bound_diffusive": "multi-behaviour-track-bound-diffusion.csv",
    "bound_constrained": "multi-behaviour-track-bound-constraint.csv",
    "constrained_diffusive": "multi-behaviour-track-constraint-diffusion.csv",
    "all_three": "multi-behaviour-track-all-three.csv",
}

MODEL_FEATURES = [
    "log_rg2_noise_ratio_w10",
    "log_hull_noise_ratio_w10",
    "log_msd1_fixed90_ratio",
    "alpha_observed",
]
CONSENSUS_ORDER = ["chromatin_bound_like", "uncertain_or_out_of_reference", "not_chromatin_bound_like"]
COLORS = {
    "chromatin_bound_like": "#216E39",
    "uncertain_or_out_of_reference": "#E69F00",
    "not_chromatin_bound_like": "#8C8C8C",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "role": role,
        "path": str(path),
        "bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def tree_stat_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (int(path.stat().st_size), int(path.stat().st_mtime_ns))
        for path in root.rglob("*")
        if path.is_file()
    }


def identity_text(value: object) -> str:
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
    return "_".join(identity_text(value) for value in (video, cell, track))


def make_track_identifier(video: object, cell: object, track: object) -> str:
    return f"V{identity_text(video)}_C{identity_text(cell)}_T{identity_text(track)}"


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_parts: list[pd.DataFrame] = []
    registry_rows: list[dict[str, object]] = []
    for behavior, filename in BEHAVIOR_FILES.items():
        path = HAP1_ROOT / filename
        frame = pd.read_csv(path)
        required = ["Video #", "Cell", "Track", "Frame", "x", "y"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        frame["track_uid"] = [make_track_uid(*row) for row in frame[["Video #", "Cell", "Track"]].itertuples(index=False, name=None)]
        frame["track_identifier"] = [make_track_identifier(*row) for row in frame[["Video #", "Cell", "Track"]].itertuples(index=False, name=None)]
        frame["source_behavior_class"] = behavior
        frame["source_coordinate_file"] = str(path)
        frame_parts.append(frame)
        for uid, track in frame.groupby("track_uid", sort=False):
            first = track.iloc[0]
            registry_rows.append(
                {
                    "Video #": first["Video #"],
                    "Video Name": first.get("Video Name", ""),
                    "Cell": first["Cell"],
                    "Track": first["Track"],
                    "track_uid": uid,
                    "track_identifier": first["track_identifier"],
                    "source_behavior_class": behavior,
                    "source_coordinate_file": str(path),
                    "raw_track_length": int(len(track)),
                }
            )
    frames = pd.concat(frame_parts, ignore_index=True)
    registry = pd.DataFrame(registry_rows)
    if registry["track_uid"].duplicated().any():
        raise ValueError("Track IDs occur in more than one behavior-coordinate file")
    regrouped = pd.read_csv(REGROUPED_ASSIGNMENTS)
    regrouped["track_uid"] = [make_track_uid(*row) for row in regrouped[["Video #", "Cell", "Track"]].itertuples(index=False, name=None)]
    subgroup = regrouped.set_index("track_uid")["scheme_A_full_merge_group"]
    registry["source_constrained_subgroup"] = registry["track_uid"].map(subgroup)
    registry["behavior_plus_constrained_subgroup"] = registry["source_behavior_class"]
    mask = registry["source_behavior_class"].eq("single_constrained") & registry["source_constrained_subgroup"].notna()
    registry.loc[mask, "behavior_plus_constrained_subgroup"] = "single_constrained__" + registry.loc[mask, "source_constrained_subgroup"].astype(str)
    return registry, frames


def trajectory_geometry(track: pd.DataFrame) -> dict[str, float | int]:
    x, y = safe_numeric(track["x"]), safe_numeric(track["y"])
    finite = x.notna() & y.notna()
    points = np.column_stack([x[finite].to_numpy(float), y[finite].to_numpy(float)])
    result: dict[str, float | int] = {
        "geometry_n_points": int(len(points)), "path_length": np.nan, "end_to_end": np.nan,
        "straightness": np.nan, "pca_linearity": np.nan, "minor_major_ratio": np.nan,
        "line_rmse_norm": np.nan, "max_step": np.nan, "median_step": np.nan,
    }
    if len(points) < 2:
        return result
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    result["path_length"] = float(steps.sum())
    result["end_to_end"] = float(np.linalg.norm(points[-1] - points[0]))
    result["straightness"] = float(result["end_to_end"] / result["path_length"]) if result["path_length"] > 0 else np.nan
    result["max_step"], result["median_step"] = float(np.max(steps)), float(np.median(steps))
    centered = points - points.mean(axis=0)
    _, singular_values, vectors_t = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2
    total = float(variances.sum())
    result["pca_linearity"] = float(variances[0] / total) if total > 0 else np.nan
    result["minor_major_ratio"] = float(variances[1] / variances[0]) if len(variances) > 1 and variances[0] > 0 else np.nan
    direction = vectors_t[0]
    projected = np.outer(centered @ direction, direction)
    rmse = float(np.sqrt(np.mean(np.sum((centered - projected) ** 2, axis=1))))
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    result["line_rmse_norm"] = rmse / extent if extent > 0 else np.nan
    return result


def apply_mislink_protocol(registry: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    historic = pd.read_csv(MISLINK_LIST)
    historic["track_uid"] = historic["track_uid"].astype(str)
    user_confirmed = set(historic.loc[historic["user_confirmed_linear_mislink"].astype(bool), "track_uid"])
    rows = []
    for uid, track in frames.groupby("track_uid", sort=False):
        values = trajectory_geometry(track.sort_values("Frame", kind="mergesort"))
        values["track_uid"] = str(uid)
        values["user_confirmed_linear_mislink"] = uid in user_confirmed
        values["auto_linear_mislink_high_confidence"] = bool(
            values["geometry_n_points"] >= 4
            and values["pca_linearity"] >= 0.9999
            and values["line_rmse_norm"] <= 0.008
            and values["max_step"] >= 100.0
        )
        values["linear_mislink_remove_flag"] = bool(values["user_confirmed_linear_mislink"] or values["auto_linear_mislink_high_confidence"])
        if values["user_confirmed_linear_mislink"] and values["auto_linear_mislink_high_confidence"]:
            reason = "user_confirmed_and_auto_geometry"
        elif values["user_confirmed_linear_mislink"]:
            reason = "user_confirmed_list"
        elif values["auto_linear_mislink_high_confidence"]:
            reason = "auto_geometry_high_confidence"
        else:
            reason = "not_removed"
        values["linear_mislink_remove_reason"] = reason
        rows.append(values)
    metrics = pd.DataFrame(rows)
    result = registry.merge(metrics, on="track_uid", how="left", validate="one_to_one")
    result["linear_artifact_excluded"] = result["linear_mislink_remove_flag"].fillna(False).astype(bool)
    old_set = set(historic.loc[historic["linear_mislink_remove_flag"].astype(bool), "track_uid"])
    new_set = set(metrics.loc[metrics["linear_mislink_remove_flag"], "track_uid"])
    exact = old_set == new_set
    audit = pd.DataFrame(
        [
            {"metric": "historic_removed_tracks", "value": len(old_set)},
            {"metric": "recomputed_removed_tracks", "value": len(new_set)},
            {"metric": "historic_only", "value": len(old_set - new_set)},
            {"metric": "recomputed_only", "value": len(new_set - old_set)},
            {"metric": "sets_match_exactly", "value": int(exact)},
            {"metric": "user_confirmed_tracks", "value": len(user_confirmed)},
            {"metric": "auto_geometry_tracks", "value": int(metrics["auto_linear_mislink_high_confidence"].sum())},
        ]
    )
    mismatches = pd.DataFrame(
        [{"track_uid": uid, "mismatch_type": "historic_only"} for uid in sorted(old_set - new_set)]
        + [{"track_uid": uid, "mismatch_type": "recomputed_only"} for uid in sorted(new_set - old_set)]
    )
    return result, metrics, pd.concat([audit, mismatches], ignore_index=True, sort=False), exact


def sequence_descriptors(track: pd.DataFrame) -> dict[str, object]:
    if "Bound" not in track.columns:
        return {}
    states = safe_numeric(track.sort_values("Frame")["Bound"])
    valid = states[states.isin([0, 1, 2])].astype(int).to_numpy()
    total = len(states)
    if not len(valid):
        return {"n_valid_states": 0, "frac_gap": 1.0 if total else np.nan}
    counts = np.bincount(valid, minlength=3)
    fractions = counts / len(valid)
    switches = int(np.sum(valid[1:] != valid[:-1])) if len(valid) > 1 else 0
    entropy = float(-np.sum(fractions[fractions > 0] * np.log(fractions[fractions > 0])))
    return {
        "n_valid_states": int(len(valid)), "frac_diffusive": float(fractions[0]),
        "frac_constrained": float(fractions[1]), "frac_bound": float(fractions[2]),
        "frac_gap": float((total - len(valid)) / total) if total else np.nan,
        "n_state_switches": switches, "switch_rate": float(switches / (len(valid) - 1)) if len(valid) > 1 else 0.0,
        "state_entropy": entropy, "first_valid_state": int(valid[0]), "last_valid_state": int(valid[-1]),
        "dominant_state": int(np.argmax(counts)), "dominant_state_fraction": float(np.max(fractions)),
    }


def radius_of_gyration(x_um: np.ndarray, y_um: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_um - x_um.mean()) ** 2 + (y_um - y_um.mean()) ** 2)))


def convex_hull_area(x_um: np.ndarray, y_um: np.ndarray) -> float:
    if len(x_um) < 3:
        return 0.0
    try:
        return float(ConvexHull(np.column_stack([x_um, y_um])).volume)
    except Exception:
        # Repeated or exactly collinear points have a well-defined zero-area hull.
        return 0.0


class NoiseCalibration:
    def __init__(self, sigma_x: float, sigma_y: float, sigma_rms: float, hull_null: np.ndarray):
        self.sigma_x_um = sigma_x
        self.sigma_y_um = sigma_y
        self.sigma_rms_um = sigma_rms
        self.msd_offset_um2 = 2.0 * (sigma_x**2 + sigma_y**2)
        self.rg2_noise_contribution_w10_um2 = (1.0 - 1.0 / WINDOW_FRAMES) * (sigma_x**2 + sigma_y**2)
        self.hull_null_values_um2 = np.sort(hull_null)
        self.hull_null_median_um2 = float(np.median(hull_null))


def load_noise_calibration() -> NoiseCalibration:
    row = pd.read_csv(LOCALIZATION_SUMMARY).iloc[0]
    sigma_x = float(row["sigma_x_center90_um"])
    sigma_y = float(row["sigma_y_center90_um"])
    sigma_rms = float(row["rms_localization_precision_um"])
    rng = np.random.default_rng(RANDOM_SEED)
    hulls = np.empty(20000)
    for index in range(len(hulls)):
        hulls[index] = convex_hull_area(
            rng.normal(0.0, sigma_x, WINDOW_FRAMES),
            rng.normal(0.0, sigma_y, WINDOW_FRAMES),
        )
    return NoiseCalibration(sigma_x, sigma_y, sigma_rms, hulls)


def valid_coordinates(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = track.sort_values("Frame", kind="mergesort")
    x, y, frame = safe_numeric(ordered["x"]), safe_numeric(ordered["y"]), safe_numeric(ordered["Frame"])
    valid = x.notna() & y.notna() & frame.notna() & (x >= 0) & (y >= 0)
    return x[valid].to_numpy(float) * PIXEL_SIZE_UM, y[valid].to_numpy(float) * PIXEL_SIZE_UM, frame[valid].to_numpy(int)


def rolling_windows(x: np.ndarray, y: np.ndarray, frame: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(x) < WINDOW_FRAMES:
        return []
    split_points = np.where(np.diff(frame) != 1)[0] + 1
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for indices in np.split(np.arange(len(frame)), split_points):
        if len(indices) < WINDOW_FRAMES:
            continue
        starts = list(range(0, len(indices) - WINDOW_FRAMES + 1, WINDOW_STRIDE))
        final_start = len(indices) - WINDOW_FRAMES
        if starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            take = indices[start : start + WINDOW_FRAMES]
            windows.append((x[take], y[take]))
    return windows


def window_geometry_features(x: np.ndarray, y: np.ndarray, frame: np.ndarray, noise: NoiseCalibration) -> dict[str, object]:
    windows = rolling_windows(x, y, frame)
    if not windows:
        return {
            "n_geometry_windows_w10": 0, "Rg_window_median_um": np.nan,
            "convex_hull_window_median_um2": np.nan, "log_rg2_noise_ratio_w10": np.nan,
            "log_hull_noise_ratio_w10": np.nan, "rg_noise_resolved_fraction_w10": np.nan,
            "hull_noise_percentile_median_w10": np.nan,
        }
    rg = np.asarray([radius_of_gyration(a, b) for a, b in windows])
    hull = np.asarray([convex_hull_area(a, b) for a, b in windows])
    rg_ratio = rg**2 / noise.rg2_noise_contribution_w10_um2
    hull_ratio = hull / noise.hull_null_median_um2
    percentiles = np.searchsorted(noise.hull_null_values_um2, hull, side="right") / len(noise.hull_null_values_um2)
    return {
        "n_geometry_windows_w10": int(len(windows)),
        "Rg_window_median_um": float(np.nanmedian(rg)),
        "Rg_window_iqr_um": float(np.nanquantile(rg, 0.75) - np.nanquantile(rg, 0.25)),
        "convex_hull_window_median_um2": float(np.nanmedian(hull)),
        "convex_hull_window_iqr_um2": float(np.nanquantile(hull, 0.75) - np.nanquantile(hull, 0.25)),
        "log_rg2_noise_ratio_w10": float(np.nanmedian(np.log10(np.maximum(rg_ratio, 1e-12)))),
        "log_hull_noise_ratio_w10": float(np.nanmedian(np.log10(np.maximum(hull_ratio, 1e-12)))),
        "rg_noise_resolved_fraction_w10": float(np.mean(rg_ratio > 1.0)),
        "hull_noise_percentile_median_w10": float(np.nanmedian(percentiles)),
    }


def msd_by_lag(x: np.ndarray, y: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {int(f): np.asarray([a, b]) for f, a, b in zip(frame, x, y)}
    lags, msds, pairs = [], [], []
    for lag in range(1, MAX_MSD_LAG + 1):
        values = [float(np.sum((lookup[f + lag] - point) ** 2)) for f, point in lookup.items() if f + lag in lookup]
        if values:
            lags.append(lag * FRAME_INTERVAL_SECONDS)
            msds.append(float(np.mean(values)))
            pairs.append(len(values))
    return np.asarray(lags), np.asarray(msds), np.asarray(pairs)


def summarize_scale_specific_mobility(lags: np.ndarray, msds: np.ndarray, pairs: np.ndarray, noise: NoiseCalibration) -> dict[str, object]:
    if len(lags) < 3 or not np.isfinite(msds).all() or np.any(msds <= 0):
        return {"mobility_summary_success": False}
    offset = noise.msd_offset_um2
    weights = np.sqrt(pairs / np.max(pairs))
    log_lags = np.log(lags)
    log_msds = np.log(msds)
    alpha_unclipped, log_prefactor = np.polyfit(log_lags, log_msds, 1, w=weights)
    alpha_observed = float(alpha_unclipped)
    predicted = np.exp(log_prefactor) * lags**alpha_unclipped
    scale = max(float(np.median(msds)), 1e-12)
    rmse = float(np.sqrt(np.mean((predicted - msds) ** 2)))
    d_app = float(msds[0] / (4.0 * FRAME_INTERVAL_SECONDS))
    ratio = float(msds[0] / offset)
    excess = float(max(msds[0] - offset, 0.0))
    # Sensitivity-only correction: 90% of the smallest observed MSD is the largest
    # offset that keeps every corrected lag positive, capped by the fixed-cell value.
    max_compatible_offset = float(min(offset, 0.9 * float(np.min(msds))))
    corrected = msds - max_compatible_offset
    alpha_sensitivity = float(np.polyfit(log_lags, np.log(corrected), 1, w=weights)[0])
    return {
        "mobility_summary_success": bool(np.isfinite([d_app, ratio, alpha_observed]).all()),
        "D_app_10ms_um2_s": d_app,
        "log_D_app_10ms": float(np.log10(d_app)),
        "log_msd1_fixed90_ratio": float(np.log10(ratio)),
        "msd1_fixed90_ratio": ratio,
        "lag1_localization_resolved_fixed90": bool(msds[0] > offset),
        "D_excess_above_fixed90_um2_s": excess / (4.0 * FRAME_INTERVAL_SECONDS),
        "D_fixed90_localization_contribution_um2_s": offset / (4.0 * FRAME_INTERVAL_SECONDS),
        "alpha_observed": alpha_observed,
        "alpha_observed_unclipped": float(alpha_unclipped),
        "alpha_outside_nominal_0_to_2": bool(alpha_unclipped < 0.0 or alpha_unclipped > 2.0),
        "alpha_max_compatible_offset_sensitivity": alpha_sensitivity,
        "max_compatible_offset_um2": max_compatible_offset,
        "fixed90_offset_fully_compatible": bool(np.min(msds) > offset),
        "mobility_powerlaw_rmse_um2": rmse,
        "mobility_powerlaw_normalized_rmse": rmse / scale,
    }


def compute_track_features(track: pd.DataFrame, noise: NoiseCalibration) -> dict[str, object]:
    ordered = track.sort_values("Frame", kind="mergesort")
    first = ordered.iloc[0]
    x, y, frame = valid_coordinates(ordered)
    result: dict[str, object] = {
        "Video #": first["Video #"], "Video Name": first.get("Video Name", ""),
        "Cell": first["Cell"], "Track": first["Track"], "track_uid": first["track_uid"],
        "track_identifier": first["track_identifier"], "source_behavior_class": first["source_behavior_class"],
        "raw_track_length": int(len(ordered)), "n_valid_frames": int(len(x)),
        "elapsed_frame_span": int(frame.max() - frame.min() + 1) if len(frame) else 0,
    }
    result.update(sequence_descriptors(ordered))
    result.update(window_geometry_features(x, y, frame, noise))
    lags, msds, pairs = msd_by_lag(x, y, frame)
    for index in range(MAX_MSD_LAG):
        result[f"msd_lag{index + 1}_um2"] = float(msds[index]) if index < len(msds) else np.nan
        result[f"msd_lag{index + 1}_pairs"] = int(pairs[index]) if index < len(pairs) else 0
    result.update(summarize_scale_specific_mobility(lags, msds, pairs, noise))
    if len(lags) >= 2:
        result["D_forced_origin_legacy_um2_s"] = float(np.dot(lags, msds) / np.dot(lags, lags) / 4.0)
        result["alpha_loglog_legacy"] = float(np.polyfit(np.log(lags), np.log(msds), 1)[0])
    else:
        result["D_forced_origin_legacy_um2_s"], result["alpha_loglog_legacy"] = np.nan, np.nan
    result["model_feature_complete"] = bool(np.isfinite(np.asarray([result.get(c, np.nan) for c in MODEL_FEATURES], dtype=float)).all())
    return result


def build_feature_table(registry: pd.DataFrame, frames: pd.DataFrame, noise: NoiseCalibration) -> pd.DataFrame:
    rows = []
    for index, (_, track) in enumerate(frames.groupby("track_uid", sort=False), start=1):
        rows.append(compute_track_features(track, noise))
        if index % 500 == 0:
            print(f"features: {index}/{len(registry)}", flush=True)
    features = pd.DataFrame(rows)
    attach_columns = [
        "track_uid", "source_coordinate_file", "source_constrained_subgroup", "behavior_plus_constrained_subgroup",
        "linear_artifact_excluded", "linear_mislink_remove_reason", "user_confirmed_linear_mislink",
        "auto_linear_mislink_high_confidence", "pca_linearity", "line_rmse_norm", "max_step", "median_step",
    ]
    return features.merge(registry[attach_columns], on="track_uid", how="left", validate="one_to_one")


def rebuild_anchor(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = table.copy()
    eligible = (~result["linear_artifact_excluded"]) & result["model_feature_complete"]
    groups = result["behavior_plus_constrained_subgroup"].astype(str)
    single_bound = groups.eq("single_bound")
    confined = groups.eq("single_constrained__confined_low_mobility")
    candidates = groups.isin(["bound_constrained", "single_constrained__intermediate_or_less_confined"])
    log_d = pd.to_numeric(result["log_D_app_10ms"], errors="coerce")
    source = log_d[eligible & single_bound].dropna()
    q1, q3 = float(source.quantile(0.25)), float(source.quantile(0.75))
    fence = q3 + 1.5 * (q3 - q1)
    result["single_bound_high_D_outlier"] = eligible & single_bound & (log_d > fence)
    result["base_anchor_member"] = eligible & ((single_bound & ~result["single_bound_high_D_outlier"]) | confined)
    covariance = EmpiricalCovariance(assume_centered=False).fit(result.loc[result["base_anchor_member"], MODEL_FEATURES].to_numpy(float))
    result["anchor_mahalanobis_distance_squared"] = np.nan
    complete = result["model_feature_complete"]
    result.loc[complete, "anchor_mahalanobis_distance_squared"] = covariance.mahalanobis(result.loc[complete, MODEL_FEATURES].to_numpy(float))
    result["anchor_mahalanobis_distance"] = np.sqrt(result["anchor_mahalanobis_distance_squared"])
    base_d2 = result.loc[result["base_anchor_member"], "anchor_mahalanobis_distance_squared"]
    cutoffs = {q: float(base_d2.quantile(q)) for q in (0.50, 0.75, 0.90)}
    d2 = result["anchor_mahalanobis_distance_squared"]
    result["anchor_containment_layer"] = "outside_anchor_90pct"
    result.loc[d2 <= cutoffs[0.90], "anchor_containment_layer"] = "moderate_inside_90pct"
    result.loc[d2 <= cutoffs[0.75], "anchor_containment_layer"] = "strict_inside_75pct"
    result.loc[d2 <= cutoffs[0.50], "anchor_containment_layer"] = "hyper_strict_inside_50pct"
    result.loc[~complete, "anchor_containment_layer"] = "unavailable_missing_features"
    result.loc[result["linear_artifact_excluded"], "anchor_containment_layer"] = "excluded_linear_artifact"
    result["overlap_candidate_evaluated"] = eligible & candidates
    result["overlap_candidate_included"] = result["overlap_candidate_evaluated"] & (d2 <= cutoffs[0.90])
    result["anchor_core2_member"] = result["base_anchor_member"] | result["overlap_candidate_included"]
    result["reference_label"] = np.where(result["anchor_core2_member"], "chromatin_bound_like", "not_chromatin_bound_like")
    result["anchor_core2_source"] = "not_anchor_core2"
    result.loc[result["base_anchor_member"], "anchor_core2_source"] = "replacement_base_anchor"
    result.loc[result["overlap_candidate_included"], "anchor_core2_source"] = "replacement_mahalanobis_overlap"
    result.loc[result["linear_artifact_excluded"], "anchor_core2_source"] = "excluded_linear_artifact"
    thresholds = pd.DataFrame(
        [
            {"metric": "single_bound_log_D_app_q1", "value": q1},
            {"metric": "single_bound_log_D_app_q3", "value": q3},
            {"metric": "single_bound_high_D_app_upper_log10_fence", "value": fence},
            {"metric": "single_bound_high_D_app_upper_um2_s_fence", "value": 10**fence},
        ]
        + [
            {"metric": f"anchor_{int(q*100)}pct_distance_squared", "value": value}
            for q, value in cutoffs.items()
        ]
    )
    composition = (
        result.groupby("behavior_plus_constrained_subgroup", dropna=False)
        .agg(
            n_tracks_total=("track_uid", "size"),
            n_linear_artifact_excluded=("linear_artifact_excluded", "sum"),
            n_model_feature_complete=("model_feature_complete", "sum"),
            n_base_anchor=("base_anchor_member", "sum"),
            n_overlap_candidates_evaluated=("overlap_candidate_evaluated", "sum"),
            n_overlap_candidates_included=("overlap_candidate_included", "sum"),
            n_anchor_core2=("anchor_core2_member", "sum"),
        )
        .reset_index()
    )
    return result, thresholds, composition


def grouped_train(table: pd.DataFrame, model_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(RRL_ROOT))
    from rrl_track_behaviour_ml.models import MODEL_NAMES, binary_metrics, consensus_labels, make_models

    eligible = table[(~table["linear_artifact_excluded"]) & table["model_feature_complete"]].copy().reset_index(drop=True)
    x = eligible[MODEL_FEATURES].to_numpy(float)
    y = eligible["anchor_core2_member"].astype(int).to_numpy()
    groups = eligible[["Video #", "Cell"]].astype(str).agg("__".join, axis=1).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof = eligible[["track_uid", "track_identifier", "Video #", "Cell", "anchor_core2_member"]].copy()
    oof["cv_fold"] = -1
    for name in MODEL_NAMES:
        oof[f"probability_{name}"] = np.nan
    fold_rows = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        if len(np.unique(y[train_index])) != 2 or len(np.unique(y[test_index])) != 2:
            raise RuntimeError(f"Grouped fold {fold} lacks one reference class")
        fold_models = make_models(RANDOM_SEED + fold)
        oof.loc[test_index, "cv_fold"] = fold
        for name, model in fold_models.items():
            model.fit(x[train_index], y[train_index])
            probability = model.predict_proba(x[test_index])[:, 1]
            oof.loc[test_index, f"probability_{name}"] = probability
            row = binary_metrics(y[test_index], probability, name)
            row.update({"cv_fold": fold, "n_train": len(train_index), "n_test": len(test_index), "n_test_groups": len(np.unique(groups[test_index]))})
            fold_rows.append(row)
    settings = {
        "positive_probability_threshold": 0.60, "negative_probability_threshold": 0.40,
        "minimum_positive_votes": 3, "maximum_negative_votes": 1,
    }
    oof = consensus_labels(oof, settings)
    metrics = pd.DataFrame([binary_metrics(y, oof[f"probability_{name}"].to_numpy(float), name) for name in MODEL_NAMES])
    metrics = pd.concat([metrics, pd.DataFrame([binary_metrics(y, oof["ensemble_probability"].to_numpy(float), "ensemble")])], ignore_index=True)
    model_dir.mkdir(parents=True, exist_ok=False)
    final_models = make_models(RANDOM_SEED)
    for name, model in final_models.items():
        model.fit(x, y)
        joblib.dump(model, model_dir / f"RRL_{name}_model.joblib")
    probabilities = pd.DataFrame({"track_uid": eligible["track_uid"].astype(str)})
    for name, model in final_models.items():
        probabilities[f"probability_{name}"] = model.predict_proba(x)[:, 1]
    probabilities["mahalanobis_distance_squared"] = final_models["mahalanobis"].mahalanobis_distance_squared(x)
    predictions = consensus_labels(probabilities, settings)
    predictions = eligible.merge(predictions, on="track_uid", how="left", validate="one_to_one")
    return final_models, oof, metrics, pd.DataFrame(fold_rows), predictions


def save_pca(table: pd.DataFrame, feature_columns: list[str], label_column: str, output: Path, stem: str, title: str) -> pd.DataFrame:
    complete = table[np.isfinite(table[feature_columns].to_numpy(float)).all(axis=1)].copy()
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(StandardScaler().fit_transform(complete[feature_columns].to_numpy(float)))
    positive = complete[label_column].astype(str).eq("chromatin_bound_like").to_numpy()
    comparison = ~positive
    pc1_sign_flipped = False
    if positive.any() and comparison.any() and float(scores[positive, 0].mean()) > float(scores[comparison, 0].mean()):
        scores[:, 0] *= -1.0
        pca.components_[0, :] *= -1.0
        pc1_sign_flipped = True
    result = complete[["track_uid", "track_identifier", label_column]].copy()
    result["PC1"], result["PC2"] = scores[:, 0], scores[:, 1]
    result["PC1_variance_fraction"], result["PC2_variance_fraction"] = pca.explained_variance_ratio_
    result["PC1_orientation_rule"] = "chromatin_bound_like_centroid_left"
    result["PC1_sign_flipped_after_fit"] = pc1_sign_flipped
    result.to_csv(output / f"{stem}_scores.csv", index=False)
    pd.DataFrame(
        {
            "feature": feature_columns,
            "PC1_loading": pca.components_[0, :],
            "PC2_loading": pca.components_[1, :],
            "PC1_orientation_rule": "chromatin_bound_like_centroid_left",
            "PC1_sign_flipped_after_fit": pc1_sign_flipped,
        }
    ).to_csv(output / f"{stem}_loadings.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.6, 6.5), dpi=180)
    labels = CONSENSUS_ORDER if label_column == "consensus_prediction" else ["chromatin_bound_like", "not_chromatin_bound_like"]
    for label in labels:
        subset = result[result[label_column] == label]
        if len(subset):
            ax.scatter(subset.PC1, subset.PC2, s=14, alpha=0.38, color=COLORS.get(label, "#777777"), edgecolors="none", label=f"{label} (n={len(subset)})")
    ax.axhline(0, color="#9AA3AF", linewidth=0.8)
    ax.axvline(0, color="#9AA3AF", linewidth=0.8)
    ax.grid(True, color="#E1E5EB", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return result


def save_feature_plot(predictions: pd.DataFrame, output: Path) -> None:
    features = [
        ("Rg_window_median_um", "Median 10-frame Rg (um)", True),
        ("convex_hull_window_median_um2", "Median 10-frame hull area (um2)", True),
        ("D_app_10ms_um2_s", "Apparent mobility at 10 ms (um2/s)", True),
        ("alpha_observed", "Observed MSD slope alpha", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=180)
    for ax, (column, ylabel, log_scale) in zip(axes.flat, features):
        arrays, labels = [], []
        for label in CONSENSUS_ORDER:
            values = safe_numeric(predictions.loc[predictions["consensus_prediction"].eq(label), column]).dropna()
            if log_scale:
                values = values[values > 0]
            if len(values):
                arrays.append(values.to_numpy(float))
                labels.append(label)
        if arrays:
            parts = ax.violinplot(arrays, showmedians=True, showextrema=False)
            for body, label in zip(parts["bodies"], labels):
                body.set_facecolor(COLORS[label])
                body.set_alpha(0.55)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#E1E5EB")
    fig.suptitle(f"{RELEASE_NAME}: self-audit feature distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "self_audit_feature_distributions.png", bbox_inches="tight")
    fig.savefig(output / "self_audit_feature_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def feature_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = list(dict.fromkeys(["raw_track_length", "Rg_window_median_um", "convex_hull_window_median_um2", "D_app_10ms_um2_s", "alpha_observed", *MODEL_FEATURES]))
    for label, subset in predictions.groupby("consensus_prediction", dropna=False):
        for column in columns:
            values = safe_numeric(subset[column]).dropna()
            rows.append(
                {
                    "consensus_prediction": label, "feature": column, "n": len(values),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                    "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                    "minimum": float(values.min()) if len(values) else np.nan,
                    "maximum": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def apply_models(models: dict, samples: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(RRL_ROOT))
    from rrl_track_behaviour_ml.models import predict_models
    settings = {
        "positive_probability_threshold": 0.60, "negative_probability_threshold": 0.40,
        "minimum_positive_votes": 3, "maximum_negative_votes": 1,
    }
    eligible = samples[samples["model_feature_complete"]].copy()
    prediction = predict_models(models, eligible[MODEL_FEATURES].to_numpy(float), eligible["sample_uid"], settings).rename(columns={"track_uid": "sample_uid"})
    return samples.merge(prediction, on="sample_uid", how="left", validate="one_to_one")


def truncation_audit(frames: pd.DataFrame, full_table: pd.DataFrame, noise: NoiseCalibration, models: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_eligible = set(full_table.loc[(~full_table["linear_artifact_excluded"]) & full_table["model_feature_complete"], "track_uid"])
    rows = []
    cutoffs = [10, 20, 40, 80, 120]
    for uid, track in frames[frames["track_uid"].isin(full_eligible)].groupby("track_uid", sort=False):
        ordered = track.sort_values("Frame", kind="mergesort")
        if len(ordered) < 120:
            continue
        for cutoff in cutoffs:
            result = compute_track_features(ordered.iloc[:cutoff].copy(), noise)
            result["cutoff_frames"] = cutoff
            result["sample_uid"] = f"{uid}__{cutoff}"
            rows.append(result)
    samples = pd.DataFrame(rows)
    predicted = apply_models(models, samples)
    common_ids = set.intersection(*[set(predicted.loc[(predicted.cutoff_frames == cutoff) & predicted.model_feature_complete, "track_uid"]) for cutoff in cutoffs])
    common = predicted[predicted["track_uid"].isin(common_ids)].copy()
    feature_rows = []
    tracked_features = list(dict.fromkeys([*MODEL_FEATURES, "Rg_window_median_um", "convex_hull_window_median_um2", "D_app_10ms_um2_s", "alpha_observed"]))
    for cutoff, subset in common.groupby("cutoff_frames"):
        for feature in tracked_features:
            values = safe_numeric(subset[feature]).dropna()
            feature_rows.append({"cutoff_frames": cutoff, "feature": feature, "n": len(values), "mean": float(values.mean()), "median": float(values.median()), "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75))})
    summary = pd.DataFrame(feature_rows)
    wide = common.pivot(index="track_uid", columns="cutoff_frames")
    paired_rows = []
    for feature in tracked_features:
        before, after = wide[feature][10], wide[feature][120]
        valid = np.isfinite(before) & np.isfinite(after)
        delta = (after[valid] - before[valid]).to_numpy(float)
        try:
            pvalue = float(wilcoxon(delta).pvalue) if len(delta) and np.any(delta != 0) else 1.0
        except Exception:
            pvalue = np.nan
        paired_rows.append(
            {
                "feature": feature, "n": int(valid.sum()), "median_10": float(before[valid].median()),
                "median_120": float(after[valid].median()), "median_paired_delta": float(np.median(delta)),
                "mean_paired_delta": float(np.mean(delta)), "increase_fraction": float(np.mean(delta > 0)),
                "wilcoxon_pvalue": pvalue,
            }
        )
    paired = pd.DataFrame(paired_rows)
    label = common.pivot(index="track_uid", columns="cutoff_frames", values="consensus_prediction")
    probability = common.pivot(index="track_uid", columns="cutoff_frames", values="ensemble_probability")
    label_stability = pd.DataFrame(
        [
            {
                "comparison": "10_vs_120", "n_tracks": len(label),
                "stable_labels": int((label[10] == label[120]).sum()),
                "label_stability_fraction": float((label[10] == label[120]).mean()),
                "median_probability_change": float(np.median(probability[120] - probability[10])),
                "mean_probability_change": float(np.mean(probability[120] - probability[10])),
            }
        ]
    )
    common.to_csv(output / "truncation_track_level_results.csv", index=False)
    summary.to_csv(output / "truncation_feature_summary.csv", index=False)
    paired.to_csv(output / "truncation_paired_10_vs_120.csv", index=False)
    label_stability.to_csv(output / "truncation_prediction_stability.csv", index=False)
    return paired, label_stability, common


def simulate_path(scenario: str, n: int, rng: np.random.Generator, noise: NoiseCalibration) -> tuple[np.ndarray, np.ndarray]:
    substeps = 5
    dt = FRAME_INTERVAL_SECONDS / substeps
    total = n * substeps
    position = np.zeros((total, 2), dtype=float)
    state = 0
    for index in range(1, total):
        t0, t1 = (index - 1) * dt, index * dt
        if scenario == "immobile":
            increment = np.zeros(2)
        elif scenario == "brownian_D0.03":
            increment = rng.normal(0, math.sqrt(2 * 0.03 * dt), 2)
        elif scenario == "confined_D0.03_R0.08":
            increment = rng.normal(0, math.sqrt(2 * 0.03 * dt), 2)
        elif scenario == "scaled_brownian_alpha0.4":
            variance = 2 * 0.03 * max(t1**0.4 - t0**0.4, 1e-12)
            increment = rng.normal(0, math.sqrt(variance), 2)
        elif scenario == "two_state_switching":
            if rng.random() < 0.02:
                state = 1 - state
            diffusion = 0.005 if state == 0 else 0.12
            increment = rng.normal(0, math.sqrt(2 * diffusion * dt), 2)
        elif scenario == "directed":
            increment = np.asarray([0.12 * dt, 0.0]) + rng.normal(0, math.sqrt(2 * 0.01 * dt), 2)
        else:
            raise ValueError(scenario)
        position[index] = position[index - 1] + increment
        if scenario == "confined_D0.03_R0.08":
            radius = np.linalg.norm(position[index])
            if radius > 0.08:
                position[index] *= 0.08 / radius
    blurred = position.reshape(n, substeps, 2).mean(axis=1)
    blurred[:, 0] += rng.normal(0.0, noise.sigma_x_um, n)
    blurred[:, 1] += rng.normal(0.0, noise.sigma_y_um, n)
    return blurred[:, 0], blurred[:, 1]


def simulation_audit(noise: NoiseCalibration, output: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 99)
    scenarios = ["immobile", "brownian_D0.03", "confined_D0.03_R0.08", "scaled_brownian_alpha0.4", "two_state_switching", "directed"]
    rows = []
    for scenario in scenarios:
        for n in [10, 20, 40, 80, 120]:
            for replicate in range(SIMULATION_TRACKS_PER_SCENARIO_LENGTH):
                x, y = simulate_path(scenario, n, rng, noise)
                frame = np.arange(n)
                geometry = window_geometry_features(x, y, frame, noise)
                lags, msds, pairs = msd_by_lag(x, y, frame)
                mobility = summarize_scale_specific_mobility(lags, msds, pairs, noise)
                rows.append({"scenario": scenario, "track_length": n, "replicate": replicate, **geometry, **mobility})
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "simulation_track_level_results.csv", index=False)
    summary = (
        detail.groupby(["scenario", "track_length"], dropna=False)
        .agg(
            n=("replicate", "size"),
            summary_success_fraction=("mobility_summary_success", "mean"),
            median_D_app_10ms=("D_app_10ms_um2_s", "median"),
            median_log_msd1_fixed90_ratio=("log_msd1_fixed90_ratio", "median"),
            median_alpha_observed=("alpha_observed", "median"),
            fixed90_resolved_fraction=("lag1_localization_resolved_fixed90", "mean"),
            median_log_rg2_noise_ratio=("log_rg2_noise_ratio_w10", "median"),
            median_log_hull_noise_ratio=("log_hull_noise_ratio_w10", "median"),
            median_alpha_offset_sensitivity=("alpha_max_compatible_offset_sensitivity", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output / "simulation_recovery_summary.csv", index=False)
    return summary


def model_gate_table(mislink_exact: bool, anchor: pd.DataFrame, metrics: pd.DataFrame, paired: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    lookup = paired.set_index("feature")
    ensemble_auc = float(metrics.loc[metrics.model.eq("ensemble"), "roc_auc"].iloc[0])
    gates = [
        ("mislink_protocol_exact_match", mislink_exact, "required exact set match"),
        ("model_eligible_tracks_at_least_3500", int(((~anchor.linear_artifact_excluded) & anchor.model_feature_complete).sum()) >= 3500, ">=3500"),
        ("anchor_members_at_least_500", int(anchor.anchor_core2_member.sum()) >= 500, ">=500"),
        ("grouped_cv_ensemble_auc", ensemble_auc >= 0.80, ">=0.80"),
        ("apparent_mobility_log10_median_drift", abs(float(lookup.loc["log_msd1_fixed90_ratio", "median_paired_delta"])) <= 0.15, "absolute median <=0.15"),
        ("alpha_observed_median_drift", abs(float(lookup.loc["alpha_observed", "median_paired_delta"])) <= 0.10, "absolute median <=0.10"),
        ("Rg_ratio_median_drift", abs(float(lookup.loc["log_rg2_noise_ratio_w10", "median_paired_delta"])) <= 0.25, "absolute median <=0.25"),
        ("hull_ratio_median_drift", abs(float(lookup.loc["log_hull_noise_ratio_w10", "median_paired_delta"])) <= 0.25, "absolute median <=0.25"),
        ("prediction_label_stability", float(stability.label_stability_fraction.iloc[0]) >= 0.70, ">=0.70"),
    ]
    return pd.DataFrame([{"gate": name, "passed": bool(passed), "criterion": criterion} for name, passed, criterion in gates])


def write_report(output: Path, noise: NoiseCalibration, anchor: pd.DataFrame, metrics: pd.DataFrame, paired: pd.DataFrame, stability: pd.DataFrame, gates: pd.DataFrame, safety_ok: bool, simulation: pd.DataFrame) -> None:
    eligible = anchor[(~anchor.linear_artifact_excluded) & anchor.model_feature_complete]
    status = "active_replacement" if bool(gates.passed.all()) and safety_ok else "candidate_not_promoted"
    lag1_resolved = int(anchor["lag1_localization_resolved_fixed90"].sum())
    alpha_outside = int(anchor["alpha_outside_nominal_0_to_2"].sum())
    brownian_120 = simulation[(simulation.scenario == "brownian_D0.03") & (simulation.track_length == 120)].iloc[0]
    gate_lines = "\n".join(f"| {row.gate} | {'PASS' if row.passed else 'FAIL'} | {row.criterion} |" for row in gates.itertuples())
    metric_lines = "\n".join(f"| {row.model} | {row.roc_auc:.3f} | {row.average_precision:.3f} | {row.balanced_accuracy:.3f} |" for row in metrics.itertuples())
    drift_lines = "\n".join(f"| {row.feature} | {row.n} | {row.median_paired_delta:.4g} | {row.increase_fraction:.3f} | {row.wilcoxon_pvalue:.3g} |" for row in paired.itertuples())
    text = f"""# {RELEASE_NAME}

## Status

`{status}`

The historical H2B reference remains physically unchanged. This release supersedes it operationally only when every promotion gate passes.

## Inputs and calibration

- HAP1-H2B source: `{HAP1_ROOT}`
- Pixel conversion: `{PIXEL_SIZE_UM} um/pixel`
- Frame interval: `{FRAME_INTERVAL_MS} ms/frame`
- Exposure used in simulations: `{EXPOSURE_SECONDS*1000:g} ms`
- Fixed-cell central-90 sigma_x: `{noise.sigma_x_um:.10f} um`
- Fixed-cell central-90 sigma_y: `{noise.sigma_y_um:.10f} um`
- RMS sigma: `{noise.sigma_rms_um:.10f} um`
- Two-position 2D MSD offset: `{noise.msd_offset_um2:.10f} um2`

The build treats the central-90 values as the fixed-cell calibration. Many live H2B lag-1 MSD values lie below the corresponding fixed-cell offset, so blind subtraction would create negative values and a joint four-lag fit of offset plus alpha is not identifiable enough for operational use. The release therefore reports the lag-1 apparent mobility and normalizes its MSD to the fixed-cell offset. It also reports whether motion clears that fixed-90 floor, the non-negative excess above the floor, and an alpha sensitivity calculation using the largest compatible offset. No incompatible offset is silently subtracted.

## Replacement features

The model uses four features without a pooled track-length regression:

1. Median 10-frame log10 Rg-squared relative to the anisotropic localization-noise expectation.
2. Median 10-frame log10 convex-hull area relative to an immobile-particle Monte Carlo null using sigma_x and sigma_y.
3. Lag-1 MSD relative to the fixed-cell central-90 two-position offset in log10 units (equivalent to a fixed shift of log apparent mobility at 10 ms).
4. The raw observed first-four-lag log-MSD slope alpha. It is neither track-length residualized nor clipped; values outside 0 to 2 are flagged for audit.

Geometry uses rolling 10-frame windows with five-frame stride. Apparent mobility is `MSD(10 ms)/(4*0.010 s)` and is not claimed to be a free-Brownian diffusion coefficient. Alpha is not length-residualized. The model preserves sequence descriptors for audit but does not use them as predictors because reference labels originate from behavior classes.

This treatment follows the localization-error and motion-blur cautions described by [Berglund (2010)](https://pubmed.ncbi.nlm.nih.gov/20866658/), [Vestergaard et al. (2014)](https://doi.org/10.1103/PhysRevE.89.022726), [Kepten et al. (2015)](https://doi.org/10.1371/journal.pone.0117722), and [Calderon (2016)](https://pubmed.ncbi.nlm.nih.gov/27301001/): localization uncertainty must be represented, but an externally estimated offset should not be forced onto trajectories that are incompatible with it.

## Mislink protocol

The build re-applied the 2026-08-18 protocol before feature-space anchor construction: user-confirmed list OR pca_linearity >= 0.9999, line_rmse_norm <= 0.008, max_step >= 100 pixels, and at least four points.

## Counts

- Tracks loaded: {len(anchor):,}
- Mislink exclusions: {int(anchor.linear_artifact_excluded.sum()):,}
- Model-eligible tracks: {len(eligible):,}
- Replacement Anchor Core 2 members: {int(anchor.anchor_core2_member.sum()):,}
- Truncation common cohort: {int(stability.n_tracks.iloc[0]):,}
- Lag-1 MSD above the fixed-cell central-90 offset: {lag1_resolved:,} / {len(anchor):,} ({lag1_resolved/len(anchor):.1%})
- Raw alpha outside the nominal 0-to-2 interval (retained and flagged): {alpha_outside:,} / {len(anchor):,} ({alpha_outside/len(anchor):.1%})

## Grouped cross-validation

Folds hold out complete video-cell groups. These metrics measure reproducibility of the operational labels, not biochemical ground truth.

| Model | ROC AUC | Average precision | Balanced accuracy |
| --- | ---: | ---: | ---: |
{metric_lines}

## Paired 10-to-120-frame audit

| Feature | N | Median paired change | Fraction increasing | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: |
{drift_lines}

Prediction label stability: `{float(stability.label_stability_fraction.iloc[0]):.3f}`. Median ensemble-probability change: `{float(stability.median_probability_change.iloc[0]):.4f}`.

## Promotion gates

| Gate | Result | Criterion |
| --- | --- | --- |
{gate_lines}

Safety snapshot unchanged: `{'PASS' if safety_ok else 'FAIL'}`.

## Simulation coverage

The simulation audit includes immobile, Brownian, confined Brownian, scaled-Brownian subdiffusion, two-state switching, and directed motion. It uses five acquisition substeps per frame, full-frame averaging for motion blur, and the fixed-cell anisotropic localization noise. It is a calibration stress test rather than proof that one simulated process generated the H2B trajectories. Simulation summary rows: {len(simulation)}.

For the 120-frame Brownian D=0.03 um2/s condition, the median raw observed alpha was `{float(brownian_120.median_alpha_observed):.3f}`, not 1, after adding the measured localization noise. The maximum-compatible-offset sensitivity result is provided only as an audit bound; it is not used as a model feature.

## Critical limitations

- `D_app_10ms_um2_s` is an apparent 10-ms mobility index, not a claim of free-Brownian D.
- Raw alpha is descriptive. A low raw value alone does not prove biological subdiffusion because localization noise and motion blur can depress the observed short-lag slope.
- The fixed-cell central-90 precision is a conservative empirical calibration for this release, but fixed-cell and live-cell localization precision need not be identical.
- Cross-validation measures reproduction of the operational anchor labels, not biochemical chromatin-binding truth.

## Interpretation

Labels remain operational H2B mobility-reference labels. They do not prove biochemical chromatin binding. Tracks outside the reference do not automatically indicate condensate localization.

## Safety

The build created a new release folder. It did not modify the localization-error folder, HAP1 source files, historical analysis outputs, the current Anchor Core 2, or its trained models.
"""
    (output / "TRAINING_SET_BUILD_REPORT.md").write_text(text, encoding="utf-8")
    status_payload = {
        "release_name": RELEASE_NAME,
        "status": status,
        "promotion_gates_passed": bool(gates.passed.all()),
        "source_safety_snapshot_unchanged": bool(safety_ok),
        "supersedes": str(OLD_REFERENCE) if status == "active_replacement" else None,
        "historical_reference_modified": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (output / "RELEASE_STATUS.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    (output / "SUPERSEDES.json").write_text(
        json.dumps(
            {
                "new_release": RELEASE_NAME,
                "legacy_reference": str(OLD_REFERENCE),
                "legacy_self_test": str(OLD_SELF_TEST),
                "legacy_files_deleted_or_changed": False,
                "operational_replacement_active": status == "active_replacement",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-visual-videos", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    for folder in ["self_audit", "truncation_validation", "simulation_validation", "scripts", "audit"]:
        (output / folder).mkdir()

    localization_before = tree_stat_snapshot(LOCALIZATION_ROOT)
    protected_paths = [HAP1_ROOT / filename for filename in BEHAVIOR_FILES.values()] + [REGROUPED_ASSIGNMENTS, MISLINK_LIST, MISLINK_REPORT, LOCALIZATION_SUMMARY, LOCALIZATION_METHOD_SCRIPT, LOCALIZATION_METHOD_DECK]
    source_manifest_before = pd.DataFrame([file_record(path, "input") for path in protected_paths])
    source_manifest_before.to_csv(output / "audit" / "source_manifest_before.csv", index=False)

    print("Loading HAP1 inputs", flush=True)
    registry, frames = load_inputs()
    print("Re-applying linear-mislink protocol", flush=True)
    registry, geometry_metrics, mislink_audit, mislink_exact = apply_mislink_protocol(registry, frames)
    geometry_metrics.to_csv(output / "audit" / "mislink_geometry_metrics.csv", index=False)
    mislink_audit.to_csv(output / "audit" / "mislink_protocol_audit.csv", index=False)

    noise = load_noise_calibration()
    calibration = pd.DataFrame(
        [
            {
                "sigma_x_center90_um": noise.sigma_x_um,
                "sigma_y_center90_um": noise.sigma_y_um,
                "sigma_rms_center90_um": noise.sigma_rms_um,
                "msd_offset_2d_two_position_um2": noise.msd_offset_um2,
                "rg2_noise_contribution_w10_um2": noise.rg2_noise_contribution_w10_um2,
                "hull_null_median_w10_um2": noise.hull_null_median_um2,
                "pixel_size_um": PIXEL_SIZE_UM,
                "frame_interval_ms": FRAME_INTERVAL_MS,
                "window_frames": WINDOW_FRAMES,
                "window_stride": WINDOW_STRIDE,
                "motion_blur_coefficient_simulation": MOTION_BLUR_COEFFICIENT,
                "source": str(LOCALIZATION_SUMMARY),
            }
        ]
    )
    calibration.to_csv(output / "localization_error_calibration.csv", index=False)
    pd.DataFrame({"hull_area_um2": noise.hull_null_values_um2}).to_csv(output / "simulation_validation" / "immobile_hull_null_w10.csv", index=False)

    print("Computing replacement features", flush=True)
    features = build_feature_table(registry, frames, noise)
    features.to_csv(output / "replacement_track_feature_table.csv", index=False)
    print("Rebuilding operational anchor", flush=True)
    anchor, thresholds, composition = rebuild_anchor(features)
    anchor.to_csv(output / "anchor_core2_replacement_assignments.csv", index=False)
    thresholds.to_csv(output / "anchor_core2_replacement_thresholds.csv", index=False)
    composition.to_csv(output / "anchor_core2_replacement_composition.csv", index=False)

    old = pd.read_csv(OLD_REFERENCE / "H2B_model_training_table_linear_artifacts_removed.csv", usecols=["track_uid", "anchor_core2_member"])
    old["track_uid"] = old["track_uid"].astype(str)
    membership = anchor[["track_uid", "anchor_core2_member"]].merge(old, on="track_uid", how="outer", suffixes=("_replacement", "_legacy"), indicator=True)
    membership.to_csv(output / "audit" / "legacy_vs_replacement_membership.csv", index=False)
    contingency = membership[membership._merge.eq("both")].groupby(["anchor_core2_member_legacy", "anchor_core2_member_replacement"]).size().reset_index(name="n_tracks")
    contingency.to_csv(output / "audit" / "legacy_vs_replacement_membership_contingency.csv", index=False)

    print("Training grouped models", flush=True)
    models, oof, metrics, fold_metrics, predictions = grouped_train(anchor, output / "models")
    oof.to_csv(output / "grouped_out_of_fold_predictions.csv", index=False)
    metrics.to_csv(output / "grouped_cross_validation_metrics.csv", index=False)
    fold_metrics.to_csv(output / "grouped_cross_validation_fold_metrics.csv", index=False)
    predictions.to_csv(output / "self_audit" / "self_test_predictions.csv", index=False)
    feature_summary(predictions).to_csv(output / "self_audit" / "self_test_feature_dynamics_summary.csv", index=False)
    predictions.consensus_prediction.value_counts().reindex(CONSENSUS_ORDER, fill_value=0).rename_axis("consensus_prediction").reset_index(name="n_tracks").assign(proportion=lambda x: x.n_tracks / x.n_tracks.sum()).to_csv(output / "self_audit" / "self_test_consensus_proportions.csv", index=False)
    save_pca(predictions, MODEL_FEATURES, "consensus_prediction", output / "self_audit", "self_test_replacement_feature_PCA", f"{RELEASE_NAME}: replacement-feature PCA")
    save_pca(predictions, ["Rg_window_median_um", "convex_hull_window_median_um2", "D_app_10ms_um2_s", "alpha_observed"], "consensus_prediction", output / "self_audit", "self_test_physical_feature_PCA", f"{RELEASE_NAME}: physical-feature PCA")
    save_feature_plot(predictions, output / "self_audit")

    if not args.skip_visual_videos:
        sys.path.insert(0, str(RRL_ROOT))
        from rrl_track_behaviour_ml.visualization import draw_track_checks, select_visual_check_tracks, visual_check_group_summary
        selected = select_visual_check_tracks(predictions, 24, RANDOM_SEED, 8)
        selected.to_csv(output / "self_audit" / "visual_check_selection.csv", index=False)
        manifest = draw_track_checks(
            frames,
            selected,
            output / "self_audit" / "visual_checks",
            {"visual_check_tick_step_pixels": 10.0, "visual_check_minimum_span_pixels": 40.0, "video_playback_fps": 10.0, "video_tail_frames": 10, "video_context_frames": 4},
            acquisition_settings={"pixel_size_um": PIXEL_SIZE_UM, "frame_interval_ms": FRAME_INTERVAL_MS},
        )
        manifest.to_csv(output / "self_audit" / "visual_check_manifest.csv", index=False)
        visual_check_group_summary(predictions, selected, 8).to_csv(output / "self_audit" / "visual_check_group_summary.csv", index=False)

    print("Running paired truncation audit", flush=True)
    paired, stability, truncation_detail = truncation_audit(frames, anchor, noise, models, output / "truncation_validation")
    print("Running controlled simulation audit", flush=True)
    simulation = simulation_audit(noise, output / "simulation_validation")

    gates = model_gate_table(mislink_exact, anchor, metrics, paired, stability)
    gates.to_csv(output / "audit" / "promotion_gates.csv", index=False)
    source_manifest_after = pd.DataFrame([file_record(path, "input_after") for path in protected_paths])
    source_manifest_after.to_csv(output / "audit" / "source_manifest_after.csv", index=False)
    localization_after = tree_stat_snapshot(LOCALIZATION_ROOT)
    manifest_same = source_manifest_before[["path", "bytes", "modified_time_ns", "sha256"]].reset_index(drop=True).equals(source_manifest_after[["path", "bytes", "modified_time_ns", "sha256"]].reset_index(drop=True))
    localization_same = localization_before == localization_after
    safety_ok = bool(manifest_same and localization_same)
    pd.DataFrame(
        [
            {"check": "key_input_hashes_and_metadata_unchanged", "passed": manifest_same},
            {"check": "localization_error_tree_size_and_mtime_unchanged", "passed": localization_same},
            {"check": "historical_reference_output_untouched_by_design", "passed": True},
        ]
    ).to_csv(output / "audit" / "source_safety_audit.csv", index=False)

    model_manifest = {
        "release_name": RELEASE_NAME,
        "model_features": MODEL_FEATURES,
        "pixel_size_um": PIXEL_SIZE_UM,
        "frame_interval_seconds": FRAME_INTERVAL_SECONDS,
        "exposure_seconds_simulation": EXPOSURE_SECONDS,
        "sigma_x_center90_um": noise.sigma_x_um,
        "sigma_y_center90_um": noise.sigma_y_um,
        "msd_offset_um2": noise.msd_offset_um2,
        "geometry_window_frames": WINDOW_FRAMES,
        "geometry_window_stride": WINDOW_STRIDE,
        "random_seed": RANDOM_SEED,
        "training_tracks": int(((~anchor.linear_artifact_excluded) & anchor.model_feature_complete).sum()),
        "positive_reference_tracks": int(anchor.anchor_core2_member.sum()),
        "operational_labels_only": True,
    }
    (output / "MODEL_MANIFEST.json").write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), output / "scripts" / Path(__file__).name)
    write_report(output, noise, anchor, metrics, paired, stability, gates, safety_ok, simulation)
    print(json.dumps(json.loads((output / "RELEASE_STATUS.json").read_text(encoding="utf-8")), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
