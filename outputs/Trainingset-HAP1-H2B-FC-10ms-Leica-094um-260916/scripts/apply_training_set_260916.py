"""Apply the 260916 replacement HAP1-H2B training set to SMol-FIESTA CSVs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_builder(release: Path):
    path = release / "scripts" / "build_training_set_260916_complete.py"
    spec = importlib.util.spec_from_file_location("training_set_260916_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import release builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reviewed_track_ids(path: Path | None, builder) -> set[str]:
    if path is None:
        return set()
    table = pd.read_csv(path)
    if "track_uid" in table:
        return set(table["track_uid"].astype(str).str.strip())
    if "track_identifier" in table:
        extracted = table["track_identifier"].astype(str).str.extract(r"V(.+?)_C(.+?)_T(.+)$")
        return {
            builder.make_track_uid(v, c, t)
            for v, c, t in extracted.dropna().itertuples(index=False, name=None)
        }
    required = {"Video #", "Cell", "Track"}
    if required.issubset(table.columns):
        return {
            builder.make_track_uid(row["Video #"], row["Cell"], row["Track"])
            for _, row in table.iterrows()
        }
    raise ValueError("Reviewed mislink CSV needs track_uid, track_identifier, or Video #/Cell/Track")


def attach_mislink_flags(registry: pd.DataFrame, frames: pd.DataFrame, reviewed: set[str], builder) -> pd.DataFrame:
    rows = []
    for uid, track in frames.groupby("track_uid", sort=False):
        values = builder.trajectory_geometry(track)
        values["track_uid"] = uid
        rows.append(values)
    metrics = pd.DataFrame(rows)
    metrics["user_confirmed_linear_mislink"] = metrics["track_uid"].isin(reviewed)
    metrics["auto_linear_mislink_high_confidence"] = (
        (metrics["pca_linearity"] >= 0.9999)
        & (metrics["line_rmse_norm"] <= 0.008)
        & (metrics["max_step"] >= 100.0)
        & (metrics["geometry_n_points"] >= 4)
    )
    metrics["linear_mislink_remove_flag"] = (
        metrics["user_confirmed_linear_mislink"]
        | metrics["auto_linear_mislink_high_confidence"]
    )
    metrics["linear_mislink_remove_reason"] = np.select(
        [
            metrics["user_confirmed_linear_mislink"] & metrics["auto_linear_mislink_high_confidence"],
            metrics["user_confirmed_linear_mislink"],
            metrics["auto_linear_mislink_high_confidence"],
        ],
        ["reviewed_and_auto", "reviewed", "auto_geometry"],
        default="not_removed",
    )
    result = registry.merge(metrics, on="track_uid", how="left", validate="one_to_one")
    result["linear_artifact_excluded"] = result["linear_mislink_remove_flag"].fillna(False).astype(bool)
    result["source_constrained_subgroup"] = "not_available_target_application"
    result["behavior_plus_constrained_subgroup"] = result["source_behavior_class"].astype(str)
    return result


def release_noise(release: Path, builder):
    row = pd.read_csv(release / "localization_error_calibration.csv").iloc[0]
    hull = pd.read_csv(release / "simulation_validation" / "immobile_hull_null_w10.csv")["hull_area_um2"].to_numpy(float)
    return builder.NoiseCalibration(
        float(row["sigma_x_center90_um"]),
        float(row["sigma_y_center90_um"]),
        float(row["sigma_rms_center90_um"]),
        hull,
    )


def feature_summary(predictions: pd.DataFrame, builder) -> pd.DataFrame:
    return builder.feature_summary(predictions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--additional-mislink-list")
    parser.add_argument("--skip-visual-videos", action="store_true")
    parser.add_argument("--runtime-options", default="{}")
    args = parser.parse_args()
    runtime = json.loads(args.runtime_options)

    release = Path(__file__).resolve().parent.parent
    input_dir = Path(args.input_dir).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / f"{args.dataset_name}_RRL_replacement_260916_analysis"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    (output / "visual_checks").mkdir()

    builder = load_builder(release)
    sys.path.insert(0, str(builder.RRL_ROOT))
    from rrl_track_behaviour_ml.features import load_behavior_coordinate_files
    from rrl_track_behaviour_ml.models import load_models, predict_models

    registry, frames = load_behavior_coordinate_files(input_dir, builder.BEHAVIOR_FILES)
    reviewed = reviewed_track_ids(
        Path(args.additional_mislink_list).expanduser().resolve() if args.additional_mislink_list else None,
        builder,
    )
    registry = attach_mislink_flags(registry, frames, reviewed, builder)
    noise = release_noise(release, builder)
    features = builder.build_feature_table(registry, frames, noise)
    features["model_eligible"] = (~features["linear_artifact_excluded"]) & features["model_feature_complete"]
    features.to_csv(output / f"{args.dataset_name}_replacement_track_features.csv", index=False)

    eligible = features[features["model_eligible"]].copy().reset_index(drop=True)
    settings = {
        "positive_probability_threshold": 0.60,
        "negative_probability_threshold": 0.40,
        "minimum_positive_votes": 3,
        "maximum_negative_votes": 1,
    }
    models = load_models(release / "models")
    predictions = predict_models(
        models,
        eligible[builder.MODEL_FEATURES].to_numpy(float),
        eligible["track_uid"],
        settings,
    )
    predictions = eligible.merge(predictions, on="track_uid", how="left", validate="one_to_one")
    predictions.to_csv(output / f"{args.dataset_name}_replacement_predictions.csv", index=False)
    proportions = (
        predictions["consensus_prediction"]
        .value_counts()
        .reindex(builder.CONSENSUS_ORDER, fill_value=0)
        .rename_axis("consensus_prediction")
        .reset_index(name="n_tracks")
    )
    proportions["proportion"] = proportions["n_tracks"] / proportions["n_tracks"].sum()
    proportions.to_csv(output / f"{args.dataset_name}_replacement_consensus_proportions.csv", index=False)
    feature_summary(predictions, builder).to_csv(
        output / f"{args.dataset_name}_replacement_feature_dynamics_summary.csv", index=False
    )
    builder.save_pca(
        predictions,
        builder.MODEL_FEATURES,
        "consensus_prediction",
        output,
        f"{args.dataset_name}_replacement_feature_PCA",
        f"{args.dataset_name}: replacement-feature PCA",
    )

    visual_count = 0
    if not args.skip_visual_videos:
        from rrl_track_behaviour_ml.visualization import draw_track_checks, select_visual_check_tracks, visual_check_group_summary

        review_options = runtime.get('model', {})
        minimum = int(review_options.get('visual_check_minimum_per_group', 8))
        selected = select_visual_check_tracks(predictions, int(review_options.get('visual_check_sample_size', 24)), builder.RANDOM_SEED, minimum)
        selected.to_csv(output / "visual_check_selection.csv", index=False)
        manifest = draw_track_checks(
            frames,
            selected,
            output / "visual_checks",
            {
                "visual_check_tick_step_pixels": 10.0,
                "visual_check_minimum_span_pixels": 40.0,
                "video_playback_fps": 10.0,
                "video_tail_frames": 10,
                "video_context_frames": 4,
                **runtime.get('visualization', {}),
            },
            acquisition_settings={"pixel_size_um": builder.PIXEL_SIZE_UM, "frame_interval_ms": builder.FRAME_INTERVAL_MS},
        )
        manifest.to_csv(output / "visual_check_manifest.csv", index=False)
        visual_check_group_summary(predictions, selected, minimum).to_csv(output / "visual_check_group_summary.csv", index=False)
        visual_count = len(manifest)

    manifest = {
        "dataset_name": args.dataset_name,
        "input_dir": str(input_dir),
        "output_dir": str(output),
        "training_release": str(release),
        "release_status": json.loads((release / "RELEASE_STATUS.json").read_text(encoding="utf-8"))["status"],
        "model_features": builder.MODEL_FEATURES,
        "tracks_loaded": len(features),
        "linear_artifacts_excluded": int(features["linear_artifact_excluded"].sum()),
        "model_eligible_tracks": len(predictions),
        "visual_checks": visual_count,
        "legacy_residualizer_used": False,
    }
    (output / "APPLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
