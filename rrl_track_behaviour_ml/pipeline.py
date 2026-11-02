from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .anchor import rebuild_anchor_core2
from .config import resolve_path
from .features import (
    MODEL_FEATURES,
    PHYSICAL_FEATURES,
    add_log_features,
    add_residualized_features,
    add_track_identity,
    compute_track_features,
    fit_length_residualizers,
    load_behavior_coordinate_files,
)
from .models import MODEL_NAMES, load_models, predict_models, train_models_with_oof
from .replacement_reference import (
    apply_replacement_reference,
    is_replacement_backend,
    validate_replacement_reference,
)
from .visualization import (
    draw_track_checks,
    save_application_pca_visualizations,
    save_feature_distribution_plot,
    save_pca_visualization,
    select_visual_check_tracks,
    visual_check_group_summary,
)


POSITIVE_LABEL = "chromatin_bound_like"
NEGATIVE_LABEL = "not_chromatin_bound_like"


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_summary(table: pd.DataFrame, group_column: str) -> pd.DataFrame:
    features = PHYSICAL_FEATURES + MODEL_FEATURES + ["raw_track_length"]
    rows = []
    for group, subset in table.groupby(group_column, dropna=False):
        for feature in features:
            values = pd.to_numeric(subset[feature], errors="coerce").dropna()
            rows.append({
                "grouping": group_column,
                "group": group,
                "feature": feature,
                "n_tracks": len(subset),
                "n_finite": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "standard_deviation": values.std(ddof=1),
                "minimum": values.min(),
                "q05": values.quantile(0.05),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "q95": values.quantile(0.95),
                "maximum": values.max(),
            })
    return pd.DataFrame(rows)


def _proportion_table(predictions: pd.DataFrame, column: str, denominator_name: str) -> pd.DataFrame:
    counts = predictions[column].fillna("unavailable").value_counts(dropna=False).rename_axis("category").reset_index(name="n_tracks")
    counts["percentage"] = counts["n_tracks"] / len(predictions) * 100.0
    counts.insert(0, "result_column", column)
    counts.insert(1, "denominator", denominator_name)
    return counts


def _load_track_id_set(path: Path | None) -> set[str]:
    if path is None:
        return set()
    table = pd.read_csv(path)
    if "track_uid" not in table.columns:
        table = add_track_identity(table)
    return set(table["track_uid"].astype(str))


def _compute_feature_table(frames: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for uid, track in frames.groupby("track_uid", sort=False):
        row = compute_track_features(track, config["feature_calculation"], config["linear_artifact_rule"])
        row["track_uid"] = uid
        row["track_identifier"] = track["track_identifier"].iloc[0]
        row["source_behavior_class"] = track["source_behavior_class"].iloc[0]
        rows.append(row)
    return add_log_features(pd.DataFrame(rows))


def _input_manifest(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (behavior, path), subset in registry.groupby(["source_behavior_class", "source_coordinate_file"]):
        rows.append({
            "source_behavior_class": behavior,
            "input_file_name": Path(path).name,
            "input_file_path": str(Path(path).resolve()),
            "n_tracks": len(subset),
            "n_frame_rows": int(subset["source_frame_rows"].sum()),
        })
    return pd.DataFrame(rows)


def _save_visual_checks(frames: pd.DataFrame, predictions: pd.DataFrame, output: Path, prefix: str, config: dict):
    minimum = max(8, int(config["model"].get("visual_check_minimum_per_group", 8)))
    selected = select_visual_check_tracks(
        predictions, int(config["model"].get("visual_check_sample_size", 24)),
        int(config["model"]["random_seed"]), minimum_per_group=minimum,
    )
    manifest = draw_track_checks(
        frames, selected, output / "random_visual_checks", config["visualization"],
        acquisition_settings=config["feature_calculation"],
    )
    summary = visual_check_group_summary(predictions, selected, minimum)
    selected.to_csv(output / f"{prefix}_random_visual_check_selection.csv", index=False)
    manifest.to_csv(output / f"{prefix}_random_visual_check_manifest.csv", index=False)
    summary.to_csv(output / f"{prefix}_random_visual_check_group_summary.csv", index=False)
    return manifest, summary


def _pca_comparison(prior: pd.DataFrame, corrected: pd.DataFrame, output: Path) -> pd.DataFrame:
    prior_subset = prior[["track_uid"] + MODEL_FEATURES].drop_duplicates("track_uid")
    merged = corrected[["track_uid"] + MODEL_FEATURES].merge(
        prior_subset, on="track_uid", how="inner", suffixes=("_corrected", "_prior")
    )
    prior_features = [f"{column}_prior" for column in MODEL_FEATURES]
    corrected_features = [f"{column}_corrected" for column in MODEL_FEATURES]
    finite = np.isfinite(merged[prior_features + corrected_features].to_numpy(float)).all(axis=1)
    merged = merged[finite].copy().reset_index(drop=True)
    old_pca = PCA(n_components=2, random_state=0)
    new_pca = PCA(n_components=2, random_state=0)
    old_scores = old_pca.fit_transform(StandardScaler().fit_transform(merged[prior_features]))
    new_scores = new_pca.fit_transform(StandardScaler().fit_transform(merged[corrected_features]))
    summary = []
    for index in range(2):
        correlation = float(np.corrcoef(old_scores[:, index], new_scores[:, index])[0, 1])
        if correlation < 0:
            new_scores[:, index] *= -1
            correlation *= -1
        merged[f"prior_PC{index + 1}"] = old_scores[:, index]
        merged[f"corrected_PC{index + 1}"] = new_scores[:, index]
        summary.append({
            "component": f"PC{index + 1}",
            "n_matched_tracks": len(merged),
            "prior_variance_fraction": old_pca.explained_variance_ratio_[index],
            "corrected_variance_fraction": new_pca.explained_variance_ratio_[index],
            "score_correlation_after_sign_alignment": correlation,
            "median_absolute_score_change": float(np.median(np.abs(old_scores[:, index] - new_scores[:, index]))),
        })
    merged.to_csv(output / "H2B_prior_mixed_scale_vs_corrected_residualized_PCA_scores.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(output / "H2B_prior_mixed_scale_vs_corrected_residualized_PCA_summary.csv", index=False)
    return result


def _pixel_size_audit(prior: pd.DataFrame, corrected: pd.DataFrame, output: Path) -> pd.DataFrame:
    retained = ["track_uid", "behavior_class", "Rg", "convex_hull_area", "D", "alpha"] + MODEL_FEATURES
    retained = [column for column in retained if column in prior.columns]
    merged = corrected.merge(prior[retained].drop_duplicates("track_uid"), on="track_uid", how="inner", suffixes=("_corrected", "_prior"))
    rows = []
    for behavior, subset in merged.groupby("source_behavior_class"):
        for feature in ["Rg", "convex_hull_area", "D", "alpha"]:
            old = pd.to_numeric(subset[f"{feature}_prior"], errors="coerce")
            new = pd.to_numeric(subset[f"{feature}_corrected"], errors="coerce")
            valid = np.isfinite(old) & np.isfinite(new) & (old != 0)
            rows.append({
                "source_behavior_class": behavior,
                "feature": feature,
                "n_matched_finite": int(valid.sum()),
                "median_corrected_to_prior_ratio": float(np.median(new[valid] / old[valid])) if valid.any() else np.nan,
                "median_prior_value": float(np.median(old[valid])) if valid.any() else np.nan,
                "median_corrected_value": float(np.median(new[valid])) if valid.any() else np.nan,
            })
    result = pd.DataFrame(rows)
    result.to_csv(output / "H2B_pixel_size_correction_audit_by_behavior.csv", index=False)
    return result


def _write_reference_report(
    output: Path,
    registry: pd.DataFrame,
    training: pd.DataFrame,
    exclusions: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    pca_comparison: pd.DataFrame,
    membership_audit: dict[str, float | int],
    visual_manifest: pd.DataFrame,
    visual_summary: pd.DataFrame,
    config: dict,
) -> None:
    label_counts = training["reference_label"].value_counts()
    consensus_counts = predictions["consensus_prediction"].value_counts()
    settings = config["feature_calculation"]
    lines = [
        "# RRL Track Behaviour ML: Corrected H2B Reference Build",
        "", "## Purpose", "",
        "This standalone pipeline learns similarity to the operational H2B Anchor Core 2 reference. It does not establish biological ground truth or identify confirmed condensates.",
        "", "## Acquisition Settings", "",
        f"- Pixel size: {settings['pixel_size_um']} um/pixel",
        f"- Canonical frame interval: {settings['frame_interval_ms']} ms/frame",
        f"- Derived frame interval: {settings['frame_interval_seconds']} s/frame",
        f"- Derived frame rate: {settings['frame_rate_fps']} frames/s",
        "",
        "All seven behavior classes were recomputed from raw coordinates using the same corrected acquisition scale. The prior unified table mixed constrained-track features calculated at 0.094 um/pixel with other behavior classes calculated at 0.160 um/pixel; its residualized PCA is superseded.",
        "", "## Data and Quality Filter", "",
        f"- Full auditable H2B registry: {len(registry):,} tracks",
        f"- Automatic linear-artifact flags: {int(registry['linear_artifact_rule_flag'].sum()):,}",
        f"- Total audited linear-artifact exclusions: {int(registry['linear_artifact_excluded'].sum()):,}",
        f"- Model-eligible tracks: {len(training):,}",
        f"- Anchor Core 2 positive references: {int(label_counts.get(POSITIVE_LABEL, 0)):,}",
        f"- Operational negative references: {int(label_counts.get(NEGATIVE_LABEL, 0)):,}",
        f"- Excluded tracks retained in audit output: {len(exclusions):,}",
        "",
        "Linear artifact tracks are removed before residualizer fitting, PCA, model training, prediction, feature summaries, and visual-review sampling.",
        "", "## Corrected Residualized PCA Audit", "",
        pca_comparison.round(6).to_markdown(index=False),
        "",
        "Anchor Core 2 was rebuilt after uniform feature calculation and linear-artifact exclusion. The corrected count is data-derived and is not forced to the historical value of 2,963.",
        "", "## Corrected Anchor Core 2 Membership Audit", "",
        pd.DataFrame([membership_audit]).to_markdown(index=False),
        "", "## Models and Cross-validation", "",
        "Four methods were fitted independently: robust Mahalanobis reference scoring, regularized logistic regression, RBF-SVM, and histogram gradient boosting. Their probabilities were combined by equal-weight soft voting.",
        "", metrics.round(5).to_markdown(index=False),
        "", "## Consensus Output on H2B Reference Tracks", "",
    ]
    for label, count in consensus_counts.items():
        lines.append(f"- {label}: {count:,} tracks ({count / len(predictions) * 100:.2f}%)")
    lines.extend([
        "", "## Random Visual Checks", "",
        f"- Selected tracks with paired PNG and TIFF files: {len(visual_manifest):,}",
        "- At least eight per consensus group when available; smaller groups contribute all tracks.",
        "- Shortages are filled from the remaining groups to reach at least 24 when the total population permits.",
        "", visual_summary.to_markdown(index=False),
        "", "### Video-cell-track IDs", "",
    ])
    lines.extend(f"- `{identifier}`" for identifier in visual_manifest["track_identifier"])
    lines.extend([
        "", "## Interpretation Caution", "",
        "Anchor Core 2 was defined partly using related four-feature information. High classifier performance shows reproducibility of the operational reference, not independent biological truth. Complete-video/cell holdout, fold-specific anchor rebuilding, perturbation, residence-time evidence, visual review, and additional controls remain future validation options.",
        "", "All source SMol-FIESTA and historical analysis files were read only.",
    ])
    (output / "H2B_reference_build_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_h2b_reference(config: dict) -> Path:
    if is_replacement_backend(config):
        output = Path(config["paths"]["output_dir"])
        validation = validate_replacement_reference(output, config)
        print(json.dumps(validation, indent=2), flush=True)
        return output
    project_dir = Path(config["_project_dir"])
    paths = config["paths"]
    source_dir = resolve_path(paths["smol_fiesta_output_dir"], project_dir)
    historical_assignment_path = resolve_path(paths["historical_anchor_core2_assignments"], project_dir)
    prior_path = resolve_path(paths["prior_unified_feature_table"], project_dir)
    artifact_path = resolve_path(paths["audited_linear_artifact_track_list"], project_dir)
    output = Path(paths["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)

    registry, all_frames = load_behavior_coordinate_files(source_dir, config["behavior_files"])
    input_manifest = _input_manifest(registry)
    input_manifest.to_csv(output / "H2B_reference_input_files.csv", index=False)
    prior = pd.read_csv(prior_path)
    if "track_uid" not in prior.columns:
        prior = add_track_identity(prior)
    historical_assignments = pd.read_csv(historical_assignment_path)
    if "track_uid" not in historical_assignments.columns:
        historical_assignments = add_track_identity(historical_assignments)
    historical_positive_ids = set(
        historical_assignments.loc[
            _bool_series(historical_assignments["anchor_core2_member"]), "track_uid"
        ].astype(str)
    )
    audited_artifact_ids = _load_track_id_set(artifact_path)

    features = _compute_feature_table(all_frames, config)
    label_columns = [column for column in ["track_uid", "behavior_class", "behavior_plus_constrained_subgroup"] if column in prior.columns]
    features = features.merge(prior[label_columns].drop_duplicates("track_uid"), on="track_uid", how="left")
    if "behavior_class" in features.columns:
        features["source_behavior_class"] = features["behavior_class"].fillna(features["source_behavior_class"])
    features["linear_artifact_audited_flag"] = features["track_uid"].isin(audited_artifact_ids)
    features["linear_artifact_excluded"] = features["linear_artifact_rule_flag"] | features["linear_artifact_audited_flag"]
    preliminary_columns = ["log_Rg", "log_convex_hull_area", "log_D", "alpha_model"]
    preliminary_complete = np.isfinite(features[preliminary_columns].to_numpy(float)).all(axis=1)
    residualizer_population = features[~features["linear_artifact_excluded"] & preliminary_complete].copy()
    residualizers = fit_length_residualizers(residualizer_population)
    residualizers.to_csv(output / "H2B_reference_residualizer_coefficients.csv", index=False)
    features = add_residualized_features(features, residualizers)
    anchor_result = rebuild_anchor_core2(features, config["anchor_core2"])
    features = anchor_result.assignments
    anchor_result.thresholds.to_csv(output / "H2B_corrected_anchor_core2_thresholds.csv", index=False)
    anchor_result.composition.to_csv(output / "H2B_corrected_anchor_core2_composition.csv", index=False)
    features.to_csv(output / "H2B_corrected_anchor_core2_assignments.csv", index=False)

    corrected_positive_ids = set(
        features.loc[features["anchor_core2_member"], "track_uid"].astype(str)
    )
    shared_positive_ids = historical_positive_ids & corrected_positive_ids
    membership_audit = {
        "historical_anchor_core2_tracks": len(historical_positive_ids),
        "corrected_anchor_core2_tracks": len(corrected_positive_ids),
        "shared_tracks": len(shared_positive_ids),
        "historical_only_tracks": len(historical_positive_ids - corrected_positive_ids),
        "corrected_only_tracks": len(corrected_positive_ids - historical_positive_ids),
        "jaccard_similarity": (
            len(shared_positive_ids) / len(historical_positive_ids | corrected_positive_ids)
            if historical_positive_ids | corrected_positive_ids
            else np.nan
        ),
    }
    pd.DataFrame([membership_audit]).to_csv(
        output / "H2B_corrected_anchor_core2_vs_historical_membership_audit.csv", index=False
    )

    duplicated_registry_columns = [
        "Video #",
        "Cell",
        "Track",
        "Video Name",
        "track_identifier",
        "source_behavior_class",
    ]
    feature_merge = features.drop(columns=duplicated_registry_columns, errors="ignore")
    registry = registry.merge(feature_merge, on="track_uid", how="left", validate="one_to_one")
    registry["model_eligible"] = ~registry["linear_artifact_excluded"] & registry["model_feature_complete"]
    registry["reference_label"] = pd.NA
    registry.loc[registry["model_eligible"] & registry["anchor_core2_member"], "reference_label"] = POSITIVE_LABEL
    registry.loc[registry["model_eligible"] & ~registry["anchor_core2_member"], "reference_label"] = NEGATIVE_LABEL
    registry["exclusion_reason"] = ""
    registry.loc[registry["linear_artifact_excluded"], "exclusion_reason"] = "linear_artifact_quality_filter"
    registry.loc[~registry["linear_artifact_excluded"] & ~registry["model_feature_complete"], "exclusion_reason"] = "missing_or_nonfinite_model_feature"
    if len(registry) != 6158 or registry["track_uid"].duplicated().any():
        raise ValueError("H2B registry validation failed")
    if (registry["anchor_core2_member"] & ~registry["model_eligible"]).any():
        raise ValueError("Corrected Anchor Core 2 contains a non-eligible track")
    if int(registry["anchor_core2_member"].sum()) == 0:
        raise ValueError("Corrected Anchor Core 2 is empty")

    training = registry[registry["model_eligible"]].copy().reset_index(drop=True)
    exclusions = registry[~registry["model_eligible"]].copy().reset_index(drop=True)
    registry.to_csv(output / "H2B_full_track_registry_with_anchor_core2_labels.csv", index=False)
    training.to_csv(output / "H2B_model_training_table_linear_artifacts_removed.csv", index=False)
    exclusions.to_csv(output / "H2B_model_training_exclusions.csv", index=False)
    _pixel_size_audit(prior, features, output)
    pca_comparison = _pca_comparison(prior, training, output)

    X = training[MODEL_FEATURES].to_numpy(float)
    y = (training["reference_label"] == POSITIVE_LABEL).astype(int).to_numpy()
    result = train_models_with_oof(X, y, training["track_uid"], config["model"], model_dir)
    prediction_columns = ["track_uid", "track_identifier", "source_behavior_class", "behavior_plus_constrained_subgroup", "reference_label", "raw_track_length"] + PHYSICAL_FEATURES + MODEL_FEATURES
    predictions = training[prediction_columns].merge(result.out_of_fold_predictions, on="track_uid", how="left", validate="one_to_one")
    predictions.to_csv(output / "H2B_reference_out_of_fold_predictions.csv", index=False)
    result.metrics.to_csv(output / "H2B_reference_model_cross_validation_metrics.csv", index=False)
    proportions = [_proportion_table(training, "reference_label", "model_eligible_H2B_tracks")]
    for name in MODEL_NAMES:
        column = f"predicted_{name}"
        predictions[column] = np.where(predictions[f"probability_{name}"] >= 0.5, POSITIVE_LABEL, NEGATIVE_LABEL)
        proportions.append(_proportion_table(predictions, column, "out_of_fold_H2B_predictions"))
    proportions.append(_proportion_table(predictions, "consensus_prediction", "out_of_fold_H2B_predictions"))
    pd.concat(proportions, ignore_index=True).to_csv(output / "H2B_reference_label_and_prediction_proportions.csv", index=False)
    pd.concat([_feature_summary(training, "reference_label"), _feature_summary(predictions, "consensus_prediction")], ignore_index=True).to_csv(output / "H2B_reference_feature_dynamics_summary.csv", index=False)
    _, pca_manifest = save_pca_visualization(training, output)
    joblib.dump(pca_manifest, output / "H2B_reference_PCA_manifest.joblib")
    visual_manifest, visual_summary = _save_visual_checks(all_frames, predictions, output, "H2B", config)

    source_paths = [historical_assignment_path, prior_path, artifact_path] + [Path(path) for path in input_manifest["input_file_path"]]
    manifest = {
        "pipeline": "RRL_track_behaviour_ML",
        "version": "0.3.0",
        "created_at": datetime.now().isoformat(),
        "dataset_name": config["dataset_name"],
        "reference_meaning": "operational_similarity_to_H2B_Anchor_Core_2",
        "feature_columns": MODEL_FEATURES,
        "n_full_registry_tracks": len(registry),
        "n_model_eligible_tracks": len(training),
        "n_anchor_core2_positive": int((training["reference_label"] == POSITIVE_LABEL).sum()),
        "n_operational_negative": int((training["reference_label"] == NEGATIVE_LABEL).sum()),
        "n_linear_artifact_excluded": int(registry["linear_artifact_excluded"].sum()),
        "models": MODEL_NAMES + ["ensemble"],
        "consensus_settings": config["model"],
        "feature_calculation": config["feature_calculation"],
        "anchor_core2_definition": config["anchor_core2"],
        "anchor_core2_membership_audit": membership_audit,
        "linear_artifact_rule": config["linear_artifact_rule"],
        "prior_scale_warning": "Historical unified H2B features mixed 0.094 and 0.160 um/pixel calculations.",
        "software": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "source_files": [{"path": str(path), "sha256": _sha256(path)} for path in source_paths],
    }
    (output / "H2B_reference_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_reference_report(
        output,
        registry,
        training,
        exclusions,
        result.metrics,
        predictions,
        pca_comparison,
        membership_audit,
        visual_manifest,
        visual_summary,
        config,
    )
    return output


def apply_reference(
    config: dict,
    input_dir: str | Path,
    output_dir: str | Path | None,
    dataset_name: str,
    additional_linear_artifact_list: str | Path | None = None,
    skip_visual_videos: bool = False,
) -> Path:
    if is_replacement_backend(config):
        return apply_replacement_reference(
            config,
            input_dir,
            output_dir,
            dataset_name,
            additional_linear_artifact_list=additional_linear_artifact_list,
            skip_visual_videos=skip_visual_videos,
        )
    input_path = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else input_path / config["output"]["default_analysis_folder_name"]
    output.mkdir(parents=True, exist_ok=True)
    reference_output = Path(config["paths"]["output_dir"])
    residualizers = pd.read_csv(reference_output / "H2B_reference_residualizer_coefficients.csv")
    registry, frames = load_behavior_coordinate_files(input_path, config["behavior_files"])
    input_manifest = _input_manifest(registry)
    input_manifest.to_csv(output / f"{dataset_name}_RRL_input_files.csv", index=False)
    features = add_residualized_features(_compute_feature_table(frames, config), residualizers)
    reviewed_ids = _load_track_id_set(Path(additional_linear_artifact_list).expanduser().resolve() if additional_linear_artifact_list else None)
    features["linear_artifact_reviewed_flag"] = features["track_uid"].isin(reviewed_ids)
    features["linear_artifact_excluded"] = features["linear_artifact_rule_flag"] | features["linear_artifact_reviewed_flag"]
    features["model_eligible"] = ~features["linear_artifact_excluded"] & features["model_feature_complete"]
    features.to_csv(output / f"{dataset_name}_RRL_track_features.csv", index=False)
    eligible = features[features["model_eligible"]].copy().reset_index(drop=True)
    predictions = predict_models(load_models(reference_output / "models"), eligible[MODEL_FEATURES].to_numpy(float), eligible["track_uid"], config["model"])
    predictions = eligible.merge(predictions, on="track_uid", how="left", validate="one_to_one")
    predictions.to_csv(output / f"{dataset_name}_RRL_predictions.csv", index=False)
    proportions = _proportion_table(predictions, "consensus_prediction", "model_eligible_target_tracks")
    proportions.to_csv(output / f"{dataset_name}_RRL_consensus_proportions.csv", index=False)
    _feature_summary(predictions, "consensus_prediction").to_csv(output / f"{dataset_name}_RRL_feature_dynamics_summary.csv", index=False)
    save_application_pca_visualizations(predictions, output, dataset_name)
    save_feature_distribution_plot(predictions, output, dataset_name)
    visual_manifest, visual_summary = _save_visual_checks(frames, predictions, output, f"{dataset_name}_RRL", config)

    counts = dict(zip(proportions["category"], proportions["n_tracks"]))
    reviewed_only = features["linear_artifact_reviewed_flag"] & ~features["linear_artifact_rule_flag"]
    feature_only_exclusions = ~features["linear_artifact_excluded"] & ~features["model_feature_complete"]
    report = [f"# RRL Track Behaviour ML Application: {dataset_name}", "", "## Inputs", ""]
    for row in input_manifest.itertuples(index=False):
        report.append(f"- `{row.input_file_name}`: `{row.input_file_path}` ({row.n_tracks:,} tracks)")
    report.extend([
        "", "## Acquisition Settings", "",
        f"- Pixel size: {config['feature_calculation']['pixel_size_um']} um/pixel",
        f"- Canonical frame interval: {config['feature_calculation']['frame_interval_ms']} ms/frame",
        f"- Derived frame interval: {config['feature_calculation']['frame_interval_seconds']} s/frame",
        f"- Derived frame rate: {config['feature_calculation']['frame_rate_fps']} frames/s",
        "", "## Counts", "",
        f"- Coordinate tracks loaded: {len(features):,}",
        f"- Automatic linear-artifact flags: {int(features['linear_artifact_rule_flag'].sum()):,}",
        f"- Tracks present in the reviewed linear-artifact list: {int(features['linear_artifact_reviewed_flag'].sum()):,}",
        f"- Reviewed-only additions beyond the automatic rule: {int(reviewed_only.sum()):,}",
        f"- Total linear-artifact quality exclusions: {int(features['linear_artifact_excluded'].sum()):,}",
        f"- Tracks with incomplete features (may overlap artifact flags): {int((~features['model_feature_complete']).sum()):,}",
        f"- Additional feature-quality exclusions after artifact filtering: {int(feature_only_exclusions.sum()):,}",
        f"- Model-eligible tracks: {len(predictions):,}",
        f"- Predicted chromatin-bound-like: {int(counts.get(POSITIVE_LABEL, 0)):,}",
        f"- Predicted not-chromatin-bound-like: {int(counts.get(NEGATIVE_LABEL, 0)):,}",
        f"- Uncertain/out-of-reference: {int(counts.get('uncertain_or_out_of_reference', 0)):,}",
        f"- Visual checks with matching manifest: {len(visual_manifest):,}",
        f"- PNG trajectory plots: {len(visual_manifest):,}",
        f"- TIFF behavior videos: {len(visual_manifest):,}",
        "", "## Visual-check Group Allocation", "",
        visual_summary.to_markdown(index=False),
        "", "## Visual-check Track IDs", "",
    ])
    report.extend(f"- `{identifier}`" for identifier in visual_manifest["track_identifier"].tolist())
    report.extend([
        "", "## Interpretation", "",
        "Predictions indicate whole-track similarity to the corrected H2B Anchor Core 2 operational reference. Linear artifact tracks are retained only in the audit feature table and are excluded from every machine-learning calculation and review sample. Predictions do not establish chromatin binding or confirm condensate behavior.",
    ])
    (output / f"{dataset_name}_RRL_application_report.md").write_text("\n".join(report), encoding="utf-8")
    return output
