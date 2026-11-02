from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance

from .features import MODEL_FEATURES


@dataclass
class AnchorCore2Result:
    assignments: pd.DataFrame
    thresholds: pd.DataFrame
    composition: pd.DataFrame


def _group_labels(table: pd.DataFrame) -> pd.Series:
    if "behavior_plus_constrained_subgroup" in table.columns:
        combined = table["behavior_plus_constrained_subgroup"].astype("string")
    else:
        combined = pd.Series(pd.NA, index=table.index, dtype="string")
    return combined.fillna(table["source_behavior_class"].astype("string"))


def rebuild_anchor_core2(table: pd.DataFrame, settings: dict) -> AnchorCore2Result:
    """Rebuild the operational Anchor Core 2 in corrected residualized feature space."""
    result = table.copy()
    result["anchor_group_label"] = _group_labels(result)
    eligible = (~result["linear_artifact_excluded"]) & result["model_feature_complete"]

    single_bound_group = str(settings["single_bound_group"])
    confined_group = str(settings["confined_low_mobility_group"])
    candidate_groups = {str(value) for value in settings["candidate_groups"]}

    single_bound = result["anchor_group_label"].eq(single_bound_group)
    confined = result["anchor_group_label"].eq(confined_group)
    candidates = result["anchor_group_label"].isin(candidate_groups)

    valid_single_bound_log_d = pd.to_numeric(
        result.loc[eligible & single_bound, "log_D"], errors="coerce"
    ).dropna()
    if len(valid_single_bound_log_d) < 4:
        raise ValueError("Too few eligible single-bound tracks to calculate the upper Tukey fence")
    q1 = float(valid_single_bound_log_d.quantile(0.25))
    q3 = float(valid_single_bound_log_d.quantile(0.75))
    multiplier = float(settings.get("single_bound_high_D_tukey_multiplier", 1.5))
    upper_log_d_fence = q3 + multiplier * (q3 - q1)
    result["single_bound_high_D_outlier"] = (
        eligible & single_bound & (pd.to_numeric(result["log_D"], errors="coerce") > upper_log_d_fence)
    )

    result["base_anchor_member"] = eligible & (
        (single_bound & ~result["single_bound_high_D_outlier"]) | confined
    )
    base_anchor = result.loc[result["base_anchor_member"], MODEL_FEATURES]
    if len(base_anchor) <= len(MODEL_FEATURES):
        raise ValueError("Corrected base anchor has too few tracks for Mahalanobis estimation")

    covariance = EmpiricalCovariance(assume_centered=False).fit(base_anchor.to_numpy(float))
    complete = result["model_feature_complete"]
    result["anchor_mahalanobis_distance_squared"] = np.nan
    result.loc[complete, "anchor_mahalanobis_distance_squared"] = covariance.mahalanobis(
        result.loc[complete, MODEL_FEATURES].to_numpy(float)
    )
    result["anchor_mahalanobis_distance"] = np.sqrt(result["anchor_mahalanobis_distance_squared"])

    quantiles = [float(value) for value in settings.get("containment_quantiles", [0.5, 0.75, 0.9])]
    if quantiles != sorted(quantiles) or len(quantiles) != 3:
        raise ValueError("anchor_core2.containment_quantiles must contain three increasing values")
    anchor_distances = result.loc[result["base_anchor_member"], "anchor_mahalanobis_distance_squared"]
    cutoffs = {quantile: float(anchor_distances.quantile(quantile)) for quantile in quantiles}
    q50, q75, q90 = (cutoffs[value] for value in quantiles)

    d2 = result["anchor_mahalanobis_distance_squared"]
    result["anchor_containment_layer"] = "outside_anchor_90pct"
    result.loc[d2 <= q90, "anchor_containment_layer"] = "moderate_inside_90pct"
    result.loc[d2 <= q75, "anchor_containment_layer"] = "strict_inside_75pct"
    result.loc[d2 <= q50, "anchor_containment_layer"] = "hyper_strict_inside_50pct"
    result.loc[~complete, "anchor_containment_layer"] = "unavailable_missing_features"
    result.loc[result["linear_artifact_excluded"], "anchor_containment_layer"] = "excluded_linear_artifact"

    result["overlap_candidate_evaluated"] = eligible & candidates
    result["overlap_candidate_included"] = result["overlap_candidate_evaluated"] & (d2 <= q90)
    result["anchor_core2_member"] = result["base_anchor_member"] | result["overlap_candidate_included"]
    result["anchor_core2_source"] = "not_anchor_core2"
    result.loc[result["base_anchor_member"], "anchor_core2_source"] = "corrected_base_anchor"
    result.loc[result["overlap_candidate_included"], "anchor_core2_source"] = "corrected_mahalanobis_overlap"
    result.loc[result["linear_artifact_excluded"], "anchor_core2_source"] = "excluded_linear_artifact"

    threshold_rows = [
        {"metric": "single_bound_log_D_q1", "value": q1},
        {"metric": "single_bound_log_D_q3", "value": q3},
        {"metric": "single_bound_high_D_upper_log10_fence", "value": upper_log_d_fence},
        {"metric": "single_bound_high_D_upper_D_um2_s_fence", "value": 10**upper_log_d_fence},
    ]
    for quantile, d2_cutoff in cutoffs.items():
        label = int(round(quantile * 100))
        threshold_rows.extend(
            [
                {"metric": f"anchor_{label}pct_distance_squared", "value": d2_cutoff},
                {"metric": f"anchor_{label}pct_distance", "value": float(np.sqrt(d2_cutoff))},
            ]
        )
    thresholds = pd.DataFrame(threshold_rows)

    composition_rows = []
    for group, subset in result.groupby("anchor_group_label", dropna=False):
        composition_rows.append(
            {
                "anchor_group_label": group,
                "n_tracks_total": len(subset),
                "n_linear_artifact_excluded": int(subset["linear_artifact_excluded"].sum()),
                "n_model_eligible": int(eligible.loc[subset.index].sum()),
                "n_base_anchor": int(subset["base_anchor_member"].sum()),
                "n_overlap_candidates_evaluated": int(subset["overlap_candidate_evaluated"].sum()),
                "n_overlap_candidates_included": int(subset["overlap_candidate_included"].sum()),
                "n_anchor_core2": int(subset["anchor_core2_member"].sum()),
            }
        )
    return AnchorCore2Result(result, thresholds, pd.DataFrame(composition_rows))
