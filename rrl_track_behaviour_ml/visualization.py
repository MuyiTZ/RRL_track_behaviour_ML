from __future__ import annotations

from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURES
from .review_videos import prepare_review_track, render_track_video


COLORS = {
    "chromatin_bound_like": "#216E39",
    "not_chromatin_bound_like": "#8C8C8C",
    "uncertain_or_out_of_reference": "#E69F00",
}

CONSENSUS_ORDER = [
    "chromatin_bound_like",
    "uncertain_or_out_of_reference",
    "not_chromatin_bound_like",
]


def _save_consensus_pca(
    table: pd.DataFrame,
    feature_columns: list[str],
    output: Path,
    stem: str,
    title: str,
) -> pd.DataFrame:
    complete = table[np.isfinite(table[feature_columns].to_numpy(float)).all(axis=1)].copy()
    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(scaler.fit_transform(complete[feature_columns].to_numpy(float)))
    score_table = complete[["track_uid", "track_identifier", "consensus_prediction"]].copy()
    score_table["PC1"] = scores[:, 0]
    score_table["PC2"] = scores[:, 1]
    score_table["PC1_variance_fraction"] = pca.explained_variance_ratio_[0]
    score_table["PC2_variance_fraction"] = pca.explained_variance_ratio_[1]
    score_table.to_csv(output / f"{stem}_scores.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.4, 6.4), dpi=180)
    for label in CONSENSUS_ORDER:
        subset = score_table[score_table["consensus_prediction"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=15,
            alpha=0.38,
            color=COLORS[label],
            edgecolors="none",
            label=f"{label} (n={len(subset)})",
        )
    ax.axhline(0, color="#9AA3AF", linewidth=0.8)
    ax.axvline(0, color="#9AA3AF", linewidth=0.8)
    ax.grid(True, color="#E1E5EB", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return score_table


def save_application_pca_visualizations(table: pd.DataFrame, output_dir: str | Path, dataset_name: str) -> None:
    output = Path(output_dir)
    _save_consensus_pca(
        table,
        ["log_Rg", "log_convex_hull_area", "log_D", "alpha_model"],
        output,
        f"{dataset_name}_RRL_physical_PCA",
        f"{dataset_name}: physical-feature PCA",
    )
    _save_consensus_pca(
        table,
        MODEL_FEATURES,
        output,
        f"{dataset_name}_RRL_residualized_PCA",
        f"{dataset_name}: residualized-feature PCA",
    )


def save_feature_distribution_plot(table: pd.DataFrame, output_dir: str | Path, dataset_name: str) -> None:
    output = Path(output_dir)
    features = [
        ("Rg", "Radius of gyration (um)", True),
        ("convex_hull_area", "Convex hull area (um^2)", True),
        ("D", "MSD-derived D (um^2/s)", True),
        ("alpha", "Anomalous exponent alpha", False),
    ]
    labels = [label for label in CONSENSUS_ORDER if (table["consensus_prediction"] == label).any()]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=180)
    for ax, (feature, ylabel, use_log_scale) in zip(axes.flat, features):
        arrays = []
        kept_labels = []
        for label in labels:
            values = pd.to_numeric(
                table.loc[table["consensus_prediction"] == label, feature], errors="coerce"
            ).dropna()
            if use_log_scale:
                values = values[values > 0]
            if len(values):
                arrays.append(values.to_numpy(float))
                kept_labels.append(label)
        if arrays:
            parts = ax.violinplot(arrays, showmeans=False, showmedians=True, showextrema=False)
            for body, label in zip(parts["bodies"], kept_labels):
                body.set_facecolor(COLORS[label])
                body.set_edgecolor("none")
                body.set_alpha(0.55)
            for position, values, label in zip(range(1, len(arrays) + 1), arrays, kept_labels):
                ax.scatter(position, np.mean(values), color=COLORS[label], edgecolor="black", s=36, zorder=3)
            ax.set_xticks(range(1, len(kept_labels) + 1))
            ax.set_xticklabels(kept_labels, rotation=18, ha="right", fontsize=8)
        if use_log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#E1E5EB", linewidth=0.8)
    fig.suptitle(f"{dataset_name}: feature distributions by RRL consensus outcome", fontsize=15)
    fig.tight_layout()
    stem = output / f"{dataset_name}_RRL_feature_distributions_by_consensus"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_pca_visualization(table: pd.DataFrame, output_dir: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    output = Path(output_dir)
    X = table[MODEL_FEATURES].to_numpy(float)
    scaler = StandardScaler()
    z = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(z)
    score_table = table[["track_uid", "track_identifier", "reference_label"]].copy()
    score_table["PC1"] = scores[:, 0]
    score_table["PC2"] = scores[:, 1]
    score_table.to_csv(output / "H2B_reference_PCA_scores.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.4, 6.4), dpi=180)
    for label in ["not_chromatin_bound_like", "chromatin_bound_like"]:
        mask = score_table["reference_label"] == label
        ax.scatter(
            score_table.loc[mask, "PC1"],
            score_table.loc[mask, "PC2"],
            s=15,
            alpha=0.35 if label == "not_chromatin_bound_like" else 0.45,
            color=COLORS[label],
            edgecolors="none",
            label=f"{label} (n={int(mask.sum())})",
        )
    ax.axhline(0, color="#9AA3AF", linewidth=0.8)
    ax.axvline(0, color="#9AA3AF", linewidth=0.8)
    ax.grid(True, color="#E1E5EB", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("H2B Anchor Core 2 operational reference feature space")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "H2B_reference_PCA.png", bbox_inches="tight")
    fig.savefig(output / "H2B_reference_PCA.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return score_table, manifest


def select_visual_check_tracks(
    predictions: pd.DataFrame, sample_size: int, random_seed: int, minimum_per_group: int = 8,
) -> pd.DataFrame:
    if predictions["track_uid"].duplicated().any():
        raise ValueError("Visual-check sampling requires unique track_uid values")
    if not predictions["consensus_prediction"].isin(CONSENSUS_ORDER).all():
        raise ValueError("Visual-check sampling encountered an unknown consensus label")
    minimum = max(8, int(minimum_per_group))
    counts = predictions["consensus_prediction"].value_counts().reindex(CONSENSUS_ORDER, fill_value=0)
    allocation = {label: min(minimum, int(counts[label])) for label in CONSENSUS_ORDER}
    target = min(len(predictions), max(24, int(sample_size), minimum * len(CONSENSUS_ORDER)))
    # Reserve the per-group floor first. Fill shortages from groups that still
    # have tracks, keeping allocations as balanced as availability permits.
    while sum(allocation.values()) < target:
        available = [label for label in CONSENSUS_ORDER if allocation[label] < counts[label]]
        label = min(available, key=lambda group: allocation[group])
        allocation[label] += 1
    selected = []
    for index, label in enumerate(CONSENSUS_ORDER):
        subset = predictions[predictions["consensus_prediction"] == label].sort_values("track_uid")
        if allocation[label]:
            selected.append(subset.sample(n=allocation[label], random_state=random_seed + index))
    return pd.concat(selected, ignore_index=True) if selected else predictions.iloc[:0].copy()


def visual_check_group_summary(
    predictions: pd.DataFrame, selected: pd.DataFrame, minimum_per_group: int = 8,
) -> pd.DataFrame:
    rows = []
    for label in CONSENSUS_ORDER:
        available = int(predictions["consensus_prediction"].eq(label).sum())
        sampled = int(selected["consensus_prediction"].eq(label).sum())
        rows.append({
            "consensus_prediction": label,
            "available_tracks": available,
            "minimum_requested": max(8, int(minimum_per_group)),
            "selected_tracks": sampled,
            "png_count": sampled,
            "tiff_count": sampled,
            "availability_note": "all_available_tracks_selected" if available < max(8, int(minimum_per_group)) else "minimum_met",
        })
    return pd.DataFrame(rows)


def _review_paths(output: Path, identifier: str) -> tuple[Path, Path]:
    stem = identifier
    while True:
        png = output / f"{stem}_trajectory.png"
        video = output / f"{stem}_behaviour_review.tif"
        if not png.exists() and not video.exists():
            return png, video
        stem = f"{identifier}_{datetime.now():%Y%m%d_%H%M%S_%f}"


def draw_track_checks(
    frames: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: str | Path,
    visualization_settings: dict | None = None,
    *,
    acquisition_settings: dict,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = visualization_settings or {}
    tick_step = float(settings.get("visual_check_tick_step_pixels", 10.0))
    minimum_span = float(settings.get("visual_check_minimum_span_pixels", 40.0))
    if not np.isfinite([tick_step, minimum_span]).all() or min(tick_step, minimum_span) <= 0:
        raise ValueError("Visual-check spatial span and tick step must be positive and finite")
    manifest_columns = [
        "track_uid", "track_identifier", "consensus_prediction", "ensemble_probability",
        "visualization_file", "png_file", "video_file", "n_frame_rows", "shared_axis_span_pixels",
        "axis_tick_step_pixels", "video_n_frames", "video_playback_fps", "video_tail_frames",
        "video_context_frames", "gap_frame_count", "original_frame_start", "original_frame_end",
        "acquisition_frame_interval_ms", "pixel_size_um_per_pixel",
    ]
    if selected.empty:
        return pd.DataFrame(columns=manifest_columns)
    if selected["track_uid"].duplicated().any():
        raise ValueError("Visual-check selection contains duplicate track IDs")
    selected_frames = frames[frames["track_uid"].isin(selected["track_uid"])]
    missing = set(selected["track_uid"]) - set(selected_frames["track_uid"])
    if missing:
        raise ValueError(f"Missing coordinates for selected tracks: {sorted(missing)}")
    tracks = {uid: prepare_review_track(track) for uid, track in selected_frames.groupby("track_uid", sort=False)}
    spans = []
    for scale_track in tracks.values():
        scale_track = scale_track[scale_track["review_valid_coordinate"]]
        x_values = pd.to_numeric(scale_track["x"], errors="coerce")
        y_values = pd.to_numeric(scale_track["y"], errors="coerce")
        spans.append(max(float(x_values.max() - x_values.min()), float(y_values.max() - y_values.min())))
    maximum_span = max(spans, default=minimum_span)
    common_span = max(minimum_span, np.ceil((maximum_span + 2 * tick_step) / tick_step) * tick_step)
    rows = []
    selected_lookup = selected.set_index("track_uid")
    for uid, track in tracks.items():
        info = selected_lookup.loc[uid]
        valid = track["review_valid_coordinate"]
        plot_track = track[valid]
        identifier = str(info.get("track_identifier", uid))
        png_path, video_path = _review_paths(output, identifier)
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=160)
        if len(plot_track):
            frame_values = pd.to_numeric(plot_track["Frame"], errors="coerce")
            x_values = pd.to_numeric(plot_track["x"], errors="coerce")
            y_values = pd.to_numeric(plot_track["y"], errors="coerce")
            scatter = ax.scatter(
                x_values,
                y_values,
                c=frame_values,
                cmap="viridis",
                s=28,
                edgecolors="none",
            )
            coordinates = track[["x", "y"]].to_numpy(float)
            adjacent = valid.to_numpy()[:-1] & valid.to_numpy()[1:] & (np.diff(track["Frame"].to_numpy()) == 1)
            segments = np.stack([coordinates[:-1], coordinates[1:]], axis=1)[adjacent]
            ax.add_collection(LineCollection(segments, colors="#5B6472", linewidths=1.0, alpha=0.7))
            fig.colorbar(scatter, ax=ax, label="Frame")
            center_x = float((x_values.min() + x_values.max()) / 2.0)
            center_y = float((y_values.min() + y_values.max()) / 2.0)
            x_start = np.ceil((center_x - common_span / 2.0) / tick_step) * tick_step
            y_start = np.ceil((center_y - common_span / 2.0) / tick_step) * tick_step
            ax.set_xticks(np.arange(x_start, center_x + common_span / 2.0 + 1e-9, tick_step))
            ax.set_yticks(np.arange(y_start, center_y + common_span / 2.0 + 1e-9, tick_step))
            ax.set_xlim(center_x - common_span / 2.0, center_x + common_span / 2.0)
            ax.set_ylim(center_y + common_span / 2.0, center_y - common_span / 2.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#E5E7EB", linewidth=0.7)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_title(f"{identifier}\n{info['consensus_prediction']}")
        fig.tight_layout()
        with png_path.open("xb") as handle:
            fig.savefig(handle, format="png", bbox_inches="tight")
        plt.close(fig)
        video_info = render_track_video(
            track, identifier, str(info["consensus_prediction"]), video_path,
            common_span, settings, acquisition_settings,
        )
        rows.append(
            {
                "track_uid": uid,
                "track_identifier": identifier,
                "consensus_prediction": info["consensus_prediction"],
                "ensemble_probability": info["ensemble_probability"],
                "visualization_file": str(png_path),
                "png_file": str(png_path),
                "n_frame_rows": len(track),
                "shared_axis_span_pixels": common_span,
                "axis_tick_step_pixels": tick_step,
                **video_info,
            }
        )
    return pd.DataFrame(rows, columns=manifest_columns)
