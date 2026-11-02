from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib


BACKEND_NAME = "replacement_260916"
EXPECTED_MODEL_FEATURES = [
    "log_rg2_noise_ratio_w10",
    "log_hull_noise_ratio_w10",
    "log_msd1_fixed90_ratio",
    "alpha_observed",
]
REQUIRED_MODELS = [
    "RRL_mahalanobis_model.joblib",
    "RRL_logistic_model.joblib",
    "RRL_rbf_svm_model.joblib",
    "RRL_gradient_boosting_model.joblib",
]


def configured_backend(config: dict[str, Any]) -> str:
    return str(config.get("reference", {}).get("backend", "legacy_residualizer")).strip()


def is_replacement_backend(config: dict[str, Any]) -> bool:
    return configured_backend(config) == BACKEND_NAME


def validate_replacement_reference(reference_output: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(reference_output).expanduser().resolve()
    portable = (root / 'models' / 'PORTABLE_MODELS.json').is_file()
    model_files = ['PORTABLE_MODELS.json', 'mahalanobis.json', 'logistic.json', 'rbf_svm.onnx', 'gradient_boosting.json'] if portable else REQUIRED_MODELS
    required = [
        root / "RELEASE_STATUS.json",
        root / "MODEL_MANIFEST.json",
        root / "localization_error_calibration.csv",
        root / "simulation_validation" / "immobile_hull_null_w10.csv",
        root / "scripts" / "apply_training_set_260916.py",
        root / "scripts" / "build_training_set_260916_complete.py",
        root / "PCA_ORIENTATION_POLICY.json",
        *[root / "models" / name for name in model_files],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Replacement reference is incomplete. Missing: " + "; ".join(missing))

    status = json.loads((root / "RELEASE_STATUS.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    if status.get("status") != "active_replacement":
        raise ValueError(f"Replacement reference is not active: {status.get('status')!r}")
    expected_name = config.get("reference", {}).get("release_name")
    if expected_name and status.get("release_name") != expected_name:
        raise ValueError(
            f"Configured release {expected_name!r} does not match bundle {status.get('release_name')!r}"
        )
    if manifest.get("model_features") != EXPECTED_MODEL_FEATURES:
        raise ValueError(
            f"Unexpected replacement model features: {manifest.get('model_features')!r}"
        )
    feature_config = config["feature_calculation"]
    if abs(float(manifest["pixel_size_um"]) - float(feature_config["pixel_size_um"])) > 1e-12:
        raise ValueError("Configured pixel size does not match the replacement reference")
    if abs(float(manifest["frame_interval_seconds"]) - float(feature_config["frame_interval_seconds"])) > 1e-12:
        raise ValueError("Configured frame interval does not match the replacement reference")

    if portable:
        import hashlib
        metadata = json.loads((root / 'models' / 'PORTABLE_MODELS.json').read_text(encoding='utf-8'))
        for name, expected in metadata['sha256'].items():
            if Path(name).name != name:
                raise ValueError('Invalid portable model filename')
            if hashlib.sha256((root / 'models' / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Portable model checksum mismatch: {name}')
        from .portable_models import load_portable_models
        loaded_model_types = {name: type(model).__name__ for name, model in load_portable_models(root / 'models').items()}
    else:
        loaded_model_types = {}
        for name in REQUIRED_MODELS:
            path = root / "models" / name
            loaded_model_types[name] = type(joblib.load(path)).__name__
    return {
        "backend": BACKEND_NAME,
        "release_name": status["release_name"],
        "release_status": status["status"],
        "reference_output": str(root),
        "model_features": manifest["model_features"],
        "loaded_model_types": loaded_model_types,
        "pca_orientation": "chromatin_bound_like_centroid_left",
    }


def apply_replacement_reference(
    config: dict[str, Any],
    input_dir: str | Path,
    output_dir: str | Path | None,
    dataset_name: str,
    additional_linear_artifact_list: str | Path | None = None,
    skip_visual_videos: bool = False,
) -> Path:
    reference_output = Path(config["paths"]["output_dir"]).resolve()
    validate_replacement_reference(reference_output, config)
    input_path = Path(input_dir).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else input_path / config["output"]["default_analysis_folder_name"]
    )
    script = reference_output / "scripts" / "apply_training_set_260916.py"
    command = [
        sys.executable,
        str(script),
        "--input-dir",
        str(input_path),
        "--dataset-name",
        dataset_name,
        "--output-dir",
        str(output),
        "--runtime-options",
        json.dumps({'model': config.get('model', {}), 'visualization': config.get('visualization', {})}),
    ]
    if additional_linear_artifact_list:
        command.extend(
            ["--additional-mislink-list", str(Path(additional_linear_artifact_list).expanduser().resolve())]
        )
    if skip_visual_videos:
        command.append("--skip-visual-videos")
    subprocess.run(command, check=True)
    return output
