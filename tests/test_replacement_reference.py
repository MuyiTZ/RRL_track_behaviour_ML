import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from rrl_track_behaviour_ml.pipeline import build_h2b_reference
from rrl_track_behaviour_ml.replacement_reference import (
    apply_replacement_reference,
    configured_backend,
    validate_replacement_reference,
)


FEATURES = [
    "log_rg2_noise_ratio_w10",
    "log_hull_noise_ratio_w10",
    "log_msd1_fixed90_ratio",
    "alpha_observed",
]


class ReplacementReferenceTests(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        release = root / "Trainingset-HAP1-H2B-FC-10ms-Leica-094um-260916"
        (release / "models").mkdir(parents=True)
        (release / "scripts").mkdir()
        (release / "simulation_validation").mkdir()
        (release / "RELEASE_STATUS.json").write_text(
            json.dumps(
                {
                    "release_name": release.name,
                    "status": "active_replacement",
                    "pixel_size_um": 0.094,
                    "frame_interval_ms": 10.0,
                    "model_features": FEATURES,
                }
            ),
            encoding="utf-8",
        )
        (release / "MODEL_MANIFEST.json").write_text(
            json.dumps(
                {
                    "model_features": FEATURES,
                    "pixel_size_um": 0.094,
                    "frame_interval_seconds": 0.01,
                }
            ),
            encoding="utf-8",
        )
        (release / "localization_error_calibration.csv").write_text("sigma_um\n0.0328785209\n", encoding="utf-8")
        (release / "simulation_validation" / "immobile_hull_null_w10.csv").write_text(
            "track_length,null\n5,1\n", encoding="utf-8"
        )
        (release / "PCA_ORIENTATION_POLICY.json").write_text("{}", encoding="utf-8")
        (release / "scripts" / "apply_training_set_260916.py").write_text("# test\n", encoding="utf-8")
        (release / "scripts" / "build_training_set_260916_complete.py").write_text("# test\n", encoding="utf-8")
        for name in (
            "RRL_mahalanobis_model.joblib",
            "RRL_logistic_model.joblib",
            "RRL_rbf_svm_model.joblib",
            "RRL_gradient_boosting_model.joblib",
        ):
            joblib.dump({"name": name}, release / "models" / name)
        return release

    def make_config(self, release: Path) -> dict:
        return {
            "reference": {
                "backend": "replacement_260916",
                "release_name": release.name,
            },
            "paths": {"output_dir": str(release)},
            "feature_calculation": {
                "pixel_size_um": 0.094,
                "frame_interval_ms": 10.0,
                "frame_interval_seconds": 0.01,
            },
            "output": {"default_analysis_folder_name": "RRL_track_behaviour_ML_analysis"},
        }

    def test_validation_passes_and_models_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            release = self.make_bundle(Path(temp_dir))
            result = validate_replacement_reference(release, self.make_config(release))
            self.assertEqual(result["release_status"], "active_replacement")
            self.assertEqual(result["model_features"], FEATURES)
            self.assertEqual(len(result["loaded_model_types"]), 4)

    def test_validation_rejects_inactive_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            release = self.make_bundle(Path(temp_dir))
            status_path = release / "RELEASE_STATUS.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["status"] = "retired"
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not active"):
                validate_replacement_reference(release, self.make_config(release))

    def test_build_reference_validates_without_retraining(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            release = self.make_bundle(Path(temp_dir))
            result = build_h2b_reference(self.make_config(release))
            self.assertEqual(result, release.resolve())

    @patch("rrl_track_behaviour_ml.replacement_reference.subprocess.run")
    def test_apply_uses_release_bound_preprocessor(self, run_mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release = self.make_bundle(root)
            input_dir = root / "input"
            input_dir.mkdir()
            artifact_list = root / "mislinks.csv"
            artifact_list.write_text("track_id\n", encoding="utf-8")
            output_dir = root / "application"
            result = apply_replacement_reference(
                self.make_config(release),
                input_dir,
                output_dir,
                "TEST",
                additional_linear_artifact_list=artifact_list,
                skip_visual_videos=True,
            )
            self.assertEqual(result, output_dir.resolve())
            command = run_mock.call_args.args[0]
            self.assertIn(str(release / "scripts" / "apply_training_set_260916.py"), command)
            self.assertIn("--additional-mislink-list", command)
            self.assertIn("--skip-visual-videos", command)
            run_mock.assert_called_once()

    def test_backend_defaults_to_legacy_only_when_unspecified(self):
        self.assertEqual(configured_backend({}), "legacy_residualizer")


if __name__ == "__main__":
    unittest.main()
