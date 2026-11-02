# Distribution validation

Validated on Windows using the existing smol_env Python 3.11.13. Missing PyYAML, tabulate and ONNX Runtime were supplied from a temporary test directory, without installing them into the user's environment.

Observed runtime versions: NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.3, joblib 1.5.2, PyYAML 6.0.3, Pillow 11.2.1, tifffile 2025.2.18, tabulate 0.10.0 and ONNX Runtime 1.31.1.

## Checks

- Original fitted models versus portable inference: 5,275 finite-feature reference tracks; zero consensus-label changes. Numerical details: `MODEL_EXPORT_VALIDATION.json`.
- Frozen regression fixture: 256 synthetic feature vectors, expected probabilities from original fitted models, absolute probability tolerance 0.00005 and exact consensus comparison.
- Existing tests cover replacement-reference validation, visual sampling and TIFF rendering; additional tests cover portable-model parity and calibration rejection.
- All 17 unit tests passed.
- End-to-end example: 42 loaded tracks, 42 eligible predictions, 24 PNG/TIFF review pairs, PCA figures and complete application manifest.
- Relocation check: extracted the ZIP into another directory containing spaces, validated all models, and ran all 42 example tracks from outside the project directory using explicit paths.
- Reference, calibration and fitted parameters are unchanged. The export is inference-only, not retraining or recalibration.

## Packaging changes

- Resolve the frozen builder's project root relative to its location.
- Replace workstation-specific historical input paths with local training_inputs placeholders; those inputs are not needed for inference.
- Default to portable numerical/ONNX models and validate their hashes.
- Forward visual-review settings from the user's configuration to the release-bound application.
- Add dependency checking, complete runtime requirements, synthetic inputs and documentation.
- Retain original joblib models for provenance; no default inference path unpickles them.

## Limits

The synthetic example verifies execution, not biological accuracy. Export parity does not guarantee identical float32 SVM outputs for every possible input. Use the calibration specified by the frozen reference. Historical raw datasets, retraining inputs and old analysis results are excluded from the distribution.

Original sklearn serialization is version-dependent; see https://scikit-learn.org/stable/model_persistence.html.
