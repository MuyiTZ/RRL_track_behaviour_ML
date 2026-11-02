# RRL_track_behaviour_ML

Apply the frozen HAP1-H2B Anchor Core 2 reference to the seven behaviour-coordinate CSVs produced by compatible SMol-FIESTA. This inference distribution includes all four fitted models, calibration data, preprocessing code, a synthetic example, tests and an installation guide.

## Install into your existing smol_env

Use Anaconda PowerShell Prompt. No new virtual environment is required.

```powershell
conda activate smol_env
git clone https://github.com/MuyiTZ/RRL_track_behaviour_ML.git
cd RRL_track_behaviour_ML
python check_installation.py
```

Install only packages reported missing. Typical additions:

```powershell
python -m pip install PyYAML tabulate onnxruntime
python -m pip check
python check_installation.py
```

`requirements.txt` lists all runtime dependencies. `python -m pip install -r requirements.txt` can fill remaining requirements; inspect proposed changes if your environment has other version constraints. Do not downgrade NumPy or scikit-learn to the historical training versions. This project runs from its folder; `pip install -e .` is not needed.

Without Git, select **Code > Download ZIP** on the repository page, extract the whole folder and open a terminal inside it.

## First run and real analysis

```powershell
python run_RRL_track_behaviour_ML.py build-reference
python run_RRL_track_behaviour_ML.py apply --input-dir examples/synthetic_smolfiesta --dataset-name DEMO --output-dir smoke_output
python run_RRL_track_behaviour_ML.py apply --input-dir "D:/data/SMol-FIESTA-output" --dataset-name MY_DATA
```

`build-reference` validates the bundled release; it does not retrain. The example has 42 artificial tracks and is only an installation test. Each run needs a new output directory. Real data defaults to `<input-dir>/RRL_track_behaviour_ML_analysis`.

The reference is calibrated **only for 0.094 um/pixel and 10 ms/frame**. Check `RRL_user_settings.yaml`. A different acquisition scale requires a separately calibrated reference; changing the settings alone is insufficient.

See [USER_INSTRUCTIONS.md](USER_INSTRUCTIONS.md) and [Installation_Steps.docx](Installation_Steps.docx).

## Portable models

Original models were fitted with scikit-learn 1.4.2, whose serialization is incompatible with newer SMol-FIESTA environments. This distribution uses numerical JSON exports for Mahalanobis, logistic regression and gradient boosting, and ONNX for RBF SVM. No model was retrained. Original joblib files are retained for provenance and are not loaded by default.

All 5,275 finite-feature reference tracks retained their consensus labels after export. Three model probabilities were identical; the maximum SVM probability difference was approximately 0.0000136. See [MODEL_EXPORT_VALIDATION.json](MODEL_EXPORT_VALIDATION.json). This is numerical regression testing, not biological validation. Near-boundary cases outside these data may differ due to float32 inference.

Features: `log_rg2_noise_ratio_w10`, `log_hull_noise_ratio_w10`, `log_msd1_fixed90_ratio`, `alpha_observed`. Outcomes: `chromatin_bound_like`, `not_chromatin_bound_like`, or `uncertain_or_out_of_reference`. These describe operational H2B similarity, not proof of chromatin binding or condensate identity.

Historical raw data and historical analysis outputs are not included. The builder script supplies frozen feature calculations to inference. Historical retraining requires the original inputs and is outside this installation workflow.

## Verification

```powershell
python -m unittest discover -s tests -v
```

See [VALIDATION.md](VALIDATION.md) for tested versions and checks. `SHA256SUMS.json` records distribution hashes. Only load trusted model artifacts. Background: [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).
