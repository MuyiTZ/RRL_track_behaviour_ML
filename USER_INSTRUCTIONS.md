# Installation and use in an existing smol_env

## 1 Download

Open Anaconda PowerShell Prompt in the folder where you want to keep the program. Run each line separately:

```powershell
conda activate smol_env
git clone https://github.com/MuyiTZ/RRL_track_behaviour_ML.git
cd RRL_track_behaviour_ML
python -c "import sys; print(sys.executable)"
```

The Python path must belong to `smol_env`. Alternatively download the repository ZIP, extract it, then use `cd "C:/path/to/RRL_track_behaviour_ML-main"`. Keep all subfolders together. The program can sit beside SMol-FIESTA; it need not be copied into the SMol-FIESTA package.

## 2 Fill missing packages

```powershell
python check_installation.py
```

Install the missing packages printed by the checker. Typical additions:

```powershell
python -m pip install PyYAML tabulate onnxruntime
python -m pip check
python check_installation.py
```

All packages go into the active environment. Do not create another environment or downgrade scikit-learn to 1.4.2. Portable model exports avoid the historical serialization conflict. Review any dependency conflict before changing packages used by SMol-FIESTA. The complete list is in `requirements.txt`.

## 3 Verify calibration and models

The reference requires **0.094 um/pixel and 10 ms/frame** (100 frames/s). Compare `RRL_user_settings.yaml` with acquisition metadata. Input coordinates are pixels, not micrometres. Do not enter 0.010 in `frame_interval_ms`; the unit is milliseconds.

```powershell
python run_RRL_track_behaviour_ML.py build-reference
```

This checks release status, scale, feature contract, model checksums and loading of four models; it does not retrain. Another acquisition scale needs a separately calibrated reference.

## 4 Test the installation

```powershell
python run_RRL_track_behaviour_ML.py apply --input-dir examples/synthetic_smolfiesta --dataset-name DEMO --output-dir smoke_output
```

Expect prediction CSVs, PCA figures and trajectory PNG/TIFF pairs. The 42 tracks are artificial, not biological validation. On reruns choose a new output directory, such as `smoke_output_2`. Add `--skip-visual-videos` to skip trajectory PNG/TIFF generation.

## 5 Prepare input

Run compatible SMol-FIESTA through behaviour classification and breakdown. Use these coordinate files before condensate filtering:

```text
single-behaviour-track-bound.csv
single-behaviour-track-constraint-only.csv
single-behaviour-track-diffusive.csv
multi-behaviour-track-bound-diffusion.csv
multi-behaviour-track-bound-constraint.csv
multi-behaviour-track-constraint-diffusion.csv
multi-behaviour-track-all-three.csv
```

`single-behaviour-track-constraint.csv` is accepted as a fallback. Required columns: `Video #`, `Cell`, `Track`, `Frame`, `x`, `y`. Use integer frame indices and pixel coordinates. Preserve track identities and gaps; do not duplicate a track across behaviour classes.

## 6 Analyze your dataset

Replace the input folder and dataset name:

```powershell
python run_RRL_track_behaviour_ML.py apply --input-dir "D:/data/SMol-FIESTA-output" --dataset-name HAP1_SAMPLE_01
```

Use a simple dataset identifier without path separators. Default output: `RRL_track_behaviour_ML_analysis` inside the input folder. To use another new folder, append `--output-dir "D:/analysis/HAP1_SAMPLE_01"`. Existing output directories are refused.

For additional reviewed exclusions, append `--linear-artifact-list "D:/data/reviewed_exclusions.csv"`. The simplest CSV has a `track_uid` column containing IDs such as `1_1_42`. The automatic quality filter always runs.

## 7 Read results

- `*_replacement_track_features.csv`: full feature and quality audit.
- `*_replacement_predictions.csv`: eligible tracks, four probabilities, ensemble, votes and consensus.
- `*_replacement_consensus_proportions.csv`: eligible-track counts and proportions.
- `*_replacement_feature_dynamics_summary.csv`: group feature summaries.
- `*_replacement_feature_PCA.*`: PCA figures, scores and loadings.
- `visual_checks/`: trajectory PNGs and RGB TIFFs for Fiji.
- `visual_check_selection.csv`, `visual_check_manifest.csv`, `visual_check_group_summary.csv`: track selection, file pairs and group availability.
- `APPLICATION_MANIFEST.json`: paths, reference and run counts.

Gold, cyan and firebrick in trajectory reviews represent original bound, constrained and diffusive frame-level behaviour; white is the current spot. They do not represent the whole-track ML label. Playback defaults to 10 fps for inspection; acquisition remains 100 fps. Edit visual options in `RRL_user_settings.yaml`.

`chromatin_bound_like` indicates strong H2B-reference similarity; `not_chromatin_bound_like` indicates weak similarity; `uncertain_or_out_of_reference` indicates an intermediate result or model disagreement. These are operational outcomes, not biological ground truth.

## 8 Troubleshooting

| Symptom | Action |
| --- | --- |
| conda not recognized | Open Anaconda PowerShell Prompt. |
| Missing module | Verify the active Python path, run the checker and install the reported package. |
| ONNX Runtime DLL error | Verify a supported 64-bit Python/wheel; install Microsoft's supported Visual C++ runtime if the DLL error requests it. |
| Calibration mismatch | Check acquisition units; do not bypass the validation. |
| Missing model or checksum mismatch | Download and extract the entire distribution again. |
| Output directory exists | Choose a new output folder. |
| Missing files or columns | Compare names and headers with section 5. |
| No usable tracks | Check frame counts, quality flags and coordinate input. Prediction/PCA require usable tracks. |
| sklearn pickle error | Use the portable release containing models/PORTABLE_MODELS.json; do not directly load the historical joblib files. |

For daily use activate `smol_env`, enter the project directory and run the analysis. For a clean Git clone use `git pull --ff-only` to update after saving edited settings elsewhere. For ZIP installs extract the new release separately and transfer settings deliberately.
