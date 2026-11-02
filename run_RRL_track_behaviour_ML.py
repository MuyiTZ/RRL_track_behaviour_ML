from __future__ import annotations

import argparse
from pathlib import Path

from rrl_track_behaviour_ml.config import load_config
from rrl_track_behaviour_ml.pipeline import apply_reference, build_h2b_reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate/build or apply the configured RRL H2B operational reference pipeline."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_H2B_reference.yaml")),
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--settings",
        default=str(Path(__file__).with_name("RRL_user_settings.yaml")),
        help="Simple user settings file containing pixel size and frame rate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "build-reference",
        help="Validate the active replacement bundle, or build a legacy reference when its backend is configured.",
    )
    apply_parser = subparsers.add_parser("apply", help="Apply trained H2B reference models to another SMol-FIESTA output folder.")
    apply_parser.add_argument("--input-dir", required=True, help="Folder containing the seven behavior-coordinate CSV files.")
    apply_parser.add_argument(
        "--output-dir",
        help="Target output folder. Default: <input-dir>/RRL_track_behaviour_ML_analysis.",
    )
    apply_parser.add_argument("--dataset-name", required=True, help="Short dataset identifier used in output filenames.")
    apply_parser.add_argument(
        "--linear-artifact-list",
        help="Optional reviewed CSV containing additional track_uid exclusions.",
    )
    apply_parser.add_argument(
        "--skip-visual-videos",
        action="store_true",
        help="Skip the trajectory PNG/TIFF visual-review sample (replacement backend).",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.settings)
    if args.command == "build-reference":
        output = build_h2b_reference(config)
    else:
        output = apply_reference(
            config,
            args.input_dir,
            args.output_dir,
            args.dataset_name,
            additional_linear_artifact_list=args.linear_artifact_list,
            skip_visual_videos=args.skip_visual_videos,
        )
    print(f"RRL_track_behaviour_ML completed. Outputs: {output}")


if __name__ == "__main__":
    main()
