from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tifffile


STATE_COLORS = {2: (255, 215, 0), 1: (75, 207, 255), 0: (178, 34, 34)}
STATE_NAMES = {2: "bound", 1: "constrained", 0: "diffusive", -1: "gap"}


def prepare_review_track(track: pd.DataFrame) -> pd.DataFrame:
    result = track.copy()
    for column in ["Frame", "x", "y"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    frame_numbers = result["Frame"].to_numpy(float)
    if not np.isfinite(frame_numbers).all() or not np.equal(frame_numbers, np.floor(frame_numbers)).all():
        raise ValueError("Review videos require finite integer frame numbers")
    if result.empty or result["Frame"].duplicated().any():
        raise ValueError("Review track is empty or contains duplicate frame numbers")
    result = result.sort_values("Frame").reset_index(drop=True)
    result["review_valid_coordinate"] = (
        np.isfinite(result[["x", "y"]]).all(axis=1)
        & (result["x"] >= 0)
        & (result["y"] >= 0)
    )
    if not result["review_valid_coordinate"].any():
        raise ValueError("Review track contains no valid localizations")
    return result


def render_track_video(
    track: pd.DataFrame,
    identifier: str,
    consensus_prediction: str,
    path: Path,
    common_span: float,
    settings: dict,
    acquisition: dict,
) -> dict:
    track = prepare_review_track(track)
    pixel_size = float(acquisition["pixel_size_um"])
    interval_ms = float(acquisition["frame_interval_ms"])
    playback_fps = float(settings.get("video_playback_fps", 10.0))
    tail_frames = int(settings.get("video_tail_frames", 10))
    context_frames = int(settings.get("video_context_frames", 4))
    if not np.isfinite([pixel_size, interval_ms, playback_fps, common_span]).all():
        raise ValueError("Video calibration must be finite")
    if min(pixel_size, interval_ms, playback_fps, common_span) <= 0 or tail_frames < 1 or context_frames < 0:
        raise ValueError("Video calibration and tail length must be positive; context cannot be negative")

    good = track[track["review_valid_coordinate"]]
    cx = float((good.x.min() + good.x.max()) / 2)
    cy = float((good.y.min() + good.y.max()) / 2)
    size = 512
    scale = size / common_span
    first, last = int(track.Frame.min()), int(track.Frame.max())
    render_start, render_end = first - context_frames, last + context_frames
    n_frames = render_end - render_start + 1
    lookup = {int(row["Frame"]): row for row in track.to_dict("records")}
    gap_count = last - first + 1 - len(good)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    bar_pixels = min(float(settings.get("visual_check_tick_step_pixels", 10.0)), common_span / 4)

    def valid(row):
        return row is not None and bool(row["review_valid_coordinate"])

    def state_code(row):
        value = row.get("Bound", np.nan)
        return int(value) if pd.notna(value) and np.isfinite(float(value)) else None

    def position(row):
        return int(round(256 + (row["x"] - cx) * scale)), int(round(270 + (row["y"] - cy) * scale))

    def pages():
        for frame in range(render_start, render_end + 1):
            # Recreate the canvas, and draw a bounded tail only while a valid
            # current localization exists. Gaps and post-track pages stay empty.
            canvas = Image.new("RGB", (size, size), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            row = lookup.get(frame)
            if valid(row):
                state = STATE_NAMES.get(state_code(row), "unclassified")
            else:
                state = "not started" if frame < first else "ended" if frame > last else "gap / no valid localization"
            draw.text((10, 8), f"{identifier} | original frame {frame}", fill=(235, 235, 235), font=font)
            draw.text((10, 29), f"State: {state} | t = {(frame-first)*interval_ms:g} ms", fill=(195, 195, 195), font=small)
            draw.text((10, 48), f"{interval_ms:g} ms/frame acquisition | {playback_fps:g} fps playback | {tail_frames}-frame tail", fill=(145, 145, 145), font=small)
            draw.text((10, 66), f"ML: {consensus_prediction}", fill=(145, 145, 145), font=small)
            if valid(row):
                for previous in range(max(first, frame - tail_frames + 1), frame):
                    a, b = lookup.get(previous), lookup.get(previous + 1)
                    if valid(a) and valid(b):
                        draw.line([position(a), position(b)], fill=STATE_COLORS.get(state_code(b), (160, 160, 160)), width=3)
                x, y = position(row)
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 255, 255))
                width = draw.textlength(identifier, font=small)
                draw.text((min(max(x + 9, 8), size - width - 8), max(86, min(y - 7, size - 70))), identifier, fill=(245, 245, 245), font=small)
            draw.line([(24, 471), (24 + round(bar_pixels * scale), 471)], fill=(220, 220, 220), width=2)
            draw.text((24, 479), f"{bar_pixels:g} pixels = {bar_pixels*pixel_size:g} um | shared field: {common_span:g} pixels", fill=(185, 185, 185), font=small)
            yield np.asarray(canvas)

    info = {
        "track_identifier": identifier,
        "consensus_prediction": consensus_prediction,
        "original_frame_start": first,
        "original_frame_end": last,
        "rendered_frame_start": render_start,
        "rendered_frame_end": render_end,
        "acquisition_frame_interval_ms": interval_ms,
        "pixel_size_um_per_pixel": pixel_size,
        "display_span_pixels": common_span,
        "playback_fps": playback_fps,
        "tail_frames": tail_frames,
        "context_frames": context_frames,
        "gap_frame_count": gap_count,
        "state_colors": {"bound": "gold", "constrained": "cyan", "diffusive": "firebrick"},
    }
    with path.open("xb") as handle:
        tifffile.imwrite(
            handle, data=pages(), shape=(n_frames, size, size, 3), dtype=np.uint8,
            imagej=True, photometric="rgb", compression="zlib",
            metadata={"axes": "TYXS", "fps": playback_fps, "finterval": 1 / playback_fps, "unit": "um", "Info": json.dumps(info)},
            resolution=(scale / pixel_size, scale / pixel_size),
        )
    return {
        "video_file": str(path), "video_n_frames": n_frames,
        "video_playback_fps": playback_fps, "video_tail_frames": tail_frames,
        "video_context_frames": context_frames, "gap_frame_count": gap_count,
        "original_frame_start": first, "original_frame_end": last,
        "acquisition_frame_interval_ms": interval_ms, "pixel_size_um_per_pixel": pixel_size,
    }
