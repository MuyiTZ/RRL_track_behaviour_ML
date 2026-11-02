import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile

from rrl_track_behaviour_ml.visualization import (
    CONSENSUS_ORDER, draw_track_checks, select_visual_check_tracks, visual_check_group_summary,
)
from rrl_track_behaviour_ml.review_videos import prepare_review_track, render_track_video
from rrl_track_behaviour_ml.pipeline import _save_visual_checks


def population(counts):
    rows = [
        {"track_uid": f"{group}_{i}", "track_identifier": f"V1_C{group+1}_T{i}",
         "consensus_prediction": label, "ensemble_probability": 0.5}
        for group, (label, count) in enumerate(zip(CONSENSUS_ORDER, counts))
        for i in range(count)
    ]
    return pd.DataFrame(rows, columns=["track_uid", "track_identifier", "consensus_prediction", "ensemble_probability"])


def coordinates(uid="track", offset=0):
    # Explicit invalid point at 12 and absent frame 14 must both be blank.
    return pd.DataFrame({
        "track_uid": [uid] * 6,
        "Frame": [10, 11, 12, 13, 15, 16],
        "x": [100+offset, 101+offset, -1, 102+offset, 103+offset, 104+offset],
        "y": [100, 101, -1, 100, 102, 101],
        "Bound": [2, 2, 2, 1, 0, 0],
    })


ACQUISITION = {"pixel_size_um": 0.2, "frame_interval_ms": 20.0}


class SamplingTests(unittest.TestCase):
    def test_minimum_eight_each_even_with_old_twenty_setting(self):
        table = population([100, 13, 50])
        selected = select_visual_check_tracks(table, 20, 7)
        self.assertEqual(selected.consensus_prediction.value_counts().to_dict(), dict.fromkeys(CONSENSUS_ORDER, 8))
        self.assertFalse(selected.track_uid.duplicated().any())
        repeat = select_visual_check_tracks(table.sample(frac=1, random_state=8), 20, 7)
        self.assertEqual(selected.track_uid.tolist(), repeat.track_uid.tolist())

    def test_small_group_all_selected_and_total_topped_up(self):
        table = population([100, 3, 100])
        selected = select_visual_check_tracks(table, 24, 7)
        counts = selected.consensus_prediction.value_counts()
        self.assertEqual(len(selected), 24)
        self.assertEqual(counts[CONSENSUS_ORDER[1]], 3)
        self.assertGreaterEqual(counts[CONSENSUS_ORDER[0]], 8)
        self.assertGreaterEqual(counts[CONSENSUS_ORDER[2]], 8)
        summary = visual_check_group_summary(table, selected)
        self.assertEqual(summary.iloc[1].availability_note, "all_available_tracks_selected")

    def test_small_population_and_empty(self):
        for counts in ([2, 0, 3], [0, 0, 0], [2, 12, 0]):
            table = population(counts)
            selected = select_visual_check_tracks(table, 24, 7)
            self.assertEqual(set(selected.track_uid), set(table.track_uid))

    def test_larger_requested_minimum_and_missing_group(self):
        self.assertEqual(len(select_visual_check_tracks(population([40, 40, 40]), 24, 7, 10)), 30)
        selected = select_visual_check_tracks(population([50, 0, 50]), 24, 7)
        self.assertEqual(selected.consensus_prediction.value_counts().tolist(), [12, 12])

    def test_duplicate_and_unknown_label_rejected(self):
        table = population([2, 2, 2])
        with self.assertRaises(ValueError):
            select_visual_check_tracks(pd.concat([table, table]), 24, 7)
        table.loc[0, "consensus_prediction"] = "unexpected"
        with self.assertRaises(ValueError):
            select_visual_check_tracks(table, 24, 7)


class VideoTests(unittest.TestCase):
    def test_streamed_rgb_video_clears_gaps_and_end_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.tif"
            result = render_track_video(coordinates(), "V1_C1_T1", CONSENSUS_ORDER[0], path, 40, {}, ACQUISITION)
            self.assertEqual(result["gap_frame_count"], 2)
            with tifffile.TiffFile(path) as tif:
                self.assertTrue(tif.is_imagej)
                self.assertEqual(tif.series[0].shape, (15, 512, 512, 3))
                self.assertEqual(tif.series[0].dtype, np.dtype("uint8"))
                info = json.loads(tif.imagej_metadata["Info"])
                self.assertEqual(info["acquisition_frame_interval_ms"], 20)
                self.assertEqual(info["pixel_size_um_per_pixel"], 0.2)
                self.assertEqual(tif.imagej_metadata["fps"], 10)
                start = info["rendered_frame_start"]
                for original_frame in [6, 9, 12, 14, 17, 20]:
                    self.assertFalse(tif.pages[original_frame-start].asarray()[90:430].any())
                self.assertTrue(tif.pages[11-start].asarray()[90:430].any())
                self.assertFalse(np.array_equal(tif.pages[10-start].asarray(), tif.pages[11-start].asarray()))
                frame13 = tif.pages[13-start].asarray()
                self.assertFalse(np.any(np.all(frame13[90:430] == (75, 207, 255), axis=2)))

    def test_bad_frames_and_missing_state(self):
        track = coordinates()
        track["Frame"] = track["Frame"].astype(float)
        track.loc[0, "Frame"] = 10.5
        with self.assertRaises(ValueError):
            prepare_review_track(track)
        track.loc[0, "Frame"] = 11
        with self.assertRaises(ValueError):
            prepare_review_track(track)
        with tempfile.TemporaryDirectory() as temp:
            result = render_track_video(coordinates().drop(columns="Bound"), "V1_C1_T1", CONSENSUS_ORDER[0], Path(temp)/"x.tif", 40, {}, ACQUISITION)
            self.assertEqual(result["video_n_frames"], 15)

    def test_png_tiff_pair_manifest_and_no_overwrite(self):
        selected = population([1, 1, 1])
        frames = pd.concat([coordinates(uid, 10*i) for i,uid in enumerate(selected.track_uid)])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            previous = folder / "one_time_review.tif"
            previous.write_bytes(b"preserve me")
            manifest = draw_track_checks(frames, selected, folder, acquisition_settings=ACQUISITION)
            self.assertEqual(len(manifest), 3)
            self.assertEqual(manifest.shared_axis_span_pixels.unique().tolist(), [40])
            self.assertEqual(manifest.axis_tick_step_pixels.unique().tolist(), [10])
            hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
            for column in ["png_file", "video_file", "visualization_file"]:
                self.assertTrue(all(Path(value).is_file() for value in manifest[column]))
            second = draw_track_checks(frames, selected, folder, acquisition_settings=ACQUISITION)
            self.assertTrue(set(second.png_file).isdisjoint(set(manifest.png_file)))
            self.assertTrue(set(second.video_file).isdisjoint(set(manifest.video_file)))
            for path, digest in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_shared_span_expands_and_empty_manifest(self):
        selected = population([1, 1, 0])
        frames = pd.concat([coordinates(uid) for uid in selected.track_uid], ignore_index=True)
        frames.loc[0, "x"] = 40
        with tempfile.TemporaryDirectory() as temp:
            manifest = draw_track_checks(frames, selected, temp, acquisition_settings=ACQUISITION)
            self.assertEqual(manifest.shared_axis_span_pixels.nunique(), 1)
            self.assertGreater(manifest.shared_axis_span_pixels.iloc[0], 64)
            empty = draw_track_checks(frames, selected.iloc[:0], temp, acquisition_settings=ACQUISITION)
            self.assertTrue(empty.empty)
            self.assertIn("video_file", empty.columns)

    def test_pipeline_output_integration(self):
        selected = population([1, 1, 1])
        frames = pd.concat([coordinates(uid) for uid in selected.track_uid])
        config = {"model": {"random_seed": 8, "visual_check_sample_size": 24},
                  "visualization": {}, "feature_calculation": ACQUISITION}
        with tempfile.TemporaryDirectory() as temp:
            manifest, summary = _save_visual_checks(frames, selected, Path(temp), "test", config)
            self.assertEqual(len(manifest), 3)
            self.assertEqual(summary.png_count.sum(), 3)
            for suffix in ["selection", "manifest", "group_summary"]:
                self.assertTrue((Path(temp)/f"test_random_visual_check_{suffix}.csv").exists())


if __name__ == "__main__":
    unittest.main()
