import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.fixtures import archive, track_fixture
from util.class_evidence import summarize_class_evidence
from util.known_reference import RecordingClassBank, build_record_bank
from util.open_set import OpenSetProfile, apply_open_set, build_profile, normalize_features


def rows(labels, prefix):
    return [{"path": f"{prefix}/{i}.dat", "recording_id": f"{prefix}-{i}",
             "source_sha256": f"hash-{prefix}-{i}", "true_class_name": name,
             "track_id": "T0001"} for i, name in enumerate(labels)]


class RecordBalancedProfileTests(unittest.TestCase):
    def test_one_recording_cannot_supply_all_k_neighbors(self):
        ref = normalize_features([[1, 0], [1, .001], [1, .002], [0, 1]])
        bank = RecordingClassBank(ref, ["A"]*4, ["one"]*3+["two"], {"A": 0}, 2)
        value = bank.details(normalize_features([[1, 0]]))[0][0]
        self.assertAlmostEqual(value, 2**.5, places=6)

    def test_multimodal_class_keeps_both_control_and_video_prototypes(self):
        ref = normalize_features([[1, .01, 0], [.01, 1, 0], [1, -.01, 0], [-.01, 1, 0], [0, 0, 1], [.01, 0, 1]])
        records = rows(["Drone"]*4+["Other"]*2, "ref")
        records[1].update(recording_id=records[0]["recording_id"], source_sha256=records[0]["source_sha256"])
        records[3].update(recording_id=records[2]["recording_id"], source_sha256=records[2]["source_sha256"])
        cal = normalize_features([[1, .02, 0], [.02, 1, 0], [0, .01, 1]])
        cal_records = rows(["Drone", "Drone", "Other"], "cal")
        cal_records[1].update(recording_id=cal_records[0]["recording_id"], source_sha256=cal_records[0]["source_sha256"])
        metadata, arrays = build_record_bank(ref, records, cal, cal_records, {"Drone": 0, "Other": 1}, 1, 1, 100, .1)
        self.assertEqual(metadata["per_class"]["Drone"]["reference_recordings"], 2)
        self.assertEqual(metadata["per_class"]["Drone"]["reference_tracks"], 4)
        self.assertEqual(len(arrays["calibration_scores"]), 2)
        bank = RecordingClassBank(arrays["reference"], arrays["reference_labels"], arrays["reference_recording_ids"],
                                  {"Drone": 0, "Other": 1}, 1, arrays["class_scales"])
        scores, names, _, _ = bank.details(cal)
        self.assertEqual(names, ["Drone", "Drone", "Other"])
        self.assertTrue(np.all(scores <= metadata["threshold"]))

    def test_repeated_tracks_and_exports_do_not_change_scales_or_calibration(self):
        ref = normalize_features([[1, .01], [1, -.01], [1, .03]])
        rr = rows(["A"]*3, "ref")
        cal = normalize_features([[1, .02], [1, -.02], [1, .04]])
        cr = rows(["A"]*3, "cal")
        first, fa = build_record_bank(ref, rr, cal, cr, {"A": 0}, 1, 1, 100, .1)
        second, sa = build_record_bank(np.vstack([ref, np.repeat(ref[:1], 50, axis=0)]), rr+[copy.deepcopy(rr[0]) for _ in range(50)],
            np.vstack([cal, np.repeat(cal[:1], 50, axis=0)]), cr+[copy.deepcopy(cr[0]) for _ in range(50)], {"A": 0}, 1, 1, 100, .1)
        self.assertEqual(first["threshold"], second["threshold"])
        np.testing.assert_array_equal(fa["class_scales"], sa["class_scales"])
        np.testing.assert_array_equal(fa["calibration_scores"], sa["calibration_scores"])
        self.assertEqual(first["reference_recording_count"], 3)

    def test_calibration_max_counts_one_single_device_record_once(self):
        ref = normalize_features([[1, .01], [1, -.01], [1, .03]])
        cal = normalize_features([[1, .02], [1, .2], [1, -.02]])
        cr = rows(["A"]*3, "cal")
        cr[1].update(recording_id=cr[0]["recording_id"], source_sha256=cr[0]["source_sha256"])
        meta, arrays = build_record_bank(ref, rows(["A"]*3, "ref"), cal, cr, {"A": 0}, 1, 1, 100, .1)
        self.assertEqual(meta["calibration_recording_count"], 2)
        self.assertEqual(len(arrays["calibration_scores"]), 2)
        self.assertEqual(max(arrays["calibration_scores"]), max(arrays["calibration_track_scores"]))

    def test_calibration_does_not_fit_class_scales(self):
        ref = normalize_features([[1, .01], [1, -.01], [1, .03]])
        one = build_record_bank(ref, rows(["A"]*3, "ref"), normalize_features([[1, .02]]), rows(["A"], "cal"), {"A": 0}, 1, 1, 100, .1)
        two = build_record_bank(ref, rows(["A"]*3, "ref"), normalize_features([[0, 1]]), rows(["A"], "cal"), {"A": 0}, 1, 1, 100, .1)
        np.testing.assert_array_equal(one[1]["class_scales"], two[1]["class_scales"])
        self.assertNotEqual(one[0]["threshold"], two[0]["threshold"])

    def test_recording_label_conflicts_and_insufficient_independent_support_fail(self):
        ref = normalize_features([[1, 0], [1, .01]])
        records = rows(["A", "A"], "ref")
        records[1].update(recording_id=records[0]["recording_id"])
        with self.assertRaisesRegex(ValueError, "independent reference"):
            build_record_bank(ref, records, ref, rows(["A"]*2, "cal"), {"A": 0}, 2, 1, 100, .1)
        records[1]["true_class_name"] = "B"
        with self.assertRaisesRegex(ValueError, "conflicting class"):
            build_record_bank(ref, records, ref, rows(["A", "B"], "cal"), {"A": 0, "B": 1}, 1, 1, 100, .1)

    def test_two_observations_use_calibrated_track_and_guard_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            archive(root/"ref.npz", [[1, 0, 0], [1, .01, 0], [1, -.01, 0]], ["known"]*3, "ref")
            archive(root/"cal.npz", [[1, .02, 0]], ["known"], "cal")
            build_profile(root/"ref.npz", root/"cal.npz", root/"profile.npz", k=1, min_calibration_per_class=1)
            profile = OpenSetProfile(root/"profile.npz", {"sha256": "fixture"}, {"known": 0})
            results, groups = track_fixture([[1, 0, 0], [0, 1, 0]])
            self.assertEqual(len(apply_open_set(profile, results, groups)), 1)
            self.assertEqual(results[0]["signals"][0]["novel_observation_share"], .5)
            results, groups = track_fixture([[1, 0, 0], [0, 1, 0]])
            self.assertFalse(apply_open_set(profile, results, groups, .6, "observation_guard"))

    def test_legacy_score_profile_can_be_built_and_loaded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            archive(root/"ref.npz", [[1, 0], [1, .01]], ["known"]*2, "ref")
            archive(root/"cal.npz", [[1, .02]], ["known"], "cal")
            build_profile(root/"ref.npz", root/"cal.npz", root/"profile.npz", k=1,
                          min_calibration_per_class=1, score_method="global_knn")
            profile = OpenSetProfile(root/"profile.npz", {"sha256": "fixture"}, {"known": 0})
            self.assertEqual(profile.score_method, "global_knn")
            self.assertIsNone(profile.score_details([[1, 0]])[1][0])

    def test_component_summary_keeps_frequency_and_does_not_claim_device_identity(self):
        for open_set in (False, True):
            tracks = [{"track_id": f"T{i}", "status": "ok", "class_name": "Drone" if i < 2 else "Other",
                       "start_ms": 0., "end_ms": 100., "frequency_bands_offset_mhz": [[i*10., i*10.+1.]]}
                      for i in range(3)]
            if open_set:
                for t in tracks:
                    t["recognition_status"] = "known"
            result = {"path": "capture.dat", "signals": tracks, "true_class_names": ["never_used"]}
            rows_by_file = {"capture.dat": [{"groups": [{"track_id": "T0", "group_type": "hopping"},
                                                          {"track_id": "T1", "group_type": "frequency"}]}]}
            summarize_class_evidence([result], rows_by_file)
            self.assertEqual(len(result["signals"]), 3)
            self.assertEqual(len(result["class_evidence"]), 2)
            evidence = next(item for item in result["class_evidence"] if item["class_name"] == "Drone")
            self.assertEqual(evidence["identity_status"], "category_evidence_only")
            self.assertEqual(len(evidence["components"]), 2)
            self.assertEqual(evidence["components"][1]["frequency_bands_offset_mhz"], [[10., 11.]])


if __name__ == "__main__":
    unittest.main()
