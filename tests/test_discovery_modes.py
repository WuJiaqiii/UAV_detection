import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests.fixtures import parse_test_args
from util.discovery_context import annotate_discovery_candidates, attach_recording_predictions
from util.open_set import FEATURE_VERSION, normalize_features, save_npz
from util.unknown_density import guarded_hdbscan, independent_reachability
from util.unknown_discovery import discover_unknown


def two_signals(count=10):
    items, results = [], []
    for i in range(count):
        tracks = []
        for j, feature in enumerate(([0., 1., 0.], [0., 0., 1.])):
            track = {"track_id": f"T{j}", "valid_votes": 2-j, "observation_count": 2,
                     "recognition_status": "novel_candidate", "ood_score": 10., "ood_threshold": 1.,
                     "frequency_center_offset_mhz": float(j*20), "start_ms": 0., "end_ms": 100.}
            tracks.append(track)
            items.append({"path": f"file_{i}.dat", "recording_id": f"r{i}", "feature": feature, "track": track})
        results.append({"path": f"file_{i}.dat", "signals": tracks, "true_class_names": ["unseen"]})
    return items, results


class DiscoveryModeTests(unittest.TestCase):
    def test_single_fits_only_primary_and_inherits_secondary_without_fake_membership(self):
        items, results = two_signals()
        annotate_discovery_candidates(items, results)
        found = discover_unknown(items, input_mode="single")
        self.assertEqual((found["estimated_unknown_class_count"], found["fit_tracks"]), (1, 10))
        self.assertEqual(found["inherited_secondary_tracks"], 10)
        self.assertEqual(found["clustered_unknown_tracks"], 20)
        for a, b in zip(items[::2], items[1::2]):
            self.assertEqual(a["track"]["unknown_cluster_id"], b["track"]["unknown_cluster_id"])
            self.assertEqual(b["track"]["cluster_assignment_source"], "recording_primary")
            self.assertIsNone(b["track"]["cluster_membership"])
            self.assertEqual(b["track"]["frequency_center_offset_mhz"], 20.)
        attach_recording_predictions(results,{c["path"]:c["recording_id"] for c in items},"single")
        self.assertEqual(results[0]["single_signal_prediction"]["unknown_cluster_id"],
                         items[0]["track"]["unknown_cluster_id"])

    def test_multi_preserves_two_simultaneous_signals_in_every_recording(self):
        items, results = two_signals()
        found = discover_unknown(items, input_mode="multi")
        self.assertEqual((found["estimated_unknown_class_count"], found["fit_tracks"]), (2, 20))
        self.assertEqual(found["inherited_secondary_tracks"], 0)
        for a, b in zip(items[::2], items[1::2]):
            self.assertNotEqual(a["track"]["unknown_cluster_id"], b["track"]["unknown_cluster_id"])
            self.assertEqual((a["track"]["frequency_center_offset_mhz"], b["track"]["frequency_center_offset_mhz"]), (0., 20.))

    def test_known_primary_does_not_promote_a_secondary_unknown(self):
        items, results = two_signals()
        for row in results:
            row["signals"][0]["recognition_status"] = "known"
        secondary = items[1::2]
        annotate_discovery_candidates(secondary, results)
        found = discover_unknown(secondary, input_mode="single")
        self.assertEqual(found["fit_tracks"], 0)
        self.assertEqual(found["estimated_unknown_class_count"], 0)
        self.assertTrue(all(c["track"]["cluster_status"] == "secondary_without_clustered_primary" for c in secondary))

    def test_unusable_primary_is_not_replaced_and_truth_never_changes_selection(self):
        items, results = two_signals()
        for row in results:
            row["signals"][0]["recognition_status"] = "uncertain"
        first = copy.deepcopy(items[1::2])
        annotate_discovery_candidates(first, results)
        for row in results:
            row["true_class_names"] = ["totally_different_truth"]
        second = copy.deepcopy(items[1::2])
        annotate_discovery_candidates(second, results)
        self.assertEqual([c["track"] for c in first], [c["track"] for c in second])
        self.assertTrue(all(not c["track"]["discovery_is_primary"] for c in first))

    def test_primary_is_chosen_across_files_sharing_one_original_recording(self):
        items, results = two_signals(2)
        identities = {row["path"]: "one" for row in results}
        for item in items:
            item["recording_id"] = "one"
        annotate_discovery_candidates(items, results, identities)
        self.assertEqual(sum(c["track"]["discovery_is_primary"] for c in items), 1)
        self.assertEqual(discover_unknown(items, input_mode="single")["estimated_unknown_class_count"], 0)

    def test_primary_context_is_required_and_multiple_primary_flags_are_rejected(self):
        items, _ = two_signals()
        with self.assertRaisesRegex(ValueError, "primary context"):
            discover_unknown(items, input_mode="single")
        for item in items:
            item["track"]["discovery_is_primary"] = True
        with self.assertRaisesRegex(ValueError, "More than one primary"):
            discover_unknown(items, input_mode="single")

    def test_same_recording_cannot_supply_independent_cluster_support(self):
        items, _ = two_signals()
        for item in items:
            item["recording_id"] = "only_one"
        self.assertEqual(discover_unknown(items)["estimated_unknown_class_count"], 0)

    def test_duplicating_a_record_component_does_not_change_core_density(self):
        x = normalize_features([[1.,0,0], [1.,.1,0], [1.,.4,0], [1.,.8,0]])
        a, _ = independent_reachability(x, ["a","b","c","d"], 3)
        b, _ = independent_reachability(np.vstack([x,np.tile(x[:1],(20,1))]),
                                       ["a","b","c","d"]+["a"]*20, 3)
        np.testing.assert_array_equal(a, b[:4,:4])

    def test_real_single_unimodal_population_is_allowed(self):
        x = np.random.default_rng(17).normal(0,.01,(40,10)); x[:,0] += 1.
        items = [{"path": str(i), "recording_id": str(i), "feature": f,
                  "track": {"track_id": "T1"}} for i,f in enumerate(x)]
        found = discover_unknown(items)
        self.assertEqual(found["estimated_unknown_class_count"], 1)
        self.assertFalse(found["diagnostics"]["single_cluster_check"]["accepted_split"])

    def test_root_check_accepts_separated_children_and_rejects_artificial_splits(self):
        separated = normalize_features([[1.,0,0]]*8+[[0,1.,0]]*8)
        children = np.asarray([0]*8+[1]*8)
        for values, expected in ((separated, True), (np.tile([[1.,0,0]],(16,1)), False)):
            models = [Mock(probabilities_=np.ones(16)), Mock(probabilities_=np.ones(16))]
            models[0].fit_predict.return_value = np.zeros(16,dtype=int)
            models[1].fit_predict.return_value = children
            with patch("sklearn.cluster.HDBSCAN", side_effect=models):
                labels, _, info = guarded_hdbscan(values, list(map(str,range(16))),5,3,2)
            self.assertEqual(info["single_cluster_check"]["accepted_split"], expected)
            self.assertEqual(len(set(labels)), 2 if expected else 1)

    def test_small_supported_fraction_does_not_force_multiple_classes(self):
        x = normalize_features([[1.,0,0]]*5+[[0,1.,0]]*5+[[0,0,1.]]*30)
        models = [Mock(probabilities_=np.ones(40)), Mock(probabilities_=np.ones(40))]
        models[0].fit_predict.return_value = np.zeros(40,dtype=int)
        models[1].fit_predict.return_value = np.asarray([0]*5+[1]*5+[-1]*30)
        with patch("sklearn.cluster.HDBSCAN", side_effect=models):
            _,_,info = guarded_hdbscan(x,list(map(str,range(40))),5,3,2)
        self.assertFalse(info["single_cluster_check"]["accepted_split"])

    def test_legacy_policy_keeps_original_density(self):
        items, _ = two_signals()
        found = discover_unknown(items, cluster_policy="legacy")
        self.assertEqual(found["diagnostics"]["density_unit"], "track")

    def test_old_archive_uses_full_source_context_and_offline_scoring(self):
        from scripts.cluster_unknown_features import build_parser, run
        items, files = two_signals(6)
        records = [{"path":c["path"],"recording_id":c["recording_id"],"track_id":c["track"]["track_id"]}
                   for c in items]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);features=root/"unknown.npz";source=root/"results.json"
            save_npz(features,{"kind":"unknown_candidates","schema_version":1,"feature_version":FEATURE_VERSION,
                              "signature":{"sha256":"fixture"},"records":records},
                     features=normalize_features([c["feature"] for c in items]))
            source.write_text(json.dumps({"files":files,"effective_config":{"classes":{"known":0}},"metrics":{}}))
            args=parse_test_args(build_parser(),["--features",str(features),"--results_json",str(source),
                "--unknown_input_mode","single","--no-save_feature_plots","--output_dir",str(root/"out")])
            found=run(args)
            self.assertEqual(found["estimated_unknown_class_count"],1)
            self.assertEqual(found["rescored_metrics"]["unknown_accuracy"],1.)
            self.assertTrue((root/"out/open_set_metrics.json").is_file())
            saved=json.loads((root/"out/reclustered_results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["files"][0]["signals"]),2)
            self.assertEqual(saved["files"][0]["signals"][1]["frequency_center_offset_mhz"],20.)
            self.assertEqual(saved["files"][0]["single_signal_prediction"]["unknown_cluster_id"],"Unknown_001")
            self.assertEqual(saved["files"][0]["class_evidence"][0]["display_label"],"Unknown_001")
            self.assertEqual(len(saved["files"][0]["class_evidence"][0]["components"]),2)
            document=json.loads(source.read_text())
            document["files"][0]["signals"].append({"track_id":"missing","recognition_status":"novel_candidate"})
            source.write_text(json.dumps(document))
            args.output_dir=root/"mismatch"
            with self.assertRaisesRegex(ValueError,"novel track sets differ"):
                run(args)


if __name__ == "__main__":
    unittest.main()
