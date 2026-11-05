import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from util.cli import build_task_parser, main
from util.experiment_split import _recording_groups, split_dataset
from util.open_set_experiment import run_experiment, stage_arguments

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-experiment-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = patch("util.run_context.EXPERIMENT_ROOT", self.root/"runs")
        self.run_root.start()
        self.addCleanup(self.run_root.stop)
        self.weights = self.root / "detector.pt"
        self.weights.write_bytes(b"synthetic detector path fixture")

    def configuration(self, input_type="png", **experiment):
        doc = {"experiment": {"inputs": [str(self.root / "data")], "input_type": input_type,
                              "known_classes": ["A"], **experiment},
               "train": {"yolo_weights": str(self.weights), "epochs": 1, "num_workers": 0,
                         "mask_img_size": 32, "mask_pretrained": False},
               "profile": {"knn_k": 1, "min_calibration_per_class": 1},
               "discovery": {"unknown_cluster_backend": "dcg", "unknown_min_cluster_size": 2,
                             "unknown_min_recordings": 2, "save_visuals": False}}
        path = self.root / "experiment.yaml"
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        return path, doc

    def args(self, path):
        with contextlib.redirect_stdout(io.StringIO()):
            args = main(["experiment", "--config", str(path), "--dry_run"])
        args.output_dir = self.root / "experiment"
        return args

    def png_data(self):
        for name, count, base in (("A", 8, 10), ("B", 4, 200)):
            for i in range(count):
                self.png(name, i, base + i)

    def png(self, name, i, value, figure=1):
        path = self.root / "data" / f"{name}-[{i},1,0,2]-SNR-1-Figure-{figure}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", np.full((512, 750), value, dtype=np.uint8))[1].tofile(path)
        return path

    def test_split_is_seeded_disjoint_and_keeps_figure_exports_together(self):
        self.png_data()
        self.png("A", 0, 50, figure=2)
        path, _ = self.configuration()
        args = self.args(path)
        train, _ = stage_arguments(args, path)
        first, second = split_dataset(args, train), split_dataset(args, train)
        self.assertEqual(first, second)
        seen = {}
        for row in first["records"]:
            seen.setdefault(row["recording_id"], set()).add(row["split"])
            if row["class_name"] == "B":
                self.assertEqual(row["split"], "test_unknown")
        self.assertTrue(all(len(splits) == 1 for splits in seen.values()))
        self.assertEqual(first["classes"], {"A": 0})
        self.assertEqual(sum(v["recordings"] for v in first["summary"]["A"].values()), 8)

    def test_iq_parent_labels_and_duplicate_contents_are_grouped(self):
        for name, count in (("设备甲", 5), ("设备乙", 2)):
            for i in range(count):
                p = self.root / "data" / name / f"capture{i}.dat"
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("wb") as stream:
                    stream.write(f"{name}:{i}".encode())
                    stream.truncate(20_000_000)
        original = self.root / "data/设备甲/capture0.dat"
        alias = original.with_name("duplicate.dat")
        alias.write_bytes(original.read_bytes())
        path, _ = self.configuration("iq_dat", known_classes=["设备甲"])
        args = self.args(path)
        train, _ = stage_arguments(args, path)
        manifest = split_dataset(args, train)
        duplicates = [r for r in manifest["records"] if r["path"] in {str(alias), str(original)}]
        self.assertEqual(len({(r["recording_id"], r["split"]) for r in duplicates}), 1)
        self.assertEqual(manifest["unknown_classes"], ["设备乙"])
        self.assertEqual(manifest["classes"], {"设备甲": 0})

    def test_recording_aliases_are_transitive_and_conflicting_labels_fail(self):
        rows = [{"path": str(i), "class_name": "A", "source_sha256": digest, "recording_id": identity}
                for i, (digest, identity) in enumerate((("h1", "r1"), ("h1", "r2"), ("h2", "r2")))]
        self.assertEqual(len(_recording_groups(rows)), 1)
        rows[-1]["class_name"] = "B"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            _recording_groups(rows)

    def test_explicit_dataset_exclusions_never_become_unknown_test_classes(self):
        self.png_data()
        self.png("Noise", 99, 99)
        path, _ = self.configuration(exclude_classes=["Noise"])
        args = self.args(path)
        train, _ = stage_arguments(args, path)
        manifest = split_dataset(args, train)
        self.assertEqual(manifest["unknown_classes"], ["B"])
        self.assertTrue(all(r["split"] == "excluded" for r in manifest["records"] if r["class_name"] == "Noise"))

    def test_existing_model_head_must_match_the_manifest(self):
        self.png_data()
        path, _ = self.configuration()
        args = self.args(path)
        def wrong_head(train):
            destination = Path(train._run_dir) / "models/best.pth"
            destination.parent.mkdir(parents=True)
            torch.save({"config": {"classes": {"B": 0}}, "model_state_dict": {}}, destination)
        with self.assertRaisesRegex(ValueError, "class mapping"):
            run_experiment(args, path, train_runner=wrong_head)
        self.assertFalse((args.output_dir / "open_set").exists())

    def test_bad_partitions_ratios_and_small_calibration_fail_before_training(self):
        self.png_data()
        for fields in ({"known_classes": ["missing"]}, {"known_classes": ["A"], "unknown_classes": ["A", "B"]},
                       {"val_ratio": .5, "calibration_ratio": .3, "test_ratio": .3}):
            path, _ = self.configuration(**fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                run_experiment(self.args(path), path)
            self.assertFalse((self.root / "experiment").exists())
        path, doc = self.configuration()
        doc["profile"]["min_calibration_per_class"] = 20
        path.write_text(yaml.safe_dump(doc))
        with self.assertRaisesRegex(ValueError, "Insufficient independent"):
            run_experiment(self.args(path), path)
        self.assertFalse((self.root / "experiment").exists())

    def test_reserved_data_paths_and_iq_rate_mismatch_are_rejected(self):
        for section, key, value in (("train", "exclude_classes", ["B"]),
                                    ("profile", "calibration_inputs", ["leak"]),
                                    ("evaluation", "iq_sample_rate", 122880000)):
            path, doc = self.configuration("iq_dat")
            doc.setdefault(section, {})[key] = value
            path.write_text(yaml.safe_dump(doc))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.args(path)

    def test_frontend_manifest_capture_rate_is_resolved_before_all_stages(self):
        from dataclasses import asdict
        from util.spectrogram_frontend import FRONTEND_VERSION, FrontendConfig, spectrogram_contract
        frontend = self.root / "frontend.json"
        frontend.write_text(json.dumps({"version": FRONTEND_VERSION, "spectrogram": spectrogram_contract(),
            "frontend": asdict(FrontendConfig()), "source": {"input_type": "iq_dat", "iq_sample_rate": 122880000, "iq_byte_order": ">"}}))
        path, doc = self.configuration("iq_dat")
        doc["train"]["frontend_config"] = str(frontend)
        path.write_text(yaml.safe_dump(doc))
        train, profile = stage_arguments(self.args(path), path)
        self.assertEqual(train.iq_sample_rate, 122880000)
        self.assertEqual(profile.iq_sample_rate, 122880000)
        self.assertEqual(profile.iq_byte_order, ">")

    def test_invalid_prediction_options_fail_during_preflight(self):
        for key, value in (("min_votes", 0), ("iq_stride_ms", -1),
                           ("min_novel_observation_share", 2), ("track_max_gap_ms", -1)):
            path, doc = self.configuration()
            doc["evaluation"] = {key: value}
            path.write_text(yaml.safe_dump(doc))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.args(path)
        self.assertFalse((self.root / "experiment").exists())

    def test_split_only_needs_no_weights_and_writes_reusable_audit_files(self):
        self.png_data()
        path, _ = self.configuration(split_only=True)
        self.weights.unlink()
        report = run_experiment(self.args(path), path)
        self.assertEqual(report["status"], "split_only")
        output = self.root / "experiment"
        self.assertTrue((output / "split_manifest.json").is_file())
        self.assertFalse((output / "training").exists())
        train = json.loads((output / "effective_train.json").read_text())
        self.assertEqual(train["class_names"], ["A"])
        self.assertTrue(all(Path(p).name.startswith("A-") for p in train["train_dataset_path"]))

    def test_training_failure_is_recorded_and_existing_outputs_are_preserved(self):
        self.png_data()
        path, _ = self.configuration()
        args = self.args(path)
        def fail(train):
            raise RuntimeError("fixture training failure")
        with self.assertRaisesRegex(RuntimeError, "fixture training failure"):
            run_experiment(args, path, train_runner=fail)
        state = json.loads((args.output_dir / "experiment.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stage"], "training")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            run_experiment(args, path, train_runner=fail)

    def test_complete_experiment_exports_profile_metrics_and_unlabeled_inference_config(self):
        self.png_data()
        path, _ = self.configuration()
        args = self.args(path)
        def train_runner(train):
            self.assertEqual(train.class_names, ["A"])
            self.assertTrue(all(Path(p).name.startswith("A-") for p in train.train_dataset_path + train.val_dataset_path))
            destination = Path(train._run_dir) / "models/best.pth"
            destination.parent.mkdir(parents=True)
            torch.save({"model_state_dict": {}, "config": {"classes": {"A": 0}, "mask_img_size": 32}}, destination)
        class Predictor:
            def predict(self, spectrum, domain):
                embedding = [0., 1., 0.] if spectrum[0, 0] >= 200 else [1., 0., 0.]
                return {"status": "ok", "detected_boxes": 2, "groups": [{"group_id": 0, "status": "ok",
                    "class_name": "A", "class_id": 0, "confidence": 1., "probabilities": [1.],
                    "selected_boxes": [[10, 100, 50, 110], [100, 100, 140, 110]],
                    "selected_group_score": 1., "embedding": embedding}]}
        from scripts.evaluate_single_signal import run as actual_run
        with patch("util.train_entry.main", side_effect=train_runner), \
             patch("scripts.evaluate_single_signal.run", side_effect=lambda a, predictor=None: actual_run(a, Predictor())), \
             contextlib.redirect_stdout(io.StringIO()):
            report = main(["experiment", "--config", str(path)])
        args.output_dir = Path(report["run_dir"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["metrics"]["open_set_evaluation"]["all_accuracy"], 1.)
        self.assertEqual(report["metrics"]["estimated_unknown_class_count"], 1)
        self.assertTrue((args.output_dir / "test/open_set_metrics.json").is_file())
        self.assertTrue(Path(report["profile_path"]).is_file())
        config = args.output_dir / "inference.yaml"
        contents = yaml.safe_load(config.read_text())
        self.assertEqual(contents["infer"]["unknown_input_mode"], "multi")
        self.assertEqual(contents["evaluation"]["unknown_input_mode"], "single")
        self.assertNotIn("labels_json", contents["evaluation"])
        self.assertNotIn("class_names", contents["evaluation"])
        # Use the exported config on the same files without exposing their labels.
        from scripts.evaluate_single_signal import run as actual_run
        with patch("scripts.evaluate_single_signal.run", side_effect=lambda a: actual_run(a, Predictor())), \
             contextlib.redirect_stdout(io.StringIO()):
            prediction = main(["infer", "--config", str(config), "--inputs", str(self.root / "data"), "--no-save_visuals"])
        self.assertEqual(prediction["metrics"]["labeled_files"], 0)
        self.assertEqual(prediction["metrics"]["estimated_unknown_class_count"], 1)
        self.assertIsNone(prediction["metrics"]["open_set_evaluation"]["all_accuracy"])

    def test_infer_requires_profile_and_rejects_truth_even_in_dry_run(self):
        base = ["infer", "--inputs", "unlabeled", "--checkpoint_path", "model.pth", "--yolo_weights", "detector.pt", "--dry_run"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(base)
        for option in ("--label_from_parent", "--label_from_filename", "--export_features"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "unlabeled"):
                main([*base, "--open_set_profile", "profile.npz", option])

    def test_new_examples_validate_without_reading_datasets(self):
        for task, name in (("experiment", "experiment_iq.yaml"), ("experiment", "experiment_all_known_iq.yaml"),
                           ("infer", "infer_open_set.yaml")):
            with self.subTest(task=task), contextlib.redirect_stdout(io.StringIO()):
                main([task, "--config", str(ROOT / "config" / name), "--dry_run"])

    def test_all_known_automatically_discovers_every_class_and_has_no_test_partition(self):
        for name, base in (("A", 10), ("B", 100)):
            for i in range(40):
                self.png(name, i, base+i)
        # A same-recording export with different pixels must follow its source.
        self.png("A", 0, 70, figure=2)
        path, _ = self.configuration(protocol="all_known", known_classes=None, split_only=True)
        args=self.args(path)
        self.assertEqual(args.test_ratio, 0.)
        train,profile=stage_arguments(args,path)
        first=split_dataset(args,train)
        self.assertEqual(first,split_dataset(args,train))
        self.assertEqual((first["known_classes"],first["unknown_classes"]),(["A","B"],[]))
        for name in ("A","B"):
            self.assertEqual([first["summary"][name][s]["recordings"] for s in
                              ("train","val","calibration","test_known","test_unknown")], [26,6,8,0,0])
        grouped={}
        for row in first["records"]:
            grouped.setdefault(row["recording_id"],set()).add(row["split"])
        self.assertTrue(all(len(s)==1 for s in grouped.values()))
        self.assertFalse(hasattr(profile,"test_inputs"))
        result=run_experiment(args,path)
        self.assertFalse(result["test_enabled"])
        self.assertTrue((args.output_dir/"effective_profile.json").is_file())
        self.assertFalse((args.output_dir/"test_labels.json").exists())
        self.assertTrue(all(Path(r["path"]).is_file() for r in first["records"]))

    def test_all_known_exclusions_are_not_simulated_unknown_classes(self):
        self.png_data()
        path,_=self.configuration(protocol="all_known",known_classes=None,exclude_classes=["B"])
        args=self.args(path);train,_=stage_arguments(args,path)
        manifest=split_dataset(args,train)
        self.assertEqual(manifest["classes"],{"A":0})
        self.assertFalse(manifest["unknown_classes"])
        self.assertTrue(all(r["split"]=="excluded" for r in manifest["records"] if r["class_name"]=="B"))

    def test_all_known_selectors_and_invalid_ratios_fail_before_data_access(self):
        for fields in ({"known_classes":["A"]}, {"known_classes":None,"unknown_classes":["B"]},
                       {"known_classes":None,"val_ratio":0.}, {"known_classes":None,"test_ratio":-.1},
                       {"known_classes":None,"val_ratio":.8,"calibration_ratio":.3}):
            path,_=self.configuration(protocol="all_known",**fields)
            with self.subTest(fields=fields),patch("util.eval_common.collect_inputs") as collect:
                with self.assertRaises(ValueError):self.args(path)
                collect.assert_not_called()

    def test_all_known_optional_known_test_and_small_calibration_support(self):
        self.png_data()
        path,_=self.configuration(protocol="all_known",known_classes=None,test_ratio=.15)
        args=self.args(path);train,profile=stage_arguments(args,path)
        manifest=split_dataset(args,train)
        self.assertTrue(hasattr(profile,"test_inputs"))
        self.assertTrue(all(c["test_known"]["recordings"]==1 for c in manifest["summary"].values()))
        path,doc=self.configuration(protocol="all_known",known_classes=None)
        doc["profile"]["min_calibration_per_class"]=5
        path.write_text(yaml.safe_dump(doc))
        with self.assertRaisesRegex(ValueError,"Insufficient independent"):
            run_experiment(self.args(path),path)
        self.assertFalse((self.root/"experiment/training").exists())

    def test_all_known_iq_folder_labels_are_dynamic_and_duplicate_captures_stay_together(self):
        for name in ("设备甲","设备乙"):
            for i in range(4):
                p=self.root/"data"/name/f"{i}.dat";p.parent.mkdir(parents=True,exist_ok=True)
                with p.open("wb") as stream:
                    stream.write(f"{name}:{i}".encode());stream.truncate(20_000_000)
        original=self.root/"data/设备甲/0.dat";alias=original.with_name("alias.dat")
        alias.write_bytes(original.read_bytes())
        path,_=self.configuration("iq_dat",protocol="all_known",known_classes=None,split_only=True)
        args=self.args(path);train,_=stage_arguments(args,path)
        manifest=split_dataset(args,train)
        self.assertEqual(manifest["known_classes"],["设备乙","设备甲"])
        duplicate=[r for r in manifest["records"] if r["path"] in {str(original),str(alias)}]
        self.assertEqual(len({(r["recording_id"],r["split"]) for r in duplicate}),1)

    def test_all_known_complete_pipeline_exports_profile_and_external_unlabeled_inference(self):
        self.png_data()
        path,_=self.configuration(protocol="all_known",known_classes=None)
        observed={}
        def train_runner(train):
            observed["train"]=train
            classes={name:i for i,name in enumerate(train.class_names)}
            self.assertEqual(classes,{"A":0,"B":1})
            destination=Path(train._run_dir)/"models/best.pth";destination.parent.mkdir(parents=True)
            torch.save({"model_state_dict":{},"config":{"classes":classes,"mask_img_size":32}},destination)
        class Predictor:
            def predict(self,spectrum,domain):
                value=spectrum[0,0]
                feature=[0.,0.,1.] if value>=240 else [0.,1.,0.] if value>=180 else [1.,0.,0.]
                category=1 if 180<=value<240 else 0
                return {"status":"ok","detected_boxes":2,"groups":[{"group_id":0,"status":"ok",
                    "class_name":"B" if category else "A","class_id":category,"confidence":1.,
                    "probabilities":[0.,1.] if category else [1.,0.],
                    "selected_boxes":[[10,100,50,110],[100,100,140,110]],"selected_group_score":1.,"embedding":feature}]}
        from scripts.evaluate_single_signal import run as actual_run
        with patch("util.train_entry.main",side_effect=train_runner), \
             patch("scripts.evaluate_single_signal.run",side_effect=lambda a,predictor=None:actual_run(a,Predictor())), \
             contextlib.redirect_stdout(io.StringIO()):
            result=main(["experiment","--config",str(path)])
        output=Path(result["run_dir"])
        self.assertTrue(output.name.startswith("all_known_training_"))
        self.assertEqual((result["protocol"],result["status"],result["test_enabled"]),("all_known","complete",False))
        self.assertEqual(result["metrics"]["known_class_count"],2)
        self.assertFalse((output/"test").exists())
        self.assertFalse((output/"test_labels.json").exists())
        self.assertTrue(Path(result["profile_path"]).is_file())
        self.assertTrue((output/"open_set/reference/features.npz").is_file())
        self.assertTrue((output/"open_set/calibration/features.npz").is_file())
        config=yaml.safe_load((output/"inference.yaml").read_text())
        self.assertEqual(config["infer"]["unknown_input_mode"],"multi")
        self.assertNotIn("labels_json",config["evaluation"])
        self.assertNotIn("recording_ids_json",config["evaluation"])
        external=self.root/"external";external.mkdir()
        for i,value in enumerate([44,240,241,242,243,244]):
            cv2.imencode(".png",np.full((512,750),value,dtype=np.uint8))[1].tofile(external/f"query{i}.png")
        with patch("scripts.evaluate_single_signal.run",side_effect=lambda a:actual_run(a,Predictor())), \
             contextlib.redirect_stdout(io.StringIO()):
            inferred=main(["infer","--config",str(output/"inference.yaml"),"--inputs",str(external),"--no-save_visuals"])
        self.assertEqual(inferred["metrics"]["labeled_files"],0)
        self.assertEqual(inferred["metrics"]["error_files"],0)
        self.assertEqual(inferred["metrics"]["estimated_unknown_class_count"],1)
        self.assertIsNone(inferred["metrics"]["open_set_evaluation"]["all_accuracy"])

    def test_experiment_detector_override_is_shared_by_profile_and_deployment(self):
        path,doc=self.configuration(protocol="all_known",known_classes=None)
        doc["train"]["yolo_weights"]="/missing/template.pt";path.write_text(yaml.safe_dump(doc))
        with contextlib.redirect_stdout(io.StringIO()):
            args=main(["experiment","--config",str(path),"--yolo_weights",str(self.weights),"--dry_run"])
        train,profile=stage_arguments(args,path)
        self.assertEqual(Path(train.yolo_weights),self.weights)
        self.assertEqual(profile.yolo_weights,self.weights)
