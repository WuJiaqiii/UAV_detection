import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.fixtures import archive
from util.mahalanobis import RecordingMahalanobis
from util.open_set import OpenSetProfile, apply_open_set, build_profile, load_npz, normalize_features, save_npz
from util.recording_features import primary_track_indices


class MainlineMahalanobisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.classes = {"A":0,"B":1}
        self.signature = {"sha256":"fixture"}
        archive(self.root/"ref.npz",[[1,.02,0],[1,-.02,0],[.02,1,0],[-.02,1,0]],
                ["A","A","B","B"],"ref",classes=self.classes)
        archive(self.root/"cal.npz",[[1,.04,0],[.04,1,0]], ["A","B"],"cal",classes=self.classes)

    def build(self, target="profile.npz", **kwargs):
        metadata = build_profile(self.root/"ref.npz",self.root/"cal.npz",self.root/target,
            min_calibration_per_class=1,**kwargs)
        return metadata,OpenSetProfile(self.root/target,self.signature,self.classes)

    def test_default_profile_roundtrip_uses_maha_primary_and_head_labels(self):
        metadata,profile = self.build()
        self.assertEqual(metadata["score_method"],"maha_pp_shrink")
        self.assertEqual(metadata["calibration_unit"],"primary")
        self.assertEqual(metadata["known_label_policy"],"head")
        self.assertEqual(metadata["calibration_values"],2)
        # k=5 is unused by this scorer: two independent references/class suffice.
        self.assertEqual(profile.k,5)
        self.assertLess(profile.score([[1,0,0]])[0],profile.threshold)
        self.assertGreater(profile.score([[0,0,1]])[0],profile.threshold)

    def test_nearest_mean_is_diagnostic_and_cannot_override_voting_head(self):
        _,profile = self.build()
        track={"status":"ok","track_id":"T1","class_name":"B","class_id":1}
        group={"track_id":"T1","vote_accepted":True,"class_name":"B","embedding":[1,0,0]}
        results=[{"path":"query.dat","signals":[track]}]
        self.assertFalse(apply_open_set(profile,results,{"query.dat":[{"groups":[group]}]}))
        self.assertEqual(track["profile_class_name"],"A")
        self.assertEqual((track["class_name"],track["class_id"]),("B",1))

    def test_secondary_calibration_outlier_cannot_inflate_primary_threshold(self):
        first,_ = self.build()
        meta,arrays=load_npz(self.root/"cal.npz","track_features")
        second=copy.deepcopy(meta["records"][0]);second.update(track_id="secondary",is_primary=False)
        meta["records"].append(second)
        save_npz(self.root/"extra.npz",meta,features=np.vstack([arrays["features"],[0,0,1]]))
        altered=build_profile(self.root/"ref.npz",self.root/"extra.npz",self.root/"other.npz",min_calibration_per_class=1)
        self.assertEqual(first["threshold"],altered["threshold"])
        maximum=build_profile(self.root/"ref.npz",self.root/"extra.npz",self.root/"max.npz",
            min_calibration_per_class=1,calibration_unit="record_max")
        self.assertGreater(maximum["threshold"],first["threshold"])

    def test_reference_fit_is_unchanged_by_calibration_shift(self):
        self.build()
        meta,arrays=load_npz(self.root/"cal.npz","track_features")
        save_npz(self.root/"shifted.npz",meta,features=normalize_features([[1,.3,0],[.3,1,0]]))
        build_profile(self.root/"ref.npz",self.root/"shifted.npz",self.root/"shift_profile.npz",min_calibration_per_class=1)
        a,x=load_npz(self.root/"profile.npz","open_set_profile")
        b,y=load_npz(self.root/"shift_profile.npz","open_set_profile")
        np.testing.assert_array_equal(x["class_means"],y["class_means"])
        np.testing.assert_array_equal(x["precision"],y["precision"])
        self.assertNotEqual(a["threshold"],b["threshold"])

    def test_duplicating_all_components_of_one_record_adds_no_weight(self):
        x=normalize_features([[1,.1,0],[.1,1,0],[1,-.2,0],[0,1,.1],[0,1,-.1]])
        rows=[{"recording_id":r,"true_class_name":c} for r,c in [("a1","A"),("a1","A"),("a2","A"),("b1","B"),("b2","B")]]
        a=RecordingMahalanobis.fit(x,rows,["A","B"])
        b=RecordingMahalanobis.fit(np.vstack([x,np.tile(x[:2],(8,1))]),rows+rows[:2]*8,["A","B"])
        np.testing.assert_allclose(a.means,b.means,atol=1e-12)
        np.testing.assert_allclose(a.precision,b.precision,rtol=1e-12)

    def test_missing_primary_is_not_replaced_by_secondary(self):
        records=[{"path":"record","track_id":"secondary","recording_id":"one"}]
        files=[{"path":"record","signals":[{"track_id":"missing","valid_votes":2,"observation_count":2},
            {"track_id":"secondary","valid_votes":1,"observation_count":1}]}]
        self.assertEqual(len(primary_track_indices(records,files)),0)

    def test_old_archives_need_primary_annotations_for_the_new_method(self):
        meta,arrays=load_npz(self.root/"cal.npz","track_features")
        for r in meta["records"]:r.pop("is_primary")
        save_npz(self.root/"old.npz",meta,**arrays)
        with self.assertRaisesRegex(ValueError,"regenerate"):
            build_profile(self.root/"ref.npz",self.root/"old.npz",self.root/"old_profile.npz",min_calibration_per_class=1)

    def test_corrupt_precision_and_label_policy_fail_before_inference(self):
        self.build()
        metadata,arrays=load_npz(self.root/"profile.npz","open_set_profile")
        for name,meta,precision in [("precision",metadata,-np.eye(3)),
                ("policy",{**metadata,"known_label_policy":"nearest_reference"},arrays["precision"])]:
            save_npz(self.root/f"{name}.npz",meta,**{**arrays,"precision":precision})
            with self.assertRaises(ValueError):OpenSetProfile(self.root/f"{name}.npz",self.signature,self.classes)

    def test_legacy_class_knn_profile_remains_explicitly_available(self):
        meta,profile=self.build(k=1,score_method="record_class_knn")
        self.assertEqual(profile.score_method,"record_class_knn")
        self.assertEqual(profile.known_label_policy,"nearest_reference")
        self.assertEqual(profile.score_details([[1,0,0]])[1],["A"])

    def test_invalid_covariance_is_rejected_before_loading_archives(self):
        with self.assertRaisesRegex(ValueError,"shrinkage"):
            build_profile("missing_ref.npz","missing_cal.npz",self.root/"invalid.npz",covariance_shrinkage=-.1)

    def test_mainline_public_config_selects_the_complete_baseline(self):
        from util.cli import build_task_parser,load_settings,settings_tokens
        from util.open_set_experiment import stage_arguments
        path=Path(__file__).resolve().parents[1]/"config/experiment_iq.yaml"
        parser=build_task_parser("experiment");settings,_=load_settings(path,"experiment")
        args=parser.parse_args(settings_tokens(parser,settings))
        train,profile=stage_arguments(args,path)
        self.assertEqual((train.train_group_policy,train.supcon_weight),("primary",0.))
        self.assertEqual((train.backbone,train.mask_img_size,train.mask_pretrained),("resnet18",224,True))
        self.assertEqual((profile.profile_score_method,profile.profile_calibration_unit,profile.profile_covariance_shrinkage),
                         ("maha_pp_shrink","primary",.1))
        self.assertEqual(profile.known_false_reject_rate,.01)
        self.assertEqual((profile.unknown_input_mode,profile.unknown_cluster_policy),("single","auto"))
        self.assertEqual((profile.unknown_cluster_backend,profile.unknown_min_cluster_size,profile.unknown_min_samples,
                          profile.unknown_min_recordings,profile.unknown_allow_single_cluster),("hdbscan",5,3,2,True))


if __name__=="__main__":
    unittest.main()
