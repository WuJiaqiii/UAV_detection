"""Record-balanced Mahalanobis++ adaptation shared by profiles and ablations.

Features are normalized before fitting/scoring. Class means and a shared shrunk
covariance are fitted only on labeled known reference recordings.
"""
import numpy as np

from util.recording_features import annotated_primary_indices, calibrate_scores, record_weights, record_groups

SCORING_VERSION = "record-balanced-l2-maha-shrink-v1"
MIN_REFERENCE_RECORDINGS = 2


class RecordingMahalanobis:
    def __init__(self, names, means, precision):
        self.names = list(names)
        self.means = np.asarray(means, dtype=np.float64)
        self.precision = np.asarray(precision, dtype=np.float64)
        if (not self.names or len(set(self.names)) != len(self.names) or
                any(not isinstance(n, str) or not n for n in self.names) or
                self.means.ndim != 2 or self.means.shape[0] != len(self.names) or not self.means.shape[1] or
                self.precision.shape != (self.means.shape[1], self.means.shape[1]) or
                not np.isfinite(self.means).all() or not np.isfinite(self.precision).all() or
                not np.allclose(self.precision, self.precision.T, rtol=1e-7, atol=1e-8)):
            raise ValueError("Invalid Mahalanobis class means/precision")
        try:
            np.linalg.cholesky((self.precision+self.precision.T)/2)
        except np.linalg.LinAlgError as error:
            raise ValueError("Mahalanobis precision must be positive definite") from error

    @classmethod
    def fit(cls, features, records, names, shrinkage=.1):
        reference = np.asarray(features, dtype=np.float64)
        if (reference.ndim != 2 or len(reference) != len(records) or not len(reference) or
                not np.isfinite(reference).all() or not np.allclose(np.linalg.norm(reference,axis=1),1.,atol=1e-5)):
            raise ValueError("Mahalanobis fit requires aligned L2-normalized reference features")
        if not np.isfinite(shrinkage) or not 0 < shrinkage <= 1:
            raise ValueError("profile_covariance_shrinkage must be in (0,1]")
        labels = np.asarray([r["true_class_name"] for r in records])
        if set(labels) != set(names):
            raise ValueError("Every active known class needs reference data")
        recording_labels = {}
        for row in records:
            if recording_labels.setdefault(row["recording_id"],row["true_class_name"])!=row["true_class_name"]:
                raise ValueError("One reference recording has conflicting class labels")
        weights = record_weights(records)
        means, residual = [], reference.copy()
        for name in names:
            keep = labels == name
            mean = np.average(reference[keep],axis=0,weights=weights[keep])
            means.append(mean)
            residual[keep] -= mean
        covariance = residual.T @ (weights[:,None]*residual)
        dimension = covariance.shape[0]
        variance = max(float(np.trace(covariance)/dimension),1e-12)
        covariance = (1-shrinkage)*covariance + shrinkage*variance*np.eye(dimension)
        return cls(names,np.asarray(means),np.linalg.inv(covariance))

    def details(self, features):
        query = np.asarray(features,dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != self.means.shape[1] or not np.isfinite(query).all():
            raise ValueError("Invalid Mahalanobis query features")
        distance = np.stack([np.sum(((query-mean) @ self.precision)*(query-mean),axis=1)
                             for mean in self.means],axis=1)
        index = distance.argmin(axis=1)
        return np.maximum(distance[np.arange(len(query)),index],0.), [self.names[i] for i in index]


def build_mahalanobis_profile(reference, ref_records, calibration, cal_records, classes,
                              minimum_calibration, maximum_reference, false_reject_rate,
                              shrinkage=.1, calibration_unit="primary", source_count=None):
    primary = annotated_primary_indices(cal_records) if calibration_unit=="primary" else None
    primary_set = set(primary.tolist()) if primary is not None else None
    per_class, keep_ids = {}, set()
    for name in classes:
        ref_ids = sorted({r["recording_id"] for r in ref_records if r["true_class_name"]==name})
        cal_ids = {r["recording_id"] for r in cal_records if r["true_class_name"]==name}
        eligible = {cal_records[i]["recording_id"] for i in primary if cal_records[i]["true_class_name"]==name} if primary is not None else cal_ids
        if len(ref_ids) < MIN_REFERENCE_RECORDINGS:
            raise ValueError(f"Need at least {MIN_REFERENCE_RECORDINGS} independent reference recordings for {name}")
        if len(eligible) < minimum_calibration:
            raise ValueError(f"Need calibration tracks from {minimum_calibration} independent known recordings for {name}; got {len(eligible)} eligible {calibration_unit} recordings")
        if len(ref_ids) > maximum_reference:
            ref_ids = [ref_ids[int(i)] for i in np.linspace(0,len(ref_ids)-1,maximum_reference,dtype=int)]
        keep_ids.update(ref_ids)
        per_class[name] = {"reference_recordings":len(ref_ids),"calibration_recordings":len(cal_ids),
                           "eligible_calibration_recordings":len(eligible)}
    selected = [i for i,r in enumerate(ref_records) if r["recording_id"] in keep_ids]
    reference = reference[selected]
    selected_records = [ref_records[i] for i in selected]
    scorer = RecordingMahalanobis.fit(reference,selected_records,list(classes),shrinkage)
    scores,_ = scorer.details(calibration)
    fitted,values = calibrate_scores(scores,cal_records,calibration_unit,false_reject_rate,primary,source_count)
    for name in classes:
        rows = [i for i,r in enumerate(cal_records) if r["true_class_name"]==name]
        per_class[name]["reference_tracks"] = sum(r["true_class_name"]==name for r in selected_records)
        per_class[name]["calibration_tracks"] = len(rows)
        per_class[name]["empirical_track_false_reject_rate"] = float(np.mean(scores[rows]>fitted["threshold"]))
        if primary is not None:
            rows = [i for i in rows if i in primary_set]
            rate = float(np.mean(scores[rows]>fitted["threshold"]))
        elif calibration_unit=="record_max":
            maxima = [scores[g].max() for g in record_groups(cal_records) if cal_records[g[0]]["true_class_name"]==name]
            rate = float(np.mean(np.asarray(maxima)>fitted["threshold"]))
        else:
            weights = record_weights([cal_records[i] for i in rows])
            rate = float(weights[scores[rows]>fitted["threshold"]].sum())
        per_class[name]["empirical_false_reject_rate"] = rate
    metadata = {**fitted,"score_method":"maha_pp_shrink","scoring_version":SCORING_VERSION,
        "known_label_policy":"head","knn_k_used":False,"covariance_shrinkage":shrinkage,
        "score":"minimum squared Mahalanobis distance to class means on L2 track features",
        "reference_unit":"all reference tracks, each independent recording has total weight one",
        "covariance":"record-weighted shared covariance, shrunk toward isotropic",
        "class_order":scorer.names,"per_class":per_class,
        "reference_recording_count":len(keep_ids),"calibration_recording_count":len({r["recording_id"] for r in cal_records}),
        "reference_components":selected_records}
    arrays = {"reference":reference,"reference_labels":np.asarray([r["true_class_name"] for r in selected_records]),
        "reference_recording_ids":np.asarray([r["recording_id"] for r in selected_records]),
        "class_means":scorer.means,"precision":scorer.precision,"calibration_scores":values,
        "calibration_track_scores":scores}
    if primary is not None:
        arrays["calibration_recording_ids"] = np.asarray([cal_records[i]["recording_id"] for i in primary])
    elif calibration_unit=="record_max":
        arrays["calibration_recording_ids"] = np.asarray([cal_records[g[0]]["recording_id"] for g in record_groups(cal_records)])
    else:
        arrays["calibration_recording_ids"] = np.asarray([r["recording_id"] for r in cal_records])
        arrays["calibration_weights"] = record_weights(cal_records)
    return metadata,arrays
