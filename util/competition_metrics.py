"""Event-level diagnostics. No invented organizer total score or file-label GT."""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def interval_iou(a,b):
    a0,a1=a; b0,b1=b
    if not np.isfinite([a0,a1,b0,b1]).all() or a0>=a1 or b0>=b1:
        raise ValueError('Need finite increasing event intervals')
    intersection=max(0.,min(a1,b1)-max(a0,b0))
    return intersection/((a1-a0)+(b1-b0)-intersection)


def match_events(truth,predictions,frequency_threshold=.5,time_threshold=None):
    if not 0<=frequency_threshold<=1 or (time_threshold is not None and not 0<=time_threshold<=1):
        raise ValueError('IoU thresholds must be in [0,1]')
    costs=np.zeros((len(truth),len(predictions)))
    eligible=np.zeros(costs.shape,dtype=bool)
    for i,t in enumerate(truth):
        for j,p in enumerate(predictions):
            iou=interval_iou((t['frequency_low_hz'],t['frequency_high_hz']),
                             (p['frequency_low_hz'],p['frequency_high_hz']))
            valid=iou>=frequency_threshold
            if time_threshold is not None:
                valid=valid and interval_iou((t['start_ms'],t['end_ms']),(p['start_ms'],p['end_ms']))>=time_threshold
            eligible[i,j]=valid
            if valid:costs[i,j]=-(min(len(truth),len(predictions))+1)-iou
    rows,cols=linear_sum_assignment(costs)
    matched=[(int(i),int(j)) for i,j in zip(rows,cols) if eligible[i,j]]
    return matched


def evaluate_events(truth,predictions,known_classes=range(20),frequency_threshold=.5,time_threshold=None):
    known=set(known_classes);gts=defaultdict(list);preds=defaultdict(list)
    for destination,events in ((gts,truth),(preds,predictions)):
        for e in events:
            if not e.get('file_id') or 'class_id' not in e:raise ValueError('Events require file_id and class_id')
            interval_iou((e['frequency_low_hz'],e['frequency_high_hz']),(e['frequency_low_hz'],e['frequency_high_hz']))
            destination[e['file_id']].append(e)
    matched_total=known_tp=unknown_tp=0;ari_true=[];ari_pred=[];all_unknown_true=[];all_unknown_pred=[]
    assignments=[]
    for file_id in sorted(set(gts)|set(preds)):
        gt,pr=gts[file_id],preds[file_id]
        pairs=match_events(gt,pr,frequency_threshold,time_threshold)
        lookup=dict(pairs);matched_total+=len(pairs)
        for i,j in pairs:
            t,p=gt[i]['class_id'],pr[j]['class_id']
            known_tp+=int(t in known and t==p)
            unknown_tp+=int(t not in known and p not in known)
            if t not in known and p not in known:ari_true.append(str(t));ari_pred.append(str(p))
        for i,t in enumerate(gt):
            if t['class_id'] not in known:
                all_unknown_true.append(str(t['class_id']))
                all_unknown_pred.append(str(pr[lookup[i]]['class_id']) if i in lookup else f'missing:{file_id}:{i}')
        assignments.append({'file_id':file_id,'matches':pairs,'false_positives':len(pr)-len(pairs),'false_negatives':len(gt)-len(pairs)})
    tp=matched_total;fp=len(predictions)-tp;fn=len(truth)-tp
    def classification(correct,pred_count,true_count):
        p=correct/pred_count if pred_count else 0.;r=correct/true_count if true_count else 0.
        return {'tp':correct,'fp':pred_count-correct,'fn':true_count-correct,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}
    kt=sum(e['class_id'] in known for e in truth);kp=sum(e['class_id'] in known for e in predictions)
    return {'detection':{'tp':tp,'fp':fp,'fn':fn,'CDR':tp/(tp+fn) if tp+fn else None,'FAR':fp/(tp+fp) if tp+fp else None},
            'known_micro':classification(known_tp,kp,kt),
            'unknown_detection':classification(unknown_tp,len(predictions)-kp,len(truth)-kt),
            'matched_unknown_ari':float(adjusted_rand_score(ari_true,ari_pred)) if len(ari_true)>=2 else None,
            'matched_unknown_coverage':len(ari_true)/(len(truth)-kt) if len(truth)>kt else None,
            'all_unknown_ari_with_missing_singletons':float(adjusted_rand_score(all_unknown_true,all_unknown_pred)) if len(all_unknown_true)>=2 else None,
            'ari_missing_policy':'Each unmatched unknown truth event receives a unique missing token; report alongside coverage, not as an organizer score.',
            'matching':'frequency IoU, one-to-one maximum cardinality' if time_threshold is None else 'frequency and time IoU, one-to-one maximum cardinality',
            'assignments':assignments}
