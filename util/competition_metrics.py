"""Event-level diagnostics. No invented organizer total score or file-label GT."""
from collections import Counter, defaultdict
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
    known=_known_set(known_classes);gts=defaultdict(list);preds=defaultdict(list)
    correct_by_class=Counter();errors=Counter()
    for destination,events in ((gts,truth),(preds,predictions)):
        for e in events:
            if not e.get('file_id') or e.get('class_id') is None:raise ValueError('Events require file_id and resolved class_id')
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
            if t in known and t==p:correct_by_class[t]+=1
            unknown_tp+=int(t not in known and p not in known)
            errors['matched_correct_known' if t in known and t==p else
                   'matched_known_wrong_class' if t in known and p in known else
                   'matched_known_rejected' if t in known else
                   'matched_unknown_false_accepted' if p in known else 'matched_unknown_rejected']+=1
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
    macro=_macro_known(Counter(e['class_id'] for e in truth),Counter(e['class_id'] for e in predictions),correct_by_class,known)
    report={'detection':{'tp':tp,'fp':fp,'fn':fn,'CDR':tp/(tp+fn) if tp+fn else None,'FAR':fp/(tp+fp) if tp+fp else None},
            'known_micro':classification(known_tp,kp,kt),
            'known_macro':macro,
            'known_class_ids':sorted(known,key=str),
            'classification_diagnostics':dict(errors),
            'unknown_detection':classification(unknown_tp,len(predictions)-kp,len(truth)-kt),
            'matched_unknown_ari':float(adjusted_rand_score(ari_true,ari_pred)) if len(ari_true)>=2 else None,
            'ari_sample_count':len(ari_true),
            'matched_unknown_coverage':len(ari_true)/(len(truth)-kt) if len(truth)>kt else None,
            'all_unknown_ari_with_missing_singletons':float(adjusted_rand_score(all_unknown_true,all_unknown_pred)) if len(all_unknown_true)>=2 else None,
            'ari_missing_policy':'Each unmatched unknown truth event receives a unique missing token; report alongside coverage, not as an organizer score.',
            'matching':'frequency IoU, one-to-one maximum cardinality' if time_threshold is None else 'frequency and time IoU, one-to-one maximum cardinality',
            'assignments':assignments}
    report['feedback_style_score']=feedback_score(report,'task2')
    return report


def _known_set(classes):
    values=list(classes)
    if not values or len(set(values))!=len(values) or any(v is None or isinstance(v,bool) for v in values):
        raise ValueError('known_classes must be explicit, nonempty and unique')
    return set(values)


def _classification(tp,pred_count,true_count):
    p=tp/pred_count if pred_count else 0.;r=tp/true_count if true_count else 0.
    return dict(tp=tp,fp=pred_count-tp,fn=true_count-tp,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)


def _macro_known(truth_counts,pred_counts,correct_counts,known):
    per_class=[dict(class_id=c,support=truth_counts[c],predicted=pred_counts[c],
                    **_classification(correct_counts[c],pred_counts[c],truth_counts[c])) for c in sorted(known,key=str)]
    p=float(np.mean([r['precision'] for r in per_class]));r=float(np.mean([r['recall'] for r in per_class]))
    return dict(precision=p,recall=r,f1=float(np.mean([x['f1'] for x in per_class])),
                f1_from_macro_pr=2*p*r/(p+r) if p+r else 0.,per_class=per_class,
                averaging='all explicitly supplied known classes; zero division is zero')


def evaluate_records(truth,predictions,known_classes=range(20)):
    """One row per original recording. Missing/None predictions are abstentions.

    Unknown cluster IDs may be arbitrary resolved IDs outside known_classes.
    Never turn no_signal/error/unassigned into an extra anonymous class.
    """
    known=_known_set(known_classes)
    def index(rows,allow_abstain):
        out={}
        for row in rows:
            key=row.get('file_id');label=row.get('class_id')
            if not key or key in out or (label is None and not allow_abstain):
                raise ValueError('Require unique file_id and truth class_id')
            if isinstance(label,str) and label.casefold() in {'no_signal','error','uncertain','unknown_unassigned',''}:
                raise ValueError('Use class_id=null for abstention, never a fake unknown cluster')
            out[key]=label
        return out
    gt=index(truth,False);pr=index(predictions,True)
    if set(pr)-set(gt):raise ValueError('Predictions contain file IDs absent from truth')
    correct=Counter();unknown_true=[];unknown_pred=[];utp=0
    for key,t in gt.items():
        p=pr.get(key)
        if t in known and p==t:correct[t]+=1
        if t not in known and p is not None and p not in known:
            utp+=1;unknown_true.append(str(t));unknown_pred.append(str(p))
    tc=Counter(gt.values());pc=Counter(p for p in pr.values() if p is not None)
    kt=sum(t in known for t in gt.values());kp=sum(p in known for p in pr.values())
    uc=sum(p is not None and p not in known for p in pr.values());ut=len(gt)-kt
    report=dict(total_records=len(gt),prediction_rows=len(pr),
        abstained_records=sum(pr.get(key) is None for key in gt),
        known_class_ids=sorted(known,key=str),known_macro=_macro_known(tc,pc,correct,known),
        known_micro=_classification(sum(correct.values()),kp,kt),
        unknown_detection=_classification(utp,uc,ut),
        matched_unknown_ari=float(adjusted_rand_score(unknown_true,unknown_pred)) if len(unknown_true)>=2 else None,
        matched_unknown_coverage=utp/ut if ut else None,
        ari_sample_count=len(unknown_true),ari_scope='correctly rejected true unknown records only')
    report['feedback_style_score']=feedback_score(report,'task1')
    return report


def feedback_score(report,task):
    """Published feedback weights, with explicit local matching/ARI assumptions.

    This is a diagnostic, not a claim to reproduce the unavailable official grader.
    Empty detection denominators or undefined ARI leave the score unavailable.
    """
    ari=report.get('matched_unknown_ari')
    result={'value':None,'official_equivalence_verified':False,'ari_scope':'matched true unknown predictions only',
            'known_f1_definition':'mean_class_f1' if task=='task1' else 'harmonic_macro_precision_recall'}
    if ari is None:return result
    f=report['known_macro']['f1' if task=='task1' else 'f1_from_macro_pr'];u=report['unknown_detection']['f1']
    if task=='task1':value=.5*f+.25*u+.25*max(0.,ari)
    else:
        d=report['detection']
        if d['CDR'] is None or d['FAR'] is None:return result
        value=.3*f+.15*u+.15*max(0.,ari)+.2*d['CDR']+.2*(1-d['FAR'])
    result['value']=100*value
    return result
