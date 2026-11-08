"""Truth-free stage counters shared by compact and full inference reports."""
from collections import Counter
import json
from pathlib import Path


def capture_stages(result, rows):
    counts=Counter()
    gates=Counter()
    votes=Counter()
    errors=Counter()
    for row in rows:
        counts['slices']+=1
        counts['error_slices']+=row.get('status')=='error'
        if row.get('status')=='error':errors[row.get('error','unspecified')]+=1
        counts['detected_box_observations']+=int(row.get('detected_boxes',0))
        groups=row.get('groups',[])
        counts['group_observations']+=len(groups)
        counts['slices_without_boxes']+=not bool(row.get('detected_boxes',0)) and row.get('status')!='error'
        counts['slices_boxes_without_groups']+=bool(row.get('detected_boxes',0)) and not groups
        for d in row.get('detection_decisions',[]):
            if d.get('accepted') is False or d.get('accepted_as_event') is False:
                gates[d.get('reason','unspecified')]+=1
        for group in groups:
            votes[group.get('vote_reason','not_voted')]+=1
    result['stage_diagnostics']={**dict(counts),'detection_gate_reasons':dict(gates),
                                 'observation_vote_reasons':dict(votes),'slice_error_reasons':dict(errors)}


def write_diagnostics(output,results,classes=None):
    """No labels: stage/outcome counts only. No invented precision or recall."""
    stages=Counter();outcomes=Counter();gates=Counter();votes=Counter();errors=Counter();file_rows=[]
    labeled=Counter();known=set(classes or {})
    for r in results:
        d=r.get('stage_diagnostics',{})
        if d:
            stages.update({k:v for k,v in d.items() if isinstance(v,int)})
            gates.update(d.get('detection_gate_reasons',{}));votes.update(d.get('observation_vote_reasons',{}))
            errors.update(d.get('slice_error_reasons',{}))
        decision=r.get('single_signal_prediction')
        tracks=r.get('signals',[])
        if r.get('status')=='error':first='input_or_inference_error'
        elif not d:first='stage_details_unavailable'
        elif not d.get('detected_box_observations'):first='no_detection'
        elif not d.get('group_observations'):first='no_event_group'
        elif not tracks:first='no_associated_track'
        elif not any(t.get('valid_votes',0) for t in tracks):first='no_valid_classification_votes'
        else:first='classified_tracks_present'
        outcome=(decision or {}).get('recognition_status',r.get('status','unspecified'))
        outcomes[outcome]+=1
        statuses=Counter(t.get('recognition_status',t.get('status','unspecified')) for t in tracks)
        row={'path':r['path'],'first_observed_stage':first,'outcome':outcome,
             'stage_details_available':bool(d),'stages':d,'tracks':len(tracks),
             'track_outcomes':dict(statuses),'scored_tracks':sum(t.get('ood_score') is not None for t in tracks),
             'physical_bursts':len(r.get('bursts',[])),
             'burst_label_conflicts':r.get('burst_diagnostics',{}).get('label_conflicts',0)}
        truth=r.get('true_class_names')
        if decision and known and isinstance(truth,list) and len(truth)==1:
            true=truth[0];pred=decision.get('class_name');labeled['records']+=1
            if true in known:
                labeled['known_records']+=1
                key=('known_correct' if outcome=='known' and pred==true else
                     'known_wrong_class' if outcome=='known' else 'known_not_accepted')
            else:
                labeled['unknown_records']+=1
                key='unknown_false_accepted' if outcome=='known' else 'unknown_not_accepted'
            labeled[key]+=1
        file_rows.append(row)
    summary={'files':len(results),'stage_counts':dict(stages),'file_outcomes':dict(outcomes),
             'first_observed_stages':dict(Counter(r['first_observed_stage'] for r in file_rows)),
             'detection_gate_reasons':dict(gates),'observation_vote_reasons':dict(votes),
             'slice_error_reasons':dict(errors),
             'single_record_truth_diagnostics':dict(labeled) if labeled else None,
             'note':'Stage counts are diagnostics, not detection accuracy. Empty task2 scenes may legitimately have no detection.'}
    report={'schema_version':1,'summary':summary,'files':file_rows}
    Path(output,'failure_diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return summary
