"""Physical burst events shared by full inference, fast output and submission.

Separate same-frequency events remain separate. Duplicate observations require
time overlap; stitching additionally requires explicit adjacent window edges.
"""
import hashlib
import json
import math


def _number(value, name):
    if isinstance(value,bool): raise ValueError(f'Invalid {name}')
    try: result=float(value)
    except (ValueError,TypeError) as exc: raise ValueError(f'Invalid {name}') from exc
    if not math.isfinite(result): raise ValueError(f'Nonfinite {name}')
    return result


def _iou(a,b,c,d):
    overlap=max(0.,min(b,d)-max(a,c))
    return overlap/((b-a)+(d-c)-overlap)


def normalize_bursts(events, file_identity='', duplicate_iou=.8, frequency_iou=.8, edge_tolerance_ms=1e-6):
    if not 0<duplicate_iou<=1 or not 0<frequency_iou<=1 or not math.isfinite(edge_tolerance_ms) or edge_tolerance_ms<0:
        raise ValueError('Invalid burst normalization gates')
    prepared=[]
    for i,source in enumerate(events):
        e=dict(source)
        for key in ('start_ms','end_ms','frequency_low_offset_mhz','frequency_high_offset_mhz'):
            e[key]=_number(e.get(key),key)
        if e['start_ms']<0 or e['end_ms']<=e['start_ms']:
            raise ValueError('Burst requires 0 <= start_ms < end_ms')
        if e['frequency_high_offset_mhz']<=e['frequency_low_offset_mhz']:
            raise ValueError('Burst frequency interval must have positive width')
        if ('window_start_ms' in e) != ('window_end_ms' in e):
            raise ValueError('Both source window edges are required')
        if 'window_start_ms' in e:
            w0,w1=(_number(e[k],k) for k in ('window_start_ms','window_end_ms'))
            if not 0 <= w0 <= e['start_ms'] < e['end_ms'] <= w1:
                raise ValueError('Burst time lies outside its source window')
            e.update(window_start_ms=w0,window_end_ms=w1)
        e['source_observation_ids']=sorted(set(e.get('source_observation_ids',[]) or
            [e.get('observation_id',f"{e.get('track_id','')}:{e['start_ms']:.9g}:{e['end_ms']:.9g}")]))
        prepared.append(e)
    sort_key=lambda e:(e['start_ms'],e['end_ms'],e['frequency_low_offset_mhz'],str(e.get('label','')),str(e.get('track_id','')))
    prepared.sort(key=sort_key)
    result=[];diagnostics={'input_observations':len(prepared),'duplicates_removed':0,'boundary_fragments_joined':0,'label_conflicts':0}
    for event in prepared:
        matched=False
        for previous in reversed(result):
            # No frequency-only unions; a positive time gap is a genuine separation
            # unless both fragments actually touch the same source-window edge.
            fiou=_iou(previous['frequency_low_offset_mhz'],previous['frequency_high_offset_mhz'],
                      event['frequency_low_offset_mhz'],event['frequency_high_offset_mhz'])
            if fiou<frequency_iou: continue
            tiou=_iou(previous['start_ms'],previous['end_ms'],event['start_ms'],event['end_ms'])
            same_label=(previous.get('label'),previous.get('recognition_status'))==(event.get('label'),event.get('recognition_status'))
            duplicate=tiou>=duplicate_iou
            boundary=False
            overlap_fragment=False
            if all(k in previous and k in event for k in ('window_start_ms','window_end_ms')):
                edge=previous['window_end_ms']
                overlap=max(0.,min(previous['end_ms'],event['end_ms'])-max(previous['start_ms'],event['start_ms']))
                end_cut=abs(previous['end_ms']-edge)<=edge_tolerance_ms
                start_cut=abs(event['start_ms']-event['window_start_ms'])<=edge_tolerance_ms
                same_start=abs(previous['start_ms']-event['start_ms'])<=edge_tolerance_ms
                same_end=abs(previous['end_ms']-event['end_ms'])<=edge_tolerance_ms
                overlap_fragment=(overlap>0 and previous['window_start_ms']!=event['window_start_ms'] and
                    ((end_cut and (same_start or start_cut)) or (start_cut and same_end)))
                boundary=(previous['window_start_ms']<event['window_start_ms'] and
                    abs(edge-event['window_start_ms'])<=1e-9 and
                    abs(previous['end_ms']-edge)<=edge_tolerance_ms and
                    abs(event['start_ms']-edge)<=edge_tolerance_ms)
            if (duplicate or boundary or overlap_fragment) and not same_label:
                diagnostics['label_conflicts']+=1
                continue
            if not (duplicate or boundary or overlap_fragment): continue
            previous['start_ms']=min(previous['start_ms'],event['start_ms'])
            previous['end_ms']=max(previous['end_ms'],event['end_ms'])
            previous['frequency_low_offset_mhz']=min(previous['frequency_low_offset_mhz'],event['frequency_low_offset_mhz'])
            previous['frequency_high_offset_mhz']=max(previous['frequency_high_offset_mhz'],event['frequency_high_offset_mhz'])
            previous['source_observation_ids']=sorted(set(previous['source_observation_ids']+event['source_observation_ids']))
            if 'window_end_ms' in previous and 'window_end_ms' in event:
                previous['window_end_ms']=max(previous['window_end_ms'],event['window_end_ms'])
                previous['window_start_ms']=min(previous['window_start_ms'],event['window_start_ms'])
            diagnostics['duplicates_removed' if duplicate or overlap_fragment else 'boundary_fragments_joined']+=1
            matched=True;break
        if not matched: result.append(event)
    result.sort(key=sort_key)
    for e in result:
        identity=[file_identity,e['start_ms'],e['end_ms'],e['frequency_low_offset_mhz'],e['frequency_high_offset_mhz'],e.get('label')]
        e['burst_id']='B'+hashlib.sha256(json.dumps(identity,ensure_ascii=False).encode()).hexdigest()[:16]
    diagnostics['output_events']=len(result)
    return result,diagnostics


def collect_raw_bursts(result, rows):
    """Attach provenance before fast mode releases the detailed slice rows."""
    tracks={t['track_id']:t for t in result['signals']}
    for t in tracks.values():t['_submission_bursts']=[]
    for row in rows:
        for group in row.get('groups',[]):
            for p in group.get('pulses',[]):
                tracks[group['track_id']]['_submission_bursts'].append({**p,
                    'track_id':group['track_id'],'observation_id':group['observation_id'],
                    'window_start_ms':row.get('slice_start_ms',group.get('slice_start_ms',0.)),
                    'window_end_ms':row.get('slice_end_ms',group.get('slice_end_ms',50.))})


def finalize_bursts(results, rows_by_file=None):
    for result in results:
        tracks={t['track_id']:t for t in result.get('signals',[])}
        if rows_by_file is not None and not any('_submission_bursts' in t for t in tracks.values()):
            collect_raw_bursts(result,rows_by_file.get(result['path'],[]))
        events=[]
        for t in tracks.values():
            for p in t.get('_submission_bursts',[]):
                status=t.get('recognition_status','known' if t.get('status')=='ok' else 'uncertain')
                events.append({**p,'label':t.get('display_label') or t.get('class_name') or 'uncertain',
                               'recognition_status':status})
            t['submission_bursts']=[]
        events,diagnostics=normalize_bursts(events,result['path'])
        for e in events:tracks[e['track_id']]['submission_bursts'].append(e)
        result['bursts']=events
        result['burst_diagnostics']=diagnostics
