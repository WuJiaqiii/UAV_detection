#!/usr/bin/env python3
"""Audit existing prediction CSVs and export resolved, normalized evaluation JSONL.

No model runs. Task2 refuses partial event export if any track lacks burst data
or a resolved label. Task1 preserves abstentions as null, never a fake class.
"""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_submission import label_mapping, normalized
from util.burst_normalization import normalize_bursts


def audit(paths,task,output,label_map=None,test_item=None,file_id_mode='stem'):
    mapping=label_mapping(label_map)
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new audit directory')
    csv.field_size_limit(16*1024*1024)
    files={};groups=defaultdict(list);records=[];issues=[];sources=[];statuses=Counter();summaries=[]
    seen_tracks=set();seen_records=set();empty_scenes=set()
    for path in map(Path,paths):
        sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        diagnostic=path.parent/'failure_diagnostics.json'
        if diagnostic.exists():summaries.append(json.loads(diagnostic.read_text(encoding='utf-8'))['summary'])
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if not {'path','label','status'} <= set(reader.fieldnames or []):
                raise ValueError('Use the run predictions.csv, not results.csv or a track-only report')
            for line,row in enumerate(reader,2):
                source=row['path'];stem=PurePosixPath(source.replace('\\','/')).stem
                if test_item and not stem.startswith(test_item+'_'):continue
                file_id=source if file_id_mode=='path' else stem
                if not source or not file_id:raise ValueError('Missing prediction path')
                if file_id in files and files[file_id]!=source:raise ValueError('Colliding file IDs; use --file-id path')
                files[file_id]=source;status=row['status'];label=row['label'];statuses[status]+=1
                cls=None
                match=re.fullmatch(r'Unknown_(\d+)',label,re.I)
                if status=='known' and normalized(label) in mapping:cls=mapping[normalized(label)]
                elif status=='novel_candidate' and match and int(match[1])>0:cls=19+int(match[1])
                if row.get('error','').strip():issues.append({'file_id':file_id,'line':line,'reason':'upstream_error','details':row['error']})
                if task=='task1':
                    if file_id in seen_records:raise ValueError('Task1 needs one row per original record')
                    seen_records.add(file_id)
                    records.append({'file_id':file_id,'class_id':cls,'recognition_status':status})
                    if cls is None:issues.append({'file_id':file_id,'line':line,'reason':'unresolved_record','label':label})
                    continue
                if status=='no_signal' and label=='no_signal':
                    empty_scenes.add(file_id)
                    if row.get('bursts_json','').strip() not in ('','[]') or row.get('track_id','').strip():
                        issues.append({'file_id':file_id,'line':line,'reason':'no_signal_has_events'})
                    groups.setdefault(file_id,[]);continue
                key=(file_id,row.get('track_id'))
                if not key[1] or key in seen_tracks:
                    issues.append({'file_id':file_id,'line':line,'reason':'missing_or_duplicate_track'})
                seen_tracks.add(key)
                if cls is None:issues.append({'file_id':file_id,'line':line,'reason':'unresolved_track','label':label})
                try:
                    bursts=json.loads(row.get('bursts_json') or '[]')
                    if not isinstance(bursts,list) or not bursts:raise ValueError('No physical burst list; rerun current inference')
                    for burst in bursts:
                        groups[file_id].append({**burst,'label':str(cls),'track_id':key[1],'class_id':cls})
                except (ValueError,TypeError) as exc:
                    issues.append({'file_id':file_id,'line':line,'reason':'missing_or_invalid_bursts','details':str(exc)})
    if not files:raise ValueError('No matching prediction records')
    datasets={key[:3] for key in files if re.fullmatch(r'[12]-[1-4]_\d{4}',key)}
    if len(datasets)>1:raise ValueError('Audit one competition test item at a time; anonymous clusters are batch-local')
    for key in empty_scenes:
        if groups[key]:issues.append({'file_id':key,'reason':'empty_scene_conflicts_with_detections'})
    diagnostics=[];events=[]
    if task=='task2':
        for file_id,bursts in sorted(groups.items()):
            try:
                normalized_events,detail=normalize_bursts(bursts,file_id)
                diagnostics.append({'file_id':file_id,**detail})
                if detail['label_conflicts']:issues.append({'file_id':file_id,'reason':'burst_label_conflicts'})
                for event in normalized_events:
                    if event['end_ms']>100.000001 or not -50<=event['frequency_low_offset_mhz']<event['frequency_high_offset_mhz']<=50:
                        issues.append({'file_id':file_id,'reason':'event_outside_capture'})
                    hz=lambda value:int((Decimal(str(value))*1000000).quantize(Decimal(1),rounding=ROUND_HALF_UP))
                    events.append({'file_id':file_id,'class_id':event['class_id'],
                        'frequency_low_hz':hz(event['frequency_low_offset_mhz']),
                        'frequency_high_hz':hz(event['frequency_high_offset_mhz']),
                        'start_ms':event['start_ms'],'end_ms':event['end_ms'],'burst_id':event['burst_id']})
            except (ValueError,TypeError) as exc:issues.append({'file_id':file_id,'reason':'invalid_event_geometry','details':str(exc)})
    metric_complete=task=='task1' or not issues
    output.mkdir(parents=True)
    filename='records.jsonl' if task=='task1' else 'events.jsonl'
    if metric_complete:
        rows=records if task=='task1' else events
        (output/filename).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    report={'schema_version':1,'task':task,'sources':sources,'input_files':len(files),
        'prediction_status_counts':dict(statuses),'physical_events':len(events) if task=='task2' else None,
        'abstained_records':sum(r['class_id'] is None for r in records) if task=='task1' else None,
        'metric_input_complete':metric_complete,'all_predictions_resolved':not issues,
        'evaluation_jsonl':str(output/filename) if metric_complete else None,
        'issues':issues,'burst_normalization':diagnostics,'pipeline_summaries':summaries,
        'note':'Diagnostic conversion only. Formal submission still requires submit.sh --check. Empty task2 files remain in file_inventory.json.'}
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'file_inventory.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',nargs='+',type=Path,required=True)
    p.add_argument('--task',choices=['task1','task2'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label-map',type=Path)
    p.add_argument('--test-item',choices=['1-1','1-2','1-3','1-4','2-1','2-2'])
    p.add_argument('--file-id',choices=['stem','path'],default='stem')
    a=p.parse_args(argv)
    report=audit(a.predictions,a.task,a.output,a.label_map,a.test_item,a.file_id)
    print(json.dumps({k:v for k,v in report.items() if k not in ['issues','burst_normalization','file_inventory','sources','pipeline_summaries']},ensure_ascii=False,indent=2))
    return 0 if report['all_predictions_resolved'] else 2


if __name__=='__main__':raise SystemExit(main())
