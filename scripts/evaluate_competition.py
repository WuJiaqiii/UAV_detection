#!/usr/bin/env python3
"""Evaluate JSONL event annotations with explicit per-event truth, never file GT."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from util.competition_metrics import evaluate_events,evaluate_records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frequency-iou',type=float,default=.5)
    p.add_argument('--time-iou',type=float)
    p.add_argument('--task',choices=['task1','task2'],default='task2')
    p.add_argument('--known-classes',nargs='+',type=int,default=list(range(20)),
                   help='Explicit active known IDs; override for held-out-class development experiments')
    a=p.parse_args()
    def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if a.task=='task1':
        if a.time_iou is not None:raise ValueError('--time-iou applies only to task2')
        report=evaluate_records(read(a.truth),read(a.predictions),a.known_classes)
    else:
        report=evaluate_events(read(a.truth),read(a.predictions),known_classes=a.known_classes,
                               frequency_threshold=a.frequency_iou,time_threshold=a.time_iou)
    if a.output.exists():raise ValueError('Choose a new output file')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='assignments'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
