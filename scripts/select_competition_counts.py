#!/usr/bin/env python3
"""Select joint N1<N2 from saved, label-free bounded-discovery diagnostics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from util.competition_protocol import choose_joint_counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();scores={}
    for path in a.reports:
        report=json.loads(path.read_text(encoding='utf-8'))
        item=report.get('parameters',{}).get('test_item')
        if item in scores:raise ValueError('Duplicate test item')
        scores[item]={r['K']:r['score'] for r in report['diagnostics']['candidate_K_scores'] if r['valid']}
    counts=choose_joint_counts(scores)
    if a.output.exists():raise ValueError('Choose a new output file')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'selected_counts':counts,'source_reports':list(map(str,a.reports)),
        'usage':'Re-run cluster with the matching --test_item and --unknown_fixed_k; submission independently checks all provided sets.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False))


if __name__=='__main__':main()
