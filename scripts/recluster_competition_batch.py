#!/usr/bin/env python3
"""Offline task-one discovery, joint N1/N2 selection, and updated canonical CSVs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cluster_unknown_features import build_parser,run
from util.competition_protocol import protocol_for_item,choose_joint_counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--item',nargs=3,action='append',required=True,metavar=('TEST_ITEM','FEATURES_NPZ','PREDICTIONS_CSV'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-cluster-size',type=int,default=5)
    p.add_argument('--min-recordings',type=int,default=2)
    a=p.parse_args()
    items={item:(features,csv) for item,features,csv in a.item}
    if len(items)!=len(a.item) or any(not item.startswith('1-') for item in items):raise ValueError('Unique task-one test items are required')
    if a.output.exists():raise ValueError('Output must be a new directory')
    def execute(item,stage,fixed=None):
        features,predictions=items[item]
        tokens=['--features',features,'--predictions_csv',predictions,
            '--competition_profile',protocol_for_item(item).name,'--unknown_input_mode','single','--test_item',item,
            '--unknown_min_cluster_size',str(a.min_cluster_size),'--unknown_min_recordings',str(a.min_recordings),
            '--no-save_feature_plots']
        if fixed is not None:tokens.extend(['--unknown_fixed_k',str(fixed)])
        args=build_parser().parse_args(tokens);args.output_dir=a.output/item/stage
        return run(args)
    scores={};initial={}
    for item in sorted(items):
        initial[item]=execute(item,'initial')
        scores[item]={c['K']:c['score'] for c in initial[item]['diagnostics']['candidate_K_scores'] if c['valid']}
    counts=choose_joint_counts(scores);outputs={}
    for item,k in counts.items():
        if initial[item]['estimated_unknown_class_count']==k:stage='initial'
        else:execute(item,'joint',k);stage='joint'
        outputs[item]=str(a.output/item/stage/'predictions.csv')
    report={'selected_counts':counts,'predictions':outputs,'selection':'public N1<N2 bounds and frozen-feature diagnostics; no truth labels'}
    (a.output/'joint_selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
