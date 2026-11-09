#!/usr/bin/env python3
"""CPU-only six-item cluster inspection, including task-one joint count selection."""
import argparse
from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def inspect(args):
    from util.open_set import file_sha256,load_npz
    from util.feature_visualization import write_feature_plots,write_plot_gallery
    from util.run_context import create_directory
    from scripts.competition_pipeline import ITEMS
    batch=json.loads(args.manifest.read_text(encoding='utf-8'))
    output=args.output.expanduser().resolve() if args.output else create_directory('competition_cluster_inspection')
    if args.output:output.mkdir(parents=True,exist_ok=False)
    sources={};summary={'source_manifest':str(args.manifest.resolve()),'items':{},'joint_status':'not_run',
        'interpretation':'Diagnostic only; projected distances are not accuracy. Unknown IDs are local to each test item.'}
    for item in ITEMS:
        entry=batch.get('runs',{}).get(item,{})
        if not entry.get('run_dir'):
            summary['items'][item]={'status':'missing'};continue
        folder=Path(entry['run_dir'])
        for filename,key in [('predictions.csv','csv_sha256'),('unknown_features.npz','cache_sha256')]:
            if entry.get(key) and file_sha256(folder/filename)!=entry[key]:raise ValueError('Source artifact changed: '+str(folder/filename))
        if (folder/'predictions.csv').is_file() and (folder/'unknown_features.npz').is_file():sources[item]=folder
        else:summary['items'][item]={'status':'missing_cache'}
    if args.joint_task1 and all(item in sources for item in ITEMS[:4]):
        discovery=batch['discovery'];strategy=discovery.get('strategy',batch['fingerprint'].get('strategy','density_merge'))
        command=[sys.executable,str(ROOT/'scripts/recluster_competition_batch.py'),'--output',str(output/'joint_task1'),
                 '--bounded-strategy',strategy]
        command.append('--allow-single-cluster' if discovery.get('allow_single_cluster',True) else '--no-allow-single-cluster')
        for key,flag in [('min_cluster_size','--min-cluster-size'),('min_recordings','--min-recordings'),
                         ('min_samples','--min-samples'),('min_silhouette','--min-silhouette')]:command+=[flag,str(discovery[key])]
        if strategy=='spectral':
            options=discovery['spectral_options']
            for key,flag in [('neighbors','--spectral-neighbors'),('seed','--spectral-seed'),('n_init','--spectral-n-init')]:command+=[flag,str(options[key])]
        for item in ITEMS[:4]:command+=['--item',item,str(sources[item]/'unknown_features.npz'),str(sources[item]/'predictions.csv')]
        with (output/'joint_task1.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode==0:
            joint=json.loads((output/'joint_task1/joint_selection.json').read_text())
            summary.update(joint_status='complete',joint_selected_counts=joint['selected_counts'])
            for item,path in joint['predictions'].items():sources[item]=Path(path).parent
        else:summary.update(joint_status='failed',joint_error='See joint_task1.log; displayed task-one labels are independent, not a valid joint selection')
    rows=[];links=[]
    for item,folder in sources.items():
        metadata,arrays=load_npz(folder/'unknown_features.npz','unknown_candidates');records=metadata['records']
        target=output/item;target.mkdir()
        plots=write_feature_plots(target/'feature_plots',arrays['features'],records,method=args.projection,max_points=args.max_points,seed=42)
        write_plot_gallery(target,plots)
        grouped=defaultdict(list)
        for row in records:grouped[row.get('unknown_cluster_id') or 'Unknown_unassigned'].append(row)
        for label,values in sorted(grouped.items()):
            rows.append(dict(test_item=item,cluster=label,tracks=len(values),
                recordings=len({r['recording_id'] for r in values}),
                fit_seed_tracks=sum(r.get('discovery_role') in ('track','recording_primary') for r in values)))
        diagnostics=json.loads((folder/'unknown_clusters.json').read_text()) if (folder/'unknown_clusters.json').is_file() else {}
        summary['items'][item]={'status':batch['runs'][item].get('status'),'source':str(folder),
            'unknown_candidates':len(records),'estimated_unknown_classes':len([label for label in grouped if label!='Unknown_unassigned']),
            'unassigned_tracks':len(grouped.get('Unknown_unassigned',[])),
            'clustering_diagnostics':diagnostics.get('diagnostics',{}),'feature_report':str(target/'feature_report.html')}
        links.append(f'<li><a href="{item}/feature_report.html">{item}: PCA/t-SNE</a> — {len(records)} candidate tracks</li>')
    with (output/'cluster_sizes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['test_item','cluster','tracks','recordings','fit_seed_tracks']);writer.writeheader();writer.writerows(rows)
    (output/'inspection.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>Six-item cluster inspection</title><style>body{font:16px system-ui;margin:32px;line-height:1.7}</style><h1>六项未知聚类检查</h1>'
    page+=f'<p>任务一联合选择：{escape(summary["joint_status"])}。这里只展示未知候选；已误收为已知的未知样本不在图中。二维投影不代表准确率，不同测试项的Unknown编号不代表同一型号。</p>'
    page+='<p><a href="cluster_sizes.csv">簇规模表</a> · <a href="inspection.json">K候选分数与完整诊断</a></p><ul>'+''.join(links)+'</ul>'
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({'output':str(output),'index':str(output/'index.html'),'joint_status':summary['joint_status']},ensure_ascii=False,indent=2))
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--projection',choices=['pca','tsne','both'],default='both')
    p.add_argument('--max-points',type=int,default=3000)
    p.add_argument('--joint-task1',action=argparse.BooleanOptionalAction,default=True)
    return inspect(p.parse_args(argv))


if __name__=='__main__':main()
