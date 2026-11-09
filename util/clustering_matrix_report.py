"""Development-only matrices; one winning recipe supplies all cell metrics."""
import csv
import html
import json
from pathlib import Path

import numpy as np

from util.clustering_benchmark import fingerprint,score_partition

METRICS=['unknown_accuracy','all_accuracy','clustered_unknown_ari','clustered_unknown_nmi',
         'unknown_ari_with_missing_singletons','unknown_cluster_coverage','estimated_unknown_class_count',
         'known_accuracy','known_false_reject_rate','unknown_false_accept_rate','fit_seconds']


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def aggregate(rows):
    groups={}
    for row in rows:
        if row['status']!='complete' or row['mode']=='fixed_k_diagnostic' or not row.get('converged',True):continue
        recipe={k:row[k] for k in ['family','params','pca_dimension','mode']}
        key=(row['dataset'],fingerprint(recipe)[:16])
        groups.setdefault(key,[]).append(row)
    result=[]
    for (dataset,key),values in groups.items():
        first=values[0];row={k:first[k] for k in ['family','params','pca_dimension','mode']}
        row.update(dataset=dataset,recipe_id=key,repeats=len(values),seeds=[v['seed'] for v in values],
                   representative_case_id=values[0]['case_id'])
        expected={r['seed'] for r in rows if r['dataset']==dataset and r['family']==first['family'] and
                  r['params']==first['params'] and r['pca_dimension']==first['pca_dimension'] and
                  r['mode'] in ('auto','fixed_k_diagnostic')}
        row['expected_repeats']=len(expected)
        row['complete_seed_coverage']={v['seed'] for v in values}==expected
        for metric in METRICS:
            numbers=[v.get(metric) for v in values if v.get(metric) is not None]
            row[metric+'_mean']=float(np.mean(numbers)) if numbers else None
            row[metric+'_std']=float(np.std(numbers)) if numbers else None
        result.append(row)
    return sorted(result,key=lambda r:(r['dataset'],-(r['unknown_accuracy_mean'] or 0),r['recipe_id']))


def make_report(output,rows,datasets):
    output=Path(output);aggregates=aggregate(rows)
    flat=[{k:v for k,v in r.items() if k not in {'labels','diagnostics','warnings'}} for r in rows]
    write_csv(output/'results_long.csv',flat);write_csv(output/'aggregate_results.csv',aggregates)
    winners={}
    for row in aggregates:
        if not row['complete_seed_coverage']:continue
        key=(row['dataset'],row['family'],row['pca_dimension'])
        rank=lambda r:(r['unknown_accuracy_mean'] or 0,r['unknown_cluster_coverage_mean'] or 0,
                       r['unknown_ari_with_missing_singletons_mean'] if r['unknown_ari_with_missing_singletons_mean'] is not None else -1)
        if key not in winners or rank(row)>rank(winners[key]):winners[key]=row
    write_csv(output/'development_cell_winners.csv',list(winners.values()))
    # The same parameter recipe must have results for every dataset to enter
    # the cross-dataset table. Each dataset contributes equally, not by size.
    grouped={}
    for row in aggregates:grouped.setdefault(row['recipe_id'],[]).append(row)
    cross=[]
    for recipe,values in grouped.items():
        if len(values)!=len(datasets) or not all(r['complete_seed_coverage'] for r in values):continue
        first=values[0];record={k:first[k] for k in ['family','params','pca_dimension','mode']}
        record.update(recipe_id=recipe,datasets=len(values))
        for metric in METRICS:
            data=[r[metric+'_mean'] for r in values if r[metric+'_mean'] is not None]
            record[metric+'_macro']=float(np.mean(data)) if len(data)==len(datasets) else None
        cross.append(record)
    cross.sort(key=lambda r:-(r['unknown_accuracy_macro'] or 0))
    write_csv(output/'cross_dataset_development_ranking.csv',cross)
    matrices={}
    for dataset in datasets:
        name=dataset['name'];dims=sorted({r['pca_dimension'] for r in rows if r['dataset']==name})
        families=sorted({r['family'] for r in rows if r['dataset']==name})
        for metric in ['unknown_accuracy','clustered_unknown_ari','unknown_cluster_coverage']:
            table=[]
            for family in families:
                row={'algorithm':family}
                for dim in dims:
                    winner=winners.get((name,family,dim))
                    row['raw' if dim==0 else f'pca_{dim}']=winner.get(metric+'_mean') if winner else None
                table.append(row)
            filename=f'{name}_{metric}_matrix.csv';write_csv(output/filename,table);matrices[(name,metric)]=table
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[]
    for (name,metric),table in matrices.items():
        if metric!='unknown_accuracy' or not table:continue
        columns=[k for k in table[0] if k!='algorithm']
        array=np.array([[r[c] if r[c] is not None else np.nan for c in columns] for r in table])
        fig,ax=plt.subplots(figsize=(max(6,len(columns)*1.8),max(5,len(table)*.48)))
        im=ax.imshow(array,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(len(columns)),columns);ax.set_yticks(range(len(table)),[r['algorithm'] for r in table])
        for i in range(len(table)):
            for j in range(len(columns)):
                ax.text(j,i,'--' if np.isnan(array[i,j]) else f'{array[i,j]*100:.1f}%',ha='center',va='center',color='white' if array[i,j]>.65 else 'black')
        ax.set_title(name+'\nDevelopment-best mean unknown ACC (selected parameters)')
        fig.colorbar(im,ax=ax,label='Unknown ACC; all true unknown recordings')
        fig.tight_layout();filename=name+'_unknown_accuracy.png';fig.savefig(output/filename,dpi=170);plt.close(fig);figures.append(filename)
    source_summary=[]
    for dataset in datasets:
        name=dataset['name'];baseline=score_partition(dataset,np.full(len(dataset['ids']),-1))['metrics']
        source_summary.append({'name':name,'source':dataset['root'],'upstream_errors':dataset['upstream_errors'],
            'known_classes':len(dataset['classes']),'held_out_classes':dataset['unknown_classes'],
            'fit_primary_recordings':len(dataset['ids']),'known_test_recordings':baseline['known_recordings'],
            'unknown_test_recordings':baseline['unknown_recordings'],'frozen_known_accuracy':baseline['known_accuracy'],
            'unknown_accuracy_ceiling_from_frozen_rejection':baseline['unknown_detection_recall']})
        eligible=[r for r in aggregates if r['dataset']==name and r['complete_seed_coverage']]
        if not eligible:continue
        winner=max(eligible,key=lambda r:(r['unknown_accuracy_mean'] or 0,r['unknown_cluster_coverage_mean'] or 0))
        case=next(r for r in rows if r['dataset']==name and r['case_id']==winner['representative_case_id'])
        evaluation=score_partition(dataset,np.asarray(case['labels']))
        true_names=dataset['unknown_classes'];clusters=sorted({s['unknown_cluster_id'] for s in evaluation['samples'] if s['unknown_cluster_id']})
        columns=clusters+['Accepted known','Unassigned'];matrix=np.zeros((len(true_names),len(columns)),dtype=int)
        for sample in evaluation['samples']:
            if not sample['is_unknown']:continue
            column=sample['unknown_cluster_id'] or ('Accepted known' if sample['recognition_status']=='known' else 'Unassigned')
            matrix[true_names.index(sample['true_class_name']),columns.index(column)]+=1
        fig,ax=plt.subplots(figsize=(max(8,len(columns)*1.2),4.5))
        ax.imshow(matrix,cmap='Blues',aspect='auto');ax.set_xticks(range(len(columns)),columns,rotation=35,ha='right')
        ax.set_yticks(range(len(true_names)),true_names)
        for i in range(len(true_names)):
            for j in range(len(columns)):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
        ax.set_title(name+'\nDevelopment-winning recipe, first seed: '+winner['family'])
        fig.tight_layout();filename=name+'_winner_unknown_confusion.png';fig.savefig(output/filename,dpi=170);plt.close(fig);figures.append(filename)
    summary={'scope':'Development-selected performance on previously inspected frozen-feature archives; not a blind test.',
             'selection':'K uses geometry only; recipe ranking uses development truth. Fixed-K diagnostic rows excluded from main ranking.',
             'common_support':'Same minimum cluster/recording support filter for every algorithm; rejected known contamination retained.',
             'cases':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
             'nonconverged_raw_fits':sum(r['mode']!='label_free_selected' and not r.get('converged',True) for r in rows),
             'datasets':source_summary,'cross_dataset_top':cross[:20]}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    payload=json.dumps(flat,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e')
    note='本表用于开发集选型。主排名排除固定K诊断行；每个热图单元格按开发真值选参数，存在选择乐观偏差。两组模型与类别划分不同，跨组平均仅代表开发一致性。未覆盖记录仍计入未知准确率分母。'
    pictures=''.join(f'<img src="{html.escape(f)}" alt="Unknown accuracy matrix">' for f in figures)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>未知聚类结果矩阵</title>
<style>body{font:15px system-ui;margin:32px;color:#162c40}p{max-width:1000px;line-height:1.7}input,select{padding:8px;margin:8px}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #dbe3e9;padding:7px}th{position:sticky;top:0;background:#e6eff5;cursor:pointer}img{max-width:900px;width:100%;display:block}#wrap{overflow:auto;max-height:650px}</style>
<h1>固定特征的未知类别聚类对照</h1><p>NOTE</p><p>下载完整数据：<a href="results_long.csv">逐配置结果</a> · <a href="aggregate_results.csv">多种子汇总</a> · <a href="cross_dataset_development_ranking.csv">跨划分开发排名</a> · <a href="summary.json">范围与错误记录</a></p>
PICTURES<h2>全部配置</h2><select id="dataset"><option value="">全部数据集</option></select><select id="mode"><option value="main">主结果（自动选K）</option><option value="all">含固定K诊断</option></select><input id="search" placeholder="搜索算法或参数"><span id="count"></span><div id="wrap"><table id="table"></table></div>
<script>const rows=PAYLOAD;const cols=['dataset','family','pca_dimension','seed','mode','k','retained_k','unknown_accuracy','clustered_unknown_ari','unknown_cluster_coverage','all_accuracy','fit_seconds','status','params'];let sort='unknown_accuracy',desc=true;
const dataset=document.getElementById('dataset'),mode=document.getElementById('mode'),search=document.getElementById('search'),table=document.getElementById('table');for(const v of [...new Set(rows.map(r=>r.dataset))]){const o=document.createElement('option');o.value=v;o.textContent=v;dataset.append(o)}
function draw(){let data=rows.filter(r=>(!dataset.value||r.dataset===dataset.value)&&(mode.value==='all'||r.mode!=='fixed_k_diagnostic')&&JSON.stringify(r).toLowerCase().includes(search.value.toLowerCase()));data.sort((a,b)=>{let x=a[sort],y=b[sort];if(x==null)return 1;if(y==null)return -1;return(typeof x==='number'?x-y:String(x).localeCompare(String(y)))*(desc?-1:1)});table.replaceChildren();const head=document.createElement('tr');for(const c of cols){const t=document.createElement('th');t.textContent=c;t.onclick=()=>{desc=sort===c?!desc:true;sort=c;draw()};head.append(t)}table.append(head);for(const r of data){const tr=document.createElement('tr');for(const c of cols){const td=document.createElement('td');let v=r[c];td.textContent=v==null?'—':typeof v==='number'&&!Number.isInteger(v)?v.toFixed(4):typeof v==='object'?JSON.stringify(v):String(v);tr.append(td)}table.append(tr)}document.getElementById('count').textContent=data.length+' 行'}dataset.onchange=mode.onchange=search.oninput=draw;draw();</script></html>'''
    page=page.replace('NOTE',html.escape(note)).replace('PICTURES',pictures).replace('PAYLOAD',payload)
    (output/'report.html').write_text(page,encoding='utf-8')
    return summary
