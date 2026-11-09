#!/usr/bin/env python3
"""Re-extract known-only reference/calibration features using an unchanged 20-class CNN."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml

from util.open_set import file_sha256,load_npz
from util.gpu_wait import GpuWaitPolicy,reserve_gpu,worker_environment


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def prepare_settings(training,output,tail_policy='cover',yolo=None):
    from prepare_submission import KNOWN,normalized
    from util.cli import profile_parser,settings_tokens
    training=training.expanduser().resolve();output=output.expanduser().resolve()
    state=read(training/'experiment.json');manifest=read(training/'split_manifest.json')
    if state.get('status')!='complete' or state.get('protocol')!='all_known':
        raise ValueError('Use a completed all_known training run')
    if manifest.get('unknown_classes') or {normalized(n) for n in manifest['known_classes']}!={normalized(n) for n in KNOWN}:
        raise ValueError('Refresh requires exactly the official 20 known classes')
    profile,_=load_npz(training/'open_set/known_profile.npz','open_set_profile')
    if {normalized(n) for n in profile['classes']}!={normalized(n) for n in KNOWN}:
        raise ValueError('Original profile does not contain the official 20 classes')
    if profile.get('score_method')!='maha_pp_shrink' or profile.get('calibration_unit')!='primary':
        raise ValueError('Refresh preserves mainline Maha++ primary calibration; use a matching original profile')
    checkpoint=training/'training/models/best.pth'
    if file_sha256(checkpoint)!=profile['signature']['details']['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since original profile; refuse a silent model switch')
    effective=training/'effective_profile.json'
    if not effective.is_file():effective=training/'effective_discover.json'
    source=read(effective);parser=profile_parser();allowed={a.dest for a in parser._actions}
    managed={'output_dir','test_output_dir','test_inputs','test_labels_json','test_label_mode','experiment_dir'}
    unexpected={k for k in source if not k.startswith('_') and k not in allowed|managed}
    if unexpected:raise ValueError('Unrecognized historical settings; review migration: '+str(sorted(unexpected)))
    settings={k:v for k,v in source.items() if k in allowed and k not in managed}
    references={};calibration={};identities={};splits={};hashes={}
    for row in manifest['records']:
        if row['split']=='excluded':continue
        if row['path'] in identities:raise ValueError('Duplicate original input path')
        if row['class_name'] not in manifest['known_classes']:raise ValueError('Non-known class in all_known manifest')
        value=(row['split'],row['class_name'])
        for mapping,key in [(splits,row['recording_id']),(hashes,row['source_sha256'])]:
            if key in mapping and mapping[key]!=value:raise ValueError('Recording/content crosses partitions')
            mapping[key]=value
        identities[row['path']]=row['recording_id']
        if row['split'] in ('train','calibration'):
            if not Path(row['path']).is_file():raise FileNotFoundError('Missing original reference/calibration input: '+row['path'])
            target=references if row['split']=='train' else calibration
            target[row['path']]=row['class_name']
    if set(references.values())!=set(manifest['known_classes']) or set(calibration.values())!=set(manifest['known_classes']):
        raise ValueError('Every known class needs original train-reference and calibration records')
    detector=Path(yolo or source['yolo_weights']).expanduser().resolve()
    if file_sha256(detector)!=profile['signature']['details']['detector_sha256']:
        raise ValueError('Detector changed since original profile; use the same YOLO weights')
    settings.update(checkpoint_path=str(checkpoint),yolo_weights=str(detector),reference_inputs=list(references),
        calibration_inputs=list(calibration),reference_labels_json=str(output/'reference_labels.json'),
        calibration_labels_json=str(output/'calibration_labels.json'),recording_ids_json=str(output/'recording_ids.json'),
        label_mode='json',labels_json=None,label_from_parent=False,label_from_filename=False,open_set_profile=None,
        export_features=False,fast=False,save_visuals=False,save_feature_plots=False,known_recheck=False,
        profile_recheck=False,profile_score_method='maha_pp_shrink',profile_calibration_unit='primary',
        signal_mode='multi',tail_policy=tail_policy,iq_stride_ms=50.,max_slices=0,
        unknown_input_mode='single',competition_profile='unconstrained',test_item=None,unknown_fixed_k=None,
        unknown_cluster_backend='hdbscan',unknown_bounded_strategy='average',unknown_cluster_policy='auto')
    settings.pop('inputs',None)
    parser.parse_args(settings_tokens(parser,settings))
    provenance=dict(training_run=str(training),checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
        detector=str(detector),detector_sha256=file_sha256(detector),source_manifest_sha256=file_sha256(training/'split_manifest.json'),
        reference_files=len(references),calibration_files=len(calibration),tail_policy=tail_policy,
        interpretation='Known-only profile refresh; CNN and detector weights unchanged; no training')
    return settings,dict(reference_labels=references,calibration_labels=calibration,recording_ids=identities),provenance,manifest


def verify_extraction(profile_path,manifest):
    expected={row['path']:row for row in manifest['records']}
    for name,split in [('reference','train'),('calibration','calibration')]:
        metadata,_=load_npz(profile_path.parent/name/'features.npz','track_features')
        sources=metadata.get('source_recordings',[])
        if {r['path'] for r in sources}!={p for p,row in expected.items() if row['split']==split}:
            raise ValueError('Refreshed feature sources do not cover the original '+split+' partition')
        if any(r['source_sha256']!=expected[r['path']]['source_sha256'] for r in sources):
            raise ValueError('Original source data changed since training; do not reuse a mismatched split')


def refresh(args):
    from util.run_context import create_directory,planned_directory
    from util.cli import build_task_parser,settings_tokens
    output=args.output.expanduser().resolve() if args.output else (planned_directory('competition_profile_refresh') if args.dry_run else create_directory('competition_profile_refresh'))
    settings,labels,provenance,manifest=prepare_settings(args.training_run,output,args.tail_policy,args.yolo_weights)
    if args.dry_run:
        print(json.dumps(dict(preflight='passed',output=str(output),**provenance),ensure_ascii=False,indent=2));return provenance
    if args.output:output.mkdir(parents=True,exist_ok=False)
    for name,value in labels.items():save(output/(name+'.json'),value)
    config=output/'profile.yaml';config.write_text(yaml.safe_dump({'profile':settings},allow_unicode=True,sort_keys=False),encoding='utf-8')
    state=dict(status='running',**provenance);target=output/'refresh_manifest.json';save(target,state)
    policy=GpuWaitPolicy(args.gpu_id,args.gpu_min_free_mib) if args.wait_for_gpu else None
    try:
        with reserve_gpu(policy) if policy else nullcontext(None) as gpu:
            command=[sys.executable,str(Path(__file__).resolve()),'_worker','--config',str(config),'--result',str(output/'profile_result.json')]
            subprocess.run(command,cwd=ROOT,check=True,**({'env':worker_environment(gpu)} if gpu else {}))
        result=read(output/'profile_result.json')
        if any(result.get('extraction_errors',{}).values()):raise ValueError('Reference/calibration extraction has errors; inspect profile run')
        profile_path=Path(result['profile']).resolve();verify_extraction(profile_path,manifest)
        metadata,_=load_npz(profile_path,'open_set_profile')
        if set(metadata['classes'])!=set(manifest['known_classes']):raise ValueError('Refreshed profile class mapping changed')
        if metadata['signature']['details']['checkpoint_sha256']!=provenance['checkpoint_sha256']:
            raise ValueError('Refreshed profile is not bound to the original checkpoint')
        infer_parser=build_task_parser('infer');allowed={a.dest for a in infer_parser._actions}
        infer={k:v for k,v in settings.items() if k in allowed and k not in {'inputs'}}
        infer.update(inputs=['/path/to/competition/1-1'],open_set_profile=str(profile_path),export_features=False,
            fast=True,save_discovery_cache=True,unknown_input_mode='single',competition_profile='task1_n1',test_item='1-1',
            unknown_cluster_backend='spectral',unknown_bounded_strategy='spectral',spectral_neighbors=10,spectral_seed=42,spectral_n_init=10)
        infer_parser.parse_args(settings_tokens(infer_parser,infer))
        inference=output/'inference.yaml';inference.write_text(yaml.safe_dump({'infer':infer},allow_unicode=True,sort_keys=False),encoding='utf-8')
        state.update(status='complete',profile=str(profile_path),profile_sha256=file_sha256(profile_path),
                     inference_config=str(inference),profile_run=result.get('run_dir'))
        save(target,state);print(json.dumps(state,ensure_ascii=False,indent=2));return state
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
        save(target,state);raise


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='_worker':
        p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
        a=p.parse_args(argv[1:]);from util.cli import main as cli
        save(a.result,cli(['profile','--config',str(a.config)]));return 0
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('training_run',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--tail-policy',choices=['cover','drop'],default='cover');p.add_argument('--yolo-weights',type=Path)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--wait-for-gpu',action='store_true')
    p.add_argument('--gpu-id',default='0');p.add_argument('--gpu-min-free-mib',type=int,default=16384)
    refresh(p.parse_args(argv));return 0


if __name__=='__main__':raise SystemExit(main())
