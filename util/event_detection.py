"""Detection acceptance independent of classifier train-length requirements."""
import numpy as np


def select_event_groups(preprocessor, boxes, spectrum, config, scores=None, source_rate=None):
    values=np.asarray(boxes,dtype=float).reshape(-1,4)
    scores=list(scores) if scores is not None else [None]*len(values)
    if len(scores)!=len(values): raise ValueError('Detection coordinates/confidences must align')
    kept=[];confidence=[];decisions=[]
    height,width=spectrum.shape
    for box,score in zip(values,scores):
        original=box.copy()
        if not np.isfinite(box).all() or (box[2:]<=box[:2]).any():
            decisions.append({'box':box.tolist(),'accepted':False,'reason':'invalid_geometry'});continue
        reason=None
        if source_rate is not None and getattr(config,'valid_band_policy','off')=='source':
            # Keep the resampled 122.88 MHz grid, restrict support to capture Nyquist.
            radius=float(source_rate)/float(config.sampling_rate)*height/2
            low=int(np.ceil(height//2-radius));high=int(np.floor(height//2+radius))
            box[1]=max(box[1],low);box[3]=min(box[3],high)
            if box[3]<=box[1]:reason='outside_source_nyquist'
        if reason:
            decisions.append({'box':original.tolist(),'confidence':score,'accepted':False,'reason':reason});continue
        box=np.asarray([np.floor(box[0]),np.ceil(box[1]),np.ceil(box[2]),np.floor(box[3])],dtype=int)
        box[[0,2]] = np.clip(box[[0,2]], 0, width)
        box[[1,3]] = np.clip(box[[1,3]], 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            decisions.append({'box':original.tolist(),'confidence':score,'accepted':False,
                              'reason':'empty_after_quantization'});continue
        kept.append(box);confidence.append(score)
        decisions.append({'box':original.tolist(),'accepted_box':box.tolist(),'confidence':score,
                          'source_band_truncated':not np.array_equal(box,original),'accepted':True})
    boxes=np.asarray(kept,dtype=np.int32).reshape(-1,4)
    groups=preprocessor.select_signal_groups(boxes,spectrogram=spectrum) or []
    assigned={tuple(b) for g in groups for b in g['boxes'].tolist()}
    for box,score in zip(boxes,confidence):
        if tuple(box) in assigned:continue
        stats=preprocessor._box_energy_stats(box,spectrum)
        reliable=(preprocessor._basic_filter(box[None]).size>0 and score is not None and np.isfinite(score) and score>=config.singleton_min_confidence and
                  stats is not None and stats['contrast_z']>=config.singleton_min_contrast_z)
        if reliable:
            group=preprocessor._build_group_stats(box[None],[stats],spectrum.shape)
            group['group_type']='isolated_burst'
            group['acceptance_source']='detection_evidence_gate'
            groups.append(group)
        decisions.append({'box':box.tolist(),'confidence':score,'accepted_as_event':bool(reliable),
                          'reason':'isolated_detection_gate' if reliable else 'insufficient_detection_evidence'})
    gap_ms=float(getattr(config,'event_context_gap_ms',0.))
    if gap_ms>0:
        split=[]
        gap=gap_ms/float(config.iq_segment_ms)*width
        for g in groups:
            # Preserve established hopping contexts; split independent carrier trains.
            if g.get('group_type')=='hopping':split.append(g);continue
            chunks=[]
            for b in sorted(g['boxes'].tolist(),key=lambda b:b[0]):
                if not chunks or b[0]-max(x[2] for x in chunks[-1])>gap:chunks.append([])
                chunks[-1].append(b)
            for chunk in chunks:
                h=preprocessor._build_group_stats(np.asarray(chunk),None,spectrum.shape)
                h['group_type']='temporal_context'
                split.append(h)
        groups=split
    return groups,decisions
