#!/usr/bin/env python3
"""Source-selected pairwise probability-fusion audit for five NCT backbone heads.

Usage:
  python run_broad_hybrid_screen.py --source-dir source --target-dir target --output-dir results

The source files are the SAME 20,000-image seed-1 NCT held-out split for each backbone
(80k-trained heads); the target files are full 100k-trained heads evaluated on CRC-VAL.
CRC labels are NEVER used to select models or weights. Retrospective/test-informed
CRC-VAL comparisons are EXPLORATORY, not independent external validation.
"""
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

MODELS = ('efficientnet_b3', 'resnet50', 'densenet121', 'mobilenetv2', 'vit_b16')
CLASSES = ('ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM')
PROBS = [f'prob_{i}_{c}' for i,c in enumerate(CLASSES)]
WEIGHTS = tuple(round(x, 2) for x in np.arange(0.,1.00001,.1))

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(dirpath, name, domain):
    path = Path(dirpath) / (f'{name}.csv' if domain=='source' else f'{name}_crc_predictions.csv')
    d=pd.read_csv(path)
    need=['path','y_true','y_pred',*PROBS]
    if not set(need).issubset(d.columns):raise ValueError(f'Missing columns: {path}')
    if d.path.duplicated().any():raise ValueError(f'duplicate sample paths: {path}')
    if domain=='source' and (len(d)!=20000 or d.dataset_index.nunique()!=20000):raise ValueError(f'unexpected source split {path}')
    if domain=='target' and len(d)!=7180:raise ValueError(f'unexpected target split {path}')
    p=d[PROBS].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any(p<0) or not np.allclose(p.sum(axis=1),1.,atol=1e-6):raise ValueError(f'bad probabilities: {path}')
    if np.any(p.argmax(axis=1)!=d.y_pred.to_numpy()):raise ValueError(f'argmax mismatch: {path}')
    return d,p,path

def metrics(y,p):
    pred=p.argmax(1)
    return dict(accuracy=float(accuracy_score(y,pred)), macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,pred)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',required=True);ap.add_argument('--target-dir',required=True);ap.add_argument('--output-dir',required=True)
    a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    ds={}; ps={}; paths={}; hashes={}; y={}; identifiers={}
    for domain,folder in [('source',a.source_dir),('target',a.target_dir)]:
        for name in MODELS:
            d,p,path=load(folder,name,domain)
            ids=d.path.str.replace('\\','/',regex=False).str.lower().str.split('/').str[-1].to_numpy()
            keys=(d.dataset_index.to_numpy() if domain=='source' else ids)
            if domain not in identifiers:
                identifiers[domain]=keys; y[domain]=d.y_true.to_numpy()
            if not np.array_equal(keys,identifiers[domain]) or not np.array_equal(d.y_true.to_numpy(),y[domain]):
                raise ValueError(f'Misaligned labels or order: {domain} {name}')
            ds[(domain,name)]=d;ps[(domain,name)]=p
            hashes[f'{domain}/{name}']=sha256(path)
    rows=[];wide=pd.DataFrame({'image_id':identifiers['target'],'y_true':y['target']});source_metrics={}
    for name in MODELS:
        s=metrics(y['source'],ps[('source',name)]);t=metrics(y['target'],ps[('target',name)])
        source_metrics[name]=s
        rows.append({'kind':'single','models':name,'source_weight_first':1.,'selection':'standalone','source_accuracy':s['accuracy'],'source_macro_f1':s['macro_f1'],'crc_accuracy':t['accuracy'],'crc_macro_f1':t['macro_f1'],'crc_balanced_accuracy':t['balanced_accuracy']})
        wide[f'{name}_single']=ps[('target',name)].argmax(1)
    pairs=[]
    for a0,b0 in itertools.combinations(MODELS,2):
        sa,sb=ps[('source',a0)],ps[('source',b0)]
        ta,tb=ps[('target',a0)],ps[('target',b0)]
        variants=[('equal_vote',.5),('source_selected',None)]
        tuning=[]
        for w in WEIGHTS:
            sm=metrics(y['source'],w*sa+(1-w)*sb)
            tuning.append((w,sm['macro_f1'],sm['accuracy']))
        # best macro F1; deterministic tie-break prefers closer to .5 then larger weight
        wbest=max(tuning,key=lambda z:(z[1],z[2],-abs(z[0]-.5),z[0]))[0]
        for kind,w in [('equal_vote',.5),('source_selected',wbest)]:
            source=metrics(y['source'],w*sa+(1-w)*sb)
            q=w*ta+(1-w)*tb;target=metrics(y['target'],q)
            pred=q.argmax(1)
            rows.append({'kind':'pair','models':a0+'+'+b0,'method':kind,'source_weight_first':w,'selection':'predefined_equal' if kind=='equal_vote' else 'source_macro_f1_grid_only','source_accuracy':source['accuracy'],'source_macro_f1':source['macro_f1'],'crc_accuracy':target['accuracy'],'crc_macro_f1':target['macro_f1'],'crc_balanced_accuracy':target['balanced_accuracy'],
                         'b3_rescues':int(np.sum((wide['efficientnet_b3_single']!=y['target']) & (pred==y['target']))),
                         'b3_damages':int(np.sum((wide['efficientnet_b3_single']==y['target']) & (pred!=y['target']))),
                         'source_optimal_weight_boundary':bool(w in (0.,1.)) if kind=='source_selected' else False})
            wide[f'{a0}__{b0}__{kind}']=pred
        pairs.append({'a':a0,'b':b0,'best_source_weight_a':wbest,'source_tuning':[{'weight_a':w,'macro_f1':f,'accuracy':acc} for w,f,acc in tuning]})
    score=pd.DataFrame(rows);score.to_csv(out/'all_comparisons.csv',index=False,float_format='%.9f')
    wide.to_csv(out/'crc_prediction_audit.csv',index=False)
    with open(out/'source_weight_grid.json','w') as f:json.dump(pairs,f,indent=2)
    with open(out/'input_sha256.json','w') as f:json.dump(hashes,f,indent=2)
    top=score.sort_values(['source_macro_f1','source_accuracy'],ascending=False)
    top_single=score[score.kind=='single'].sort_values('source_macro_f1',ascending=False)
    top_pair=score[(score.kind=='pair')&(score.method=='source_selected')].sort_values('source_macro_f1',ascending=False)
    winner=top.iloc[0]
    # all 15 alternatives eligible in source selection; no CRC labels used
    with open(out/'source_only_selection.json','w') as f:json.dump({'criterion':'max seed-1 NCT source-validation macro F1 across 5 standalone and 10 source-selected pairs','winner_kind':winner.kind,'winner_models':winner.models,'winner_method':winner.get('method','standalone') if isinstance(winner.get('method'),str) else 'standalone','winner_weight_first':winner.source_weight_first,'winner_source_macro_f1':winner.source_macro_f1,'winner_crc_accuracy_exploratory':winner.crc_accuracy,'source_best_standalone':top_single.iloc[0].models,'source_best_pair':top_pair.iloc[0].models},f,indent=2)
    base=float(score.loc[(score.kind=='single')&(score.models=='efficientnet_b3'),'crc_accuracy'].iloc[0]);
    lines=['# Five-backbone hybrid breadth screen — retrospective diagnostics','',
    'The five original proposal backbones share exactly the same 20,000 NCT seed-1 validation samples and the same 7,180 CRC-VAL files. Five individual models and all 10 possible pairs were evaluated.','',
    '**Design:** each pair is evaluated with 50/50 soft voting and with a weight chosen exclusively from NCT source-validation macro-F1 on a fixed grid 0.0,0.1,...,1.0. A 0.0 or 1.0 result is a *single model*, not a hybrid. Model/pair selection likewise uses source NCT metrics only. CRC labels are for retrospective scoring, not weight fitting.','',
    '**Important protocol difference:** source validation heads were trained on 80,000 NCT images; archived CRC heads were refitted on all 100,000 NCT images. Both share LC25000-trained frozen feature extractors. The source-selected weights thus transfer across head-refitting regimes. This is not the same protocol as the B0-KD 60k/20k/20k Stage 2 comparison.','',
    '**CRC-VAL is repeatedly inspected and all target results are exploratory/test-informed.** Do not describe any selection as an untouched independent external test, a new architecture, or evidence of future-hospital generalization.','',
    f'Baseline EfficientNet-B3 external accuracy: {base*100:.3f}%.',
    f'Source-only best model/pair: {winner.models}, kind={winner.kind}, weight(first)={winner.source_weight_first:.1f}, source macro-F1={winner.source_macro_f1*100:.3f}%, CRC-VAL accuracy={winner.crc_accuracy*100:.3f}%.','',
    '## Complete pairwise results','',
    '| Pair | Source-selected weight (first) | NCT macro-F1 (%) | CRC accuracy (%) | Equal-vote CRC accuracy (%) |',
    '|---|---:|---:|---:|---:|']
    for _,r in top_pair.iterrows():
        e=score[(score.kind=='pair')&(score.models==r.models)&(score.method=='equal_vote')].iloc[0]
        lines.append(f'| {r.models.replace("+"," + ")} | {r.source_weight_first:.1f} | {r.source_macro_f1*100:.3f} | {r.crc_accuracy*100:.3f} | {e.crc_accuracy*100:.3f} |')
    lines+=['','## Single-model baselines','','| Model | NCT macro-F1 (%) | CRC accuracy (%) | CRC macro-F1 (%) |','|---|---:|---:|---:|']
    for _,r in top_single.iterrows():lines.append(f'| {r.models} | {r.source_macro_f1*100:.3f} | {r.crc_accuracy*100:.3f} | {r.crc_macro_f1*100:.3f} |')
    lines+=['','## Interpretation','',
        '- This evaluates **decision-level hybrid breadth**, not a new trainable architecture. It should be completed before proposing custom attention or other trainable mechanisms.','- Entire source-validation split was previously used for original model benchmarking; this is a retrospective secondary analysis, not preregistered independent model selection.','- No patient/slide grouping is established for source NCT; image-level independence should not be described as patient-level independence.','- The next controlled experiment (only if this screen supports it) requires a fresh NCT development protocol and then frozen evaluation of CRC, still described as exploratory because CRC was previously inspected.','- Avoid comparing these results directly to Stage 2 B0-Dense 60k-trained-head results without indicating different training protocols.']
    (out/'SCIENTIFIC_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:18]));print('\nBEST CRC pair source-selected:',score[(score.kind=='pair')&(score.method=='source_selected')].sort_values('crc_accuracy',ascending=False)[['models','source_weight_first','crc_accuracy','source_macro_f1']].head(3).to_string(index=False))

if __name__=='__main__':main()
