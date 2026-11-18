#!/usr/bin/env python3
"""Independent saved-output verifier. Does not retrain feature extractors/heads."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score,balanced_accuracy_score
base=Path(sys.argv[1] if len(sys.argv)>1 else 'results')
scores=pd.read_csv(base/'all_comparisons.csv')
d=pd.read_csv(base/'crc_prediction_audit.csv')
y=d.y_true.to_numpy()
errors=[]
for i,r in scores.iterrows():
    if r['kind']=='single': col=f'{r.models}_single'
    else:
        a,b=r.models.split('+')
        col=f'{a}__{b}__{r.method}'
    if col not in d:errors.append(f'col missing: {col}');continue
    pred=d[col].to_numpy()
    observed=(accuracy_score(y,pred),f1_score(y,pred,average='macro'),balanced_accuracy_score(y,pred))
    expected=(r.crc_accuracy,r.crc_macro_f1,r.crc_balanced_accuracy)
    if not np.allclose(observed,expected,rtol=0,atol=2e-8):errors.append(f'mismatch {col}: {observed} != {expected}')
    if r['kind']=='pair':
        b3=d.efficientnet_b3_single.to_numpy()
        rescue=int(np.sum((b3!=y)&(pred==y)))
        harm=int(np.sum((b3==y)&(pred!=y)))
        if rescue!=int(r.b3_rescues) or harm!=int(r.b3_damages):errors.append(f'rescue count mismatch {col}')
assert len(scores)==25, f'Expected 5 singles + 10 pairs x 2 methods, got {len(scores)}'
assert len(d)==7180 and d.image_id.nunique()==7180
if errors:
    for e in errors:print('FAIL',e)
    raise SystemExit(1)
print('PASS: 25 methods, 7180 aligned target examples, accuracy/macro-F1/balanced accuracy, pair rescues/damages')
