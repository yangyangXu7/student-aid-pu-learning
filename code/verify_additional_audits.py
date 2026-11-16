"""Recompute additional diagnostic results from saved, finite raw arrays."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score
from run_r2 import ROOT

checked=0;max_error=0.
for dirname in ['denominator_audit','convergence_sensitivity','lbe_optimizer_audit']:
    for path in sorted((ROOT/'results'/dirname).rglob('*.json')):
        d=json.loads(path.read_text());z=np.load(path.with_suffix('.npz'))
        scores=z['scores'];scores=scores[None,:] if scores.ndim==1 else scores
        names=z['models'].tolist() if 'models' in z.files else [d['rows'][0]['model']]
        assert len(scores)==len(names)==len(d['rows'])
        rows={r['model']:r for r in d['rows']};y,s,ids=z['y'],z['s'],z['ids'];u=np.flatnonzero(s==0)
        for name,p in zip(names,scores):
            assert len(p)==len(y) and np.isfinite(p).all()
            metrics={'AP_full':average_precision_score(y,p),'AP_U':average_precision_score(y[u],p[u])}
            order=u[np.lexsort((ids[u],-p[u]))]
            for pct in [5,10,15]:
                k=int(len(u)*pct/100);tp=y[order[:k]].sum()
                metrics['P_U_'+str(pct)]=tp/k;metrics['HR_U_'+str(pct)]=tp/y[u].sum()
            for metric,value in metrics.items():
                err=abs(value-rows[name][metric]);max_error=max(max_error,err);assert err<1e-12,(path,name,metric,err)
            checked+=1
arrays=0;files=0
for path in (ROOT/'results').rglob('*.npz'):
    with np.load(path) as z:
        for key in z.files:
            arr=z[key]
            if arr.dtype.kind in 'biufc':assert np.isfinite(arr).all(),(path,key)
            arrays+=1
    files+=1
out={'status':'PASS','audited_score_vectors':checked,'max_recomputed_metric_error':float(max_error),
     'all_npz_files_checked':files,'all_arrays_checked':arrays,'numeric_arrays_all_finite':True}
(ROOT/'results/additional_audits_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
