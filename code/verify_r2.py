"""Meaningful numerical and protocol checks; exits nonzero on any failure."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import generator_r2 as gen
from run_r2 import calc_metrics,ROOT
from propensity_audit import known_conditionals

def main():
    checks=[]
    # Independently check the candidate population, denominators, tie handling and floor.
    y=np.array([1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1])
    s=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);ids=np.arange(len(y))[::-1]
    scores=np.ones(len(y));r,c=calc_metrics(y,s,scores,ids)
    assert r['n_U']==20 and r['n_hidden']==5 and r['k_U_5']==1 and r['k_U_10']==2
    # With ties the smallest original ID is selected first, i.e. last row is positive.
    assert r['P_U_5']==1 and r['HR_U_5']==.2
    assert r['P_U_10']==.5 and r['HR_U_10']==.2
    perm=np.random.default_rng(7).permutation(len(y))
    r2,c2=calc_metrics(y[perm],s[perm],scores[perm],ids[perm])
    for key in r:
        if not key.startswith('HR_legacy'):assert r[key]==r2[key],key
    checks.append('U-only population, floor, hidden denominator, original-ID tie order and permutation invariance')
    tiny_y=np.array([0,1]);tiny_s=np.array([0,0]);r,c=calc_metrics(tiny_y,tiny_s,np.array([.1,.9]),np.arange(2))
    assert r['k_U_5']==0 and r['P_U_5'] is None and r['HR_U_5']==0
    checks.append('Zero shortlist precision undefined; recall zero with positive denominator')
    for m in ['linear_inverse','scar','logistic','threshold','heterogeneous','reverse','nonmonotonic','unobserved']:
        d=gen.generate(seed=42,mechanism=m)
        assert np.isfinite(d[gen.FEATURES]).all().all()
        assert ((d.q_individual>=0)&(d.q_individual<=1)).all()
        assert not ((d.y_true==0)&(d.labeled==1)).any()
    checks.append('All clean mechanisms: finite features, valid propensity and one-sided labels')
    # True posterior calculation independently compared with high-resolution quadrature.
    z=np.load(ROOT/'results'/'cohorts'/'baseline'/'42_data.npz');X=z['X'];feats=z['features'].tolist()
    eta,e,g,eta_u=known_conditionals(X[:300],feats,64)
    eta2,e2,_,_=known_conditionals(X[:300],feats,128)
    np.testing.assert_allclose(eta,eta2,rtol=1e-8,atol=1e-10);np.testing.assert_allclose(e,e2,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(g/e,eta,rtol=1e-12,atol=1e-12)
    assert (eta_u<=eta+1e-12).all()
    checks.append('Conditional propensity integral convergence and population identity')
    # Recompute saved metrics from saved arrays (selected models across completed scenarios).
    nfiles=0
    for folder in sorted((ROOT/'results'/'cohorts').iterdir()):
        for seed in [42,91,141]:
            f=folder/f'{seed}.json';npz=folder/f'{seed}.npz'
            if not f.exists():continue
            d=json.loads(f.read_text());z=np.load(npz);names=z['models'].tolist()
            for row in d['rows']:
                r,_=calc_metrics(z['y'],z['s'],z['scores'][names.index(row['model'])],z['ids'])
                for k,v in r.items():
                    if v is None:assert row[k] is None
                    else:assert np.isclose(row[k],v,rtol=0,atol=1e-12),(folder,seed,k)
            nfiles+=1
    checks.append(f'Recomputed all metrics from stored predictions for {nfiles} cohort/scenario files')
    # Original seed 42 input results remain reproducible to uploaded rounded precision.
    b=json.loads((ROOT/'results/cohorts/baseline/42.json').read_text())
    mapping={'Naive':'Naive supervised (U=N)','CostSensitive':'Cost-sensitive','Spy':'Spy two-step (PU)',
             'ElkanNoto':'Elkan-Noto weighted (PU)','TrueLabel':'Oracle (true labels)'}
    source=pd.read_csv(ROOT/'original/metrics.csv').set_index('model')
    for r in b['rows']:
        if r['model'] in mapping:
            assert round(r['AP_full'],4)==source.loc[mapping[r['model']],'PR-AUC']
            for pct in [5,10,15]:assert round(r[f'HR_legacy_{pct}'],4)==source.loc[mapping[r['model']],f'HiddenR@{pct}%']
    checks.append('Original supplied metrics.csv reproduced at its four-decimal precision')
    # Published old aggregate tables are checked against the SAME old seed subsets.
    old=pd.read_csv(ROOT/'original/multi_seed_agg.csv',header=[0,1],index_col=0)
    base=[]
    for seed in range(42,52):base.extend(dict(seed=seed,**r) for r in json.loads((ROOT/f'results/cohorts/baseline/{seed}.json').read_text())['rows'])
    base=pd.DataFrame(base)
    short={'Naive':'Naive','CostSensitive':'CostSensitive','ElkanNoto':'ElkanNoto','Spy':'Spy','TrueLabel':'Oracle'}
    for model,oldname in short.items():
        x=base[base.model==model].AP_full
        assert round(x.mean(),4)==old.loc[oldname,('AP','mean')]
        assert round(x.std(ddof=1),4)==old.loc[oldname,('AP','std')]
    checks.append('All original five-method 10-seed AP means and SDs match uploaded aggregate')
    sw=pd.read_csv(ROOT/'original/sweep_agg.csv')
    for beta in [0.,.3,.62,.9]:
        rr=[]
        for seed in range(42,47):rr.extend(json.loads((ROOT/f'results/cohorts/beta_{beta}/{seed}.json').read_text())['rows'])
        dd=pd.DataFrame(rr)
        for model,oldname in short.items():
            if model=='CostSensitive':continue
            x=dd[dd.model==model];o=sw[(sw.beta==beta)&(sw.model==oldname)].iloc[0]
            for a,b in [('AP_full','AP_m'),('HR_legacy_10','HR_m')]:assert round(x[a].mean(),4)==o[b]
    checks.append('All original four-slope, four-method 5-seed AP and hidden-recall means match uploaded sweep')
    oldab=pd.read_csv(ROOT/'original/ablation_agg.csv').set_index('dropped')
    maps={'full model':'baseline','consumption level':'drop_consumption_level','consumption structure':'drop_consumption_structure','behavioral regularity':'drop_behavioral_regularity','non card records':'drop_noncard_records'}
    for name,scenario in maps.items():
        rr=[]
        for seed in range(42,47):rr.extend(json.loads((ROOT/f'results/cohorts/{scenario}/{seed}.json').read_text())['rows'])
        dd=pd.DataFrame(rr);x=dd[dd.model=='Spy']
        for a,b in [('AP_full','AP_m'),('HR_legacy_10','HR_m')]:assert round(x[a].mean(),4)==oldab.loc[name,b]
    checks.append('All original 5-seed feature-ablation AP and hidden-recall means match uploaded aggregate')
    # Re-generate every scenario's first cohort without fitting and compare exact bytes.
    import hashlib
    from run_r2 import plan
    tasks={t['scenario']:t for t in reversed(plan())}
    for scenario,t in tasks.items():
        df=gen.generate(seed=t['seed'],**t['params'])
        digest=hashlib.sha256(np.ascontiguousarray(df[gen.FEATURES+['y_true','labeled']].to_numpy()).tobytes()).hexdigest()
        record=json.loads((ROOT/f'results/cohorts/{scenario}/{t["seed"]}.json').read_text())
        assert digest==record['data_sha256'],scenario
    checks.append(f'Every one of {len(tasks)} scenarios regenerated at its first seed with exact SHA256 match')
    result={'status':'PASS','checks':checks}
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
