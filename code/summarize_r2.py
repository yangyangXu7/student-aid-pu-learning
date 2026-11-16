"""Cohort-level descriptive and paired bootstrap inference.
Bootstrap units are cohorts, not students. Intervals are unadjusted percentile CIs.
"""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
import pandas as pd
from run_r2 import ROOT,plan
B=10000
METRICS=['AP_full','AP_U','P_U_5','P_U_10','P_U_15','HR_U_5','HR_U_10','HR_U_15','HR_full_5','HR_full_10','HR_full_15']

def bootstrap(values,key):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if len(values)<2:return np.nan,np.nan
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
    ix=np.random.default_rng(seed).integers(0,len(values),(B,len(values)))
    return np.quantile(values[ix].mean(axis=1),[.025,.975])

def summarize(allow_partial=False):
    out=ROOT/'results'/'tables';out.mkdir(exist_ok=True)
    jobs=plan();rows=[];curves=[];infos=[];seen=[];meta=[]
    for task in jobs:
        f=ROOT/'results/cohorts'/task['scenario']/f"{task['seed']}.json"
        if not f.exists():
            if allow_partial:continue
            raise RuntimeError('Missing result '+str(f))
        d=json.loads(f.read_text());base={k:task[k] for k in ['scenario','seed','kind','group']}
        rows.extend(dict(**base,**r) for r in d['rows']);curves.extend(dict(**base,**r) for r in d['curves'])
        for method,info in d['method_info'].items():infos.append(dict(**base,model=method,**info))
        meta.append(dict(**base,**d['meta'],data_sha256=d['data_sha256'],seconds=d['seconds']))
        seen.append((task['scenario'],task['seed']))
    df=pd.DataFrame(rows);cf=pd.DataFrame(curves);inf=pd.DataFrame(infos)
    df.to_csv(out/'cohort_metrics.csv',index=False);cf.to_csv(out/'capacity_raw.csv',index=False)
    inf.to_csv(out/'method_diagnostics_raw.csv',index=False);pd.DataFrame(meta).to_csv(out/'cohort_metadata.csv',index=False)
    ag=[]
    for (scenario,model),g in df.groupby(['scenario','model'],sort=True):
        for m in METRICS:
            x=g.sort_values('seed')[m].to_numpy();lo,hi=bootstrap(x,scenario+model+m+'mean')
            ag.append(dict(scenario=scenario,model=model,metric=m,n=len(x),mean=np.mean(x),sd=np.std(x,ddof=1),ci_low=lo,ci_high=hi))
    agg=pd.DataFrame(ag);agg.to_csv(out/'summary_metrics.csv',index=False)
    pairs=[]
    for scenario,g in df.groupby('scenario',sort=True):
        available=sorted(g.model.unique());contrasts=[]
        if 'Naive' in available:contrasts.extend((m,'Naive') for m in available if m!='Naive')
        if 'TrueLabel' in available:contrasts.extend((m,'TrueLabel') for m in available if m not in ['TrueLabel','Naive'])
        for a,b in contrasts:
            ga=g[g.model==a].set_index('seed');gb=g[g.model==b].set_index('seed');seeds=sorted(ga.index.intersection(gb.index))
            for metric in METRICS:
                x=(ga.loc[seeds,metric]-gb.loc[seeds,metric]).to_numpy();lo,hi=bootstrap(x,scenario+a+b+metric+'paired')
                pairs.append(dict(scenario=scenario,method_a=a,method_b=b,metric=metric,n=len(x),difference=x.mean(),sd=x.std(ddof=1),ci_low=lo,ci_high=hi))
    paired=pd.DataFrame(pairs);paired.to_csv(out/'paired_method_differences.csv',index=False)
    abl=[]
    baseline=df[(df.scenario=='baseline')&(df.model=='Spy')].set_index('seed')
    for scenario in sorted(x for x in df.scenario.unique() if x.startswith('drop_')):
        ab=df[(df.scenario==scenario)&(df.model=='Spy')].set_index('seed');seeds=sorted(baseline.index.intersection(ab.index))
        for metric in METRICS:
            x=(ab.loc[seeds,metric]-baseline.loc[seeds,metric]).to_numpy();lo,hi=bootstrap(x,scenario+metric+'ablation')
            abl.append(dict(dropped=scenario[5:],metric=metric,n=len(x),difference_drop_minus_full=x.mean(),sd=x.std(ddof=1),ci_low=lo,ci_high=hi))
    pd.DataFrame(abl).to_csv(out/'paired_ablation.csv',index=False)
    diag=[]
    for (scenario,model),g in inf.groupby(['scenario','model']):
        for metric in ['rn_retained_fraction','rn_purity','hidden_in_rn_fraction','n_reliable_neg','spy_threshold','fallback','c_hat']:
            if metric not in g:continue
            x=pd.to_numeric(g[metric].dropna()).to_numpy(float)
            if not len(x):continue
            lo,hi=bootstrap(x,scenario+model+metric+'diagnostic')
            diag.append(dict(scenario=scenario,model=model,metric=metric,n=len(x),mean=x.mean(),sd=x.std(ddof=1),ci_low=lo,ci_high=hi))
    pd.DataFrame(diag).to_csv(out/'method_diagnostics_summary.csv',index=False)
    cap=cf.groupby(['scenario','model','capacity_pct']).agg(n=('HR_U','count'),HR_mean=('HR_U','mean'),HR_sd=('HR_U','std'),P_mean=('P_U','mean'),P_sd=('P_U','std'),k_mean=('k','mean')).reset_index()
    # Pointwise bootstrap bands only for baseline capacity figure.
    for idx,r in cap[cap.scenario=='baseline'].iterrows():
        x=cf[(cf.scenario=='baseline')&(cf.model==r.model)&(cf.capacity_pct==r.capacity_pct)].sort_values('seed').HR_U.to_numpy()
        cap.loc[idx,['HR_ci_low','HR_ci_high']]=bootstrap(x,'capacity'+r.model+str(r.capacity_pct))
    cap.to_csv(out/'capacity_summary.csv',index=False)
    scenario_counts=df.groupby('scenario').seed.nunique().to_dict()
    (out/'run_counts.json').write_text(json.dumps({'cohort_scenario_jobs_completed':len(seen),'jobs_planned':len(jobs),'unique_seeds':int(df.seed.nunique()),'counts':scenario_counts,'bootstrap_samples':B,'bootstrap_unit':'cohort','intervals':'unadjusted percentile 95%','paired_by':'identical scenario and cohort seed'},indent=2))
    # Paired change in Spy advantage under reverse versus original labeling.
    if 'reverse' in df.scenario.unique():
        delta=[]
        for metric in METRICS:
            gg=df[(df.scenario.isin(['baseline','reverse']))&(df.model.isin(['Spy','Naive']))].pivot(index='seed',columns=['scenario','model'],values=metric).dropna()
            x=(gg['reverse','Spy']-gg['reverse','Naive'])-(gg['baseline','Spy']-gg['baseline','Naive'])
            lo,hi=bootstrap(x,'reverse_interaction'+metric)
            delta.append(dict(metric=metric,n=len(x),difference_in_advantage=x.mean(),ci_low=lo,ci_high=hi))
        pd.DataFrame(delta).to_csv(out/'reverse_change_in_advantage.csv',index=False)
    cross=[]
    if 'logistic_learner' in df.scenario.unique():
        for method_a,method_b in [('Naive','Naive'),('Spy','Spy'),('Naive','Spy'),('TrueLabel','TrueLabel')]:
            a=df[(df.scenario=='logistic_learner')&(df.model==method_a)].set_index('seed')
            b=df[(df.scenario=='baseline')&(df.model==method_b)].set_index('seed')
            seeds=sorted(a.index.intersection(b.index))
            for metric in METRICS:
                x=a.loc[seeds,metric]-b.loc[seeds,metric]
                lo,hi=bootstrap(x,'crosslearner'+method_a+method_b+metric)
                cross.append(dict(method_a='Logistic '+method_a,method_b='HGB '+method_b,metric=metric,n=len(x),difference=x.mean(),ci_low=lo,ci_high=hi))
        pd.DataFrame(cross).to_csv(out/'cross_learner_paired.csv',index=False)
    print('Summarized',len(seen),'jobs',len(df),'model/cohort rows',flush=True)


def summarize_audit():
    files=sorted((ROOT/'results/propensity_audit').glob('*.json'))
    if not files:return
    data=[json.loads(f.read_text()) for f in files];rows=[dict(seed=d['seed'],**r) for d in data for r in d['rows']]
    df=pd.DataFrame(rows);out=ROOT/'results/tables';out.mkdir(exist_ok=True)
    df.to_csv(out/'propensity_audit_raw.csv',index=False)
    pd.DataFrame([{k:v for k,v in d.items() if k not in ['rows','curves']} for d in data]).to_csv(out/'propensity_numerator_diagnostics.csv',index=False)
    ag=[];pairs=[]
    for method,g in df.groupby('model'):
        for metric in METRICS:
            x=g.sort_values('seed')[metric].to_numpy();lo,hi=bootstrap(x,'audit'+method+metric)
            ag.append(dict(model=method,metric=metric,n=len(x),mean=x.mean(),sd=x.std(ddof=1),ci_low=lo,ci_high=hi))
            ref=df[df.model=='NaiveFullTrain'].set_index('seed');v=g.set_index('seed');seeds=sorted(ref.index.intersection(v.index));diff=v.loc[seeds,metric]-ref.loc[seeds,metric]
            lo,hi=bootstrap(diff,'auditdiff'+method+metric)
            pairs.append(dict(model=method,reference='NaiveFullTrain',metric=metric,n=len(diff),difference=diff.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(ag).to_csv(out/'propensity_audit_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'propensity_audit_paired.csv',index=False)
    print('Summarized propensity audit',len(data),'cohorts',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if not a.audit_only:summarize(a.partial)
    summarize_audit()
