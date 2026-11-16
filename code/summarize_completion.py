"""Combined, paired cohort inference for the completed expanded benchmark."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from run_r2 import ROOT
from run_complete_r2 import extension_plan
from summarize_r2 import METRICS,bootstrap

def weights(n,key):
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
    return np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=10000).astype(float)/n

CONTRASTS=[('Spy','Naive'),('Spy','TrueLabel'),('Spy_LR','Naive_LR'),
 ('CostSensitive_LR','Naive_LR'),('ElkanNoto_LR','Naive_LR'),('Naive_LR','Spy'),
 ('TrueLabel_LR','TrueLabel'),('R1_IndexEM','LBE_LF'),('SAR_EM','LBE_LF')]
for _a in ['SAR_EM','R1_IndexEM','LBE_LF']:
    CONTRASTS.extend((_a,b) for b in ['Naive','Spy','TrueLabel','Naive_LR','Spy_LR','TrueLabel_LR'])
    CONTRASTS.append((_a+'_UPosterior',_a))

def summarize(baseline_only=False):
    rows=[];curves=[];diagnostics=[];tasks=extension_plan()
    if baseline_only:tasks=[t for t in tasks if t['scenario']=='baseline']
    for t in tasks:
        for version in ['cohorts','completion']:
            path=ROOT/'results'/version/t['scenario']/f"{t['seed']}.json"
            d=json.loads(path.read_text());base={'scenario':t['scenario'],'seed':t['seed'],'group':t['group']}
            rows.extend(dict(**base,**r) for r in d['rows'])
            curves.extend(dict(**base,**c) for c in d['curves'])
            if version=='completion':
                for m in ['SAR_EM','R1_IndexEM','LBE_LF']:
                    info=d['method_info'][m]
                    diagnostics.append(dict(**base,model=m,iterations=info['iterations'],converged=info['converged'],
                        final_observed_ll=info['final_observed_ll'],**info['stop']))
    df=pd.DataFrame(rows);out=ROOT/'results'/'tables'
    ag=[];pairs=[]
    for sc,g in df.groupby('scenario',sort=True):
        wide=g.pivot(index='seed',columns='model',values=METRICS).sort_index()
        assert not wide.isna().any().any()
        mat=wide.to_numpy();n=len(wide);ww=weights(n,'completed_benchmark:'+sc)
        boot=ww@mat;lo,hi=np.quantile(boot,[.025,.975],axis=0)
        means=mat.mean(axis=0);sd=mat.std(axis=0,ddof=1)
        cols=wide.columns.tolist()
        for j,(metric,m) in enumerate(cols):
            ag.append(dict(scenario=sc,model=m,metric=metric,n=n,mean=means[j],sd=sd[j],ci_low=lo[j],ci_high=hi[j]))
        for a,b in CONTRASTS:
            for metric in METRICS:
                ia,ib=cols.index((metric,a)),cols.index((metric,b))
                diff=mat[:,ia]-mat[:,ib];bd=boot[:,ia]-boot[:,ib];q=np.quantile(bd,[.025,.975])
                pairs.append(dict(scenario=sc,method_a=a,method_b=b,metric=metric,n=n,
                    difference=diff.mean(),sd=diff.std(ddof=1),ci_low=q[0],ci_high=q[1]))
    tag='baseline_' if baseline_only else ''
    df.to_csv(out/(tag+'complete_cohort_metrics.csv'),index=False)
    pd.DataFrame(ag).to_csv(out/(tag+'complete_summary_metrics.csv'),index=False)
    pd.DataFrame(pairs).to_csv(out/(tag+'complete_paired_differences.csv'),index=False)
    dd=pd.DataFrame(diagnostics);dd.to_csv(out/(tag+'em_convergence_raw.csv'),index=False)
    ds=dd.groupby(['scenario','model']).agg(n=('seed','count'),converged_n=('converged','sum'),
        iterations_mean=('iterations','mean'),iterations_max=('iterations','max')).reset_index()
    ds.to_csv(out/(tag+'em_convergence_summary.csv'),index=False)
    cap=pd.DataFrame(curves);cs=cap.groupby(['scenario','model','capacity_pct']).agg(n=('HR_U','count'),
        HR_mean=('HR_U','mean'),HR_sd=('HR_U','std'),P_mean=('P_U','mean'),P_sd=('P_U','std'),k_mean=('k','mean')).reset_index()
    cg=cap[cap.scenario=='baseline'];cw=weights(100,'completed_capacity:baseline')
    for m,g in cg.groupby('model'):
        wide=g.pivot(index='seed',columns='capacity_pct',values='HR_U').sort_index();bb=cw@wide.to_numpy()
        lo,hi=np.quantile(bb,[.025,.975],axis=0)
        for j,pct in enumerate(wide.columns):
            ix=(cs.scenario=='baseline')&(cs.model==m)&(cs.capacity_pct==pct)
            cs.loc[ix,'HR_ci_low']=lo[j];cs.loc[ix,'HR_ci_high']=hi[j]
    cs.to_csv(out/(tag+'complete_capacity_summary.csv'),index=False)
    print('Combined benchmark',len(tasks),'cohort settings;',len(df),'score/cohort rows;',len(ag),'summaries;',len(pairs),'paired contrasts.',flush=True)

def extra_audits():
    out=ROOT/'results'/'tables';rows=[]
    for p in sorted((ROOT/'results'/'denominator_audit').glob('*.json')):
        d=json.loads(p.read_text());rows.extend(dict(seed=d['seed'],**r) for r in d['rows'])
    df=pd.DataFrame(rows);df.to_csv(out/'denominator_audit_raw.csv',index=False);summ=[];paired=[]
    for m,g in df.groupby('model'):
        for metric in METRICS:
            v=g.sort_values('seed')[metric].to_numpy();lo,hi=bootstrap(v,'denominator:'+m+metric)
            summ.append(dict(model=m,metric=metric,n=len(v),mean=v.mean(),sd=v.std(ddof=1),ci_low=lo,ci_high=hi))
            ref=df[df.model=='Raw_div_conditional_eX'].sort_values('seed')[metric].to_numpy();diff=v-ref
            lo,hi=bootstrap(diff,'denominator_pair:'+m+metric)
            paired.append(dict(model=m,reference='Raw_div_conditional_eX',metric=metric,n=len(v),
                difference=diff.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(summ).to_csv(out/'denominator_audit_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'denominator_audit_paired.csv',index=False)
    f=pd.read_csv(out/'prospective_sampling_simulation_raw.csv');s=[]
    for size,g in f.groupby('nonselected_sample'):
        for metric in ['true_missed_positives','missed_hat_bias','missed_hat_rmse','recall_hat_bias',
                       'recall_hat_rmse','probability_observe_no_missed_exact']:
            v=g.sort_values('seed')[metric].to_numpy();lo,hi=bootstrap(v,'prospective:'+str(size)+metric)
            s.append(dict(nonselected_sample=size,metric=metric,n=len(v),mean=v.mean(),sd=v.std(ddof=1),ci_low=lo,ci_high=hi))
    pd.DataFrame(s).to_csv(out/'prospective_sampling_simulation_summary.csv',index=False)
    print('Summarized denominator and audit-design checks.',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--baseline-only',action='store_true');a.add_argument('--audits-only',action='store_true');opt=a.parse_args()
    if not opt.audits_only:summarize(opt.baseline_only)
    extra_audits()
