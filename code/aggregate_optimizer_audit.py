"""Cohort-paired sensitivity to optimizing the same LBE objective after a cap.
Selection into this audit uses training convergence only, never test scores.
"""
import json
import numpy as np
import pandas as pd
from run_r2 import ROOT
from run_complete_r2 import extension_plan
from summarize_r2 import bootstrap

records=[]
for task in extension_plan():
    sc,sd=task['scenario'],task['seed'];path=ROOT/'results/completion'/sc/f'{sd}.json'
    d=json.loads(path.read_text());r=next(x for x in d['rows'] if x['model']=='LBE_LF');capped=not d['method_info']['LBE_LF']['converged']
    if capped:
        a=json.loads((ROOT/'results/lbe_optimizer_audit'/sc/f'{sd}.json').read_text());rr=a['rows'][0]
    else:a={};rr=r
    for metric in ['AP_full','AP_U','HR_U_5','HR_U_10','HR_U_15']:
        records.append({'scenario':sc,'seed':sd,'metric':metric,'primary_capped':capped,
          'primary':r[metric],'same_objective_policy':rr[metric],'difference':rr[metric]-r[metric],
          'polish_stationary':a.get('stationarity_1e_6',True)})
df=pd.DataFrame(records);df.to_csv(ROOT/'results/tables/lbe_optimizer_policy_raw.csv',index=False);rows=[]
for (sc,metric),g in df.groupby(['scenario','metric']):
    lo,hi=bootstrap(g.sort_values('seed').difference.to_numpy(),'optimizer_policy:'+sc+metric)
    rows.append({'scenario':sc,'metric':metric,'n':len(g),'primary_capped_n':int(g.primary_capped.sum()),
       'primary_mean':g.primary.mean(),'same_objective_policy_mean':g.same_objective_policy.mean(),
       'difference':g.difference.mean(),'ci_low':lo,'ci_high':hi,'not_stationary_n':int((~g.polish_stationary).sum())})
pd.DataFrame(rows).to_csv(ROOT/'results/tables/lbe_optimizer_policy_summary.csv',index=False)
print(pd.DataFrame(rows).query('metric=="AP_full" and primary_capped_n>0').to_string(index=False))
