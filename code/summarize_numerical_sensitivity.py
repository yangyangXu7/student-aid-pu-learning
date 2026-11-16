"""Report all iteration-cap cases, including failures of the extended budget."""
import json
import numpy as np
import pandas as pd
from run_r2 import ROOT
from run_complete_r2 import extension_plan

rows=[];primary_flags=[]
for task in extension_plan():
    sc,sd=task['scenario'],task['seed'];path=ROOT/'results/completion'/sc/f'{sd}.json'
    original=json.loads(path.read_text());flagged=[m for m in ['SAR_EM','R1_IndexEM','LBE_LF'] if not original['method_info'][m]['converged']]
    if not flagged:continue
    extension=ROOT/'results/convergence_sensitivity'/sc/f'{sd}.json';new=json.loads(extension.read_text())
    primary={r['model']:r for r in original['rows']};extra={r['model']:r for r in new['rows']}
    z=np.load(extension.with_suffix('.npz'));assert np.isfinite(z['scores']).all()
    for m in flagged:
        d=new['method_info'][m]
        row={'scenario':sc,'seed':sd,'model':m,'primary_iterations':original['method_info'][m]['iterations'],
             'extended_iterations':d['iterations'],'extended_converged':d['converged'],
             'max_test_score_change':d['max_test_score_change'],'mean_abs_test_score_change':d['mean_abs_test_score_change']}
        for metric in ['AP_full','AP_U','HR_U_5','HR_U_10','HR_U_15']:
            row[metric+'_primary']=primary[m][metric];row[metric+'_extended']=extra[m][metric]
            row[metric+'_difference']=extra[m][metric]-primary[m][metric]
        rows.append(row)
out=ROOT/'results/tables';df=pd.DataFrame(rows)
if rows:
    df.to_csv(out/'convergence_sensitivity_raw.csv',index=False)
    summary=df.groupby('model').agg(n=('seed','count'),extended_converged_n=('extended_converged','sum'),
        AP_full_difference_mean=('AP_full_difference','mean'),AP_full_difference_absmax=('AP_full_difference',lambda x:np.abs(x).max()),
        AP_U_difference_mean=('AP_U_difference','mean'),AP_U_difference_absmax=('AP_U_difference',lambda x:np.abs(x).max()),
        HR_U_10_difference_mean=('HR_U_10_difference','mean'),HR_U_10_difference_absmax=('HR_U_10_difference',lambda x:np.abs(x).max())).reset_index()
else:summary=pd.DataFrame(columns=['model','n','extended_converged_n','AP_full_difference_mean','AP_full_difference_absmax'])
summary.to_csv(out/'convergence_sensitivity_summary.csv',index=False)
(ROOT/'results/convergence_sensitivity_verification.json').write_text(json.dumps({'status':'PASS','all_capped_primary_fits_checked':len(rows),
    'extended_still_capped':int((~df.extended_converged).sum()) if rows else 0,
    'policy':'No replacement or test-based selection of primary predictions.'},indent=2))
print(summary.to_string(index=False))
