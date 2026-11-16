"""Same-objective optimizer audit for every LBE primary iteration-cap case.
This is supplementary; no primary score is replaced or test-selected.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.special import expit,log_expit
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from run_r2 import ROOT,calc_metrics
from generator_r2 import generate,FEATURES
from models_propensity_r2 import design

def objective(parameters,A,s):
    k=A.shape[1];za=A@parameters[:k];zb=A@parameters[k:]
    f=expit(za);e=expit(zb)
    l0=np.logaddexp(log_expit(-za),log_expit(za)+log_expit(-zb))
    l1=log_expit(za)+log_expit(zb)
    r=s+(1-s)*np.exp(log_expit(za)+log_expit(-zb)-l0)
    loss=-np.mean(s*l1+(1-s)*l0)
    grad=np.r_[A.T@(f-r)/len(s),A.T@(r*e-s)/len(s)]
    return float(loss),grad

def one(path):
    original=json.loads(path.read_text());info=original['method_info']['LBE_LF']
    if info['converged']:return None
    task=original['task'];sc,sd=task['scenario'],task['seed']
    folder=ROOT/'results'/'lbe_optimizer_audit'/sc;folder.mkdir(parents=True,exist_ok=True)
    out=folder/f'{sd}.json'
    if out.exists():return json.loads(out.read_text())
    start=time.time();df=generate(seed=sd,**task['params']);X=df[FEATURES].to_numpy();y=df.y_true.to_numpy();s=df.labeled.to_numpy()
    tr,te=train_test_split(np.arange(len(y)),test_size=.3,random_state=7,stratify=[f'{a}{b}' for a,b in zip(y,s)])
    mean=np.array(info['scaler_mean']);scale=np.array(info['scaler_scale']);A=design((X[tr]-mean)/scale)
    init=np.r_[info['class_coef'],info['propensity_coef']]
    result=minimize(objective,init,args=(A,s[tr]),jac=True,method='L-BFGS-B',
       options={'maxiter':5000,'ftol':1e-14,'gtol':1e-9,'maxls':50,'maxcor':30})
    p=expit(design((X[te]-mean)/scale)@result.x[:A.shape[1]])
    row,_=calc_metrics(y[te],s[te],p,te);oldrow=next(r for r in original['rows'] if r['model']=='LBE_LF')
    oldpred=np.load(path.with_suffix('.npz'));oldscore=oldpred['scores'][oldpred['models'].tolist().index('LBE_LF')]
    d={'scenario':sc,'seed':sd,'model':'LBE_LF','optimizer':'same unpenalized likelihood, L-BFGS-B from primary Adam solution',
       'iterations':int(result.nit),'optimizer_success':bool(result.success),'message':str(result.message),
       'gradient_inf':float(np.max(np.abs(result.jac))),'stationarity_1e_6':bool(np.max(np.abs(result.jac))<1e-6),
       'initial_negative_ll':objective(init,A,s[tr])[0],'final_negative_ll':float(result.fun),
       'max_abs_coefficient':float(np.max(np.abs(result.x))),'max_test_score_change':float(np.max(np.abs(p-oldscore))),
       'rows':[dict(model='LBE_LF_LBFGS_audit',**row)],'seconds':time.time()-start}
    for metric in ['AP_full','AP_U','HR_U_5','HR_U_10','HR_U_15']:
        d[metric+'_difference']=row[metric]-oldrow[metric]
    assert np.isfinite(p).all() and result.fun<=d['initial_negative_ll']+1e-9
    np.savez_compressed(out.with_suffix('.npz'),ids=te,y=y[te],s=s[te],scores=p,parameters=result.x)
    out.write_text(json.dumps(d,indent=2,allow_nan=False));print(sc,sd,'gradient',d['gradient_inf'],'AP_change',d['AP_full_difference'],flush=True)
    return d

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    files=list((ROOT/'results'/'completion').glob('*/*.json'))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=[r for r in pool.map(one,files) if r is not None]
    import pandas as pd
    rows=[{k:v for k,v in d.items() if k!='rows'} for d in results]
    f=pd.DataFrame(rows);f.to_csv(ROOT/'results/tables/lbe_optimizer_audit.csv',index=False)
    print('Complete optimizer audit for currently available primary cap cases:',len(rows),flush=True)
