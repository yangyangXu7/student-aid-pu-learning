"""Higher iteration budgets for ALL fits that hit their fixed primary cap.
The primary benchmark is not overwritten or selected using test performance.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from generator_r2 import generate,FEATURES
from models_propensity_r2 import fit_propensity
from run_r2 import ROOT,calc_metrics
from run_complete_r2 import extension_plan

def run_one(path):
    d=json.loads(path.read_text());task=d['task'];sd=task['seed'];sc=task['scenario']
    flagged=[m for m in ['SAR_EM','R1_IndexEM','LBE_LF'] if not d['method_info'][m]['converged']]
    if not flagged:return 0
    dest=ROOT/'results'/'convergence_sensitivity'/sc;dest.mkdir(parents=True,exist_ok=True)
    output=dest/f'{sd}.json'
    if output.exists():return 0
    df=generate(seed=sd,**task['params']);X=df[FEATURES].to_numpy();y=df.y_true.to_numpy();s=df.labeled.to_numpy()
    tr,te=train_test_split(np.arange(len(y)),test_size=.3,random_state=7,stratify=[f'{a}{b}' for a,b in zip(y,s)])
    old=np.load(path.with_suffix('.npz'));assert np.array_equal(old['ids'],te)
    rows=[];info={};scores=[];names=[]
    for name in flagged:
        start=time.time();f,di=fit_propensity(X[tr],s[tr],FEATURES,name,
          overrides={'max_iter':60000 if name=='LBE_LF' else 2000})
        pp,ee=f(X[te]);r,_=calc_metrics(y[te],s[te],pp,te);r['model']=name;rows.append(r)
        original=old['scores'][old['models'].tolist().index(name)]
        di['max_test_score_change']=float(np.max(np.abs(pp-original)))
        di['mean_abs_test_score_change']=float(np.mean(np.abs(pp-original)))
        di['primary_iterations']=d['method_info'][name]['iterations'];di['seconds']=time.time()-start
        info[name]=di;scores.append(pp);names.append(name)
    np.savez_compressed(output.with_suffix('.npz'),ids=te,y=y[te],s=s[te],models=np.array(names),scores=np.array(scores))
    output.write_text(json.dumps({'task':task,'rows':rows,'method_info':info,'primary_code_hash':d['code_hash']},indent=2,allow_nan=False))
    print('EXTENDED',sc,sd,[(m,info[m]['iterations'],info[m]['converged']) for m in flagged],flush=True)
    return len(flagged)

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor,as_completed
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    plan=extension_plan();done=set();n=0
    while True:
        files=list((ROOT/'results'/'completion').glob('*/*.json'))
        pending=[path for path in files if path not in done]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(run_one,path):path for path in pending}
            for f in as_completed(futures):n+=f.result();done.add(futures[f])
        if not a.watch or len(done)==len(plan):break
        time.sleep(10)
    print('COMPLETE sensitivity; newly extended fits',n,'primary cohorts examined',len(done),flush=True)
