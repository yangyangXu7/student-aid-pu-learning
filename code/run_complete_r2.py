"""Complete the expanded benchmark on the existing, exactly paired cohorts."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from sklearn.model_selection import train_test_split
from models_propensity_r2 import fit_propensity,CONFIG,PROPENSITY_FEATURES,HARDSHIP_SIGNS
from models_r2 import fit_model
from run_r2 import ROOT,plan,calc_metrics
import generator_r2 as gen

CODE_HASH=hashlib.sha256(b''.join((ROOT/'code'/n).read_bytes() for n in
 ['run_complete_r2.py','models_propensity_r2.py','generator_r2.py','models_r2.py','run_r2.py','pu_original.py'])).hexdigest()

def extension_plan():
    return [t for t in plan() if t['group'] in ['main','beta','mechanism','stress','alternative']]

def one(task):
    start=time.time();sc=task['scenario'];seed=task['seed']
    folder=ROOT/'results'/'completion'/sc;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{seed}.json';pred=folder/f'{seed}.npz'
    if path.exists() and pred.exists():
        old=json.loads(path.read_text())
        if old['code_hash']!=CODE_HASH or old['task']!=task:raise RuntimeError('Stale extension '+str(path))
        return sc,seed,'cached',old['seconds']
    original=json.loads((ROOT/'results'/'cohorts'/sc/f'{seed}.json').read_text())
    original_pred=np.load(ROOT/'results'/'cohorts'/sc/f'{seed}.npz')
    df=gen.generate(seed=seed,**task['params']);features=gen.FEATURES
    dh=hashlib.sha256(np.ascontiguousarray(df[features+['y_true','labeled']].to_numpy()).tobytes()).hexdigest()
    assert dh==original['data_sha256'],'cohort hash mismatch'
    X=df[features].to_numpy();y=df.y_true.to_numpy();s=df.labeled.to_numpy()
    tr,te=train_test_split(np.arange(len(y)),test_size=.3,random_state=7,stratify=[f'{a}{b}' for a,b in zip(y,s)])
    assert np.array_equal(te,original_pred['ids']) and np.array_equal(y[te],original_pred['y'])
    rows=[];curves=[];scores=[];names=[];infos={};props=[];propnames=[]
    def record(name,p,info):
        r,c=calc_metrics(y[te],s[te],p,te);r['model']=name;rows.append(r)
        curves.extend(dict(model=name,**v) for v in c);scores.append(p);names.append(name);infos[name]=info
    for name in ['Naive','CostSensitive','ElkanNoto','Spy','TrueLabel']:
        f,info=fit_model(X[tr],y[tr] if name=='TrueLabel' else s[tr],name,kind='logistic',seed=0,y_audit=y[tr])
        record(name+'_LR',f(X[te]),info)
    for name in ['SAR_EM','R1_IndexEM','LBE_LF']:
        f,info=fit_propensity(X[tr],s[tr],features,name)
        p,e=f(X[te]);record(name,p,info);props.append(e);propnames.append(name)
        # Separate supplementary score for filtering an already-unlabeled pool.
        u_score=p*(1-e)/np.maximum(1-p*e,1e-15)
        record(name+'_UPosterior',u_score,{'derived_from':name,'formula':'f*(1-e)/(1-f*e)'})
    np.savez_compressed(pred,ids=te,y=y[te],s=s[te],models=np.array(names),scores=np.array(scores),
        propensity_models=np.array(propnames),propensities=np.array(props),
        college=df.college.to_numpy()[te],frugal=df.frugal.to_numpy()[te])
    result={'task':task,'code_hash':CODE_HASH,'original_code_hash':original['code_hash'],
        'data_sha256':dh,'rows':rows,'curves':curves,'method_info':infos,'seconds':time.time()-start}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(path)
    return sc,seed,'done',round(result['seconds'],2)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--suite',default='all');parser.add_argument('--max-seeds',type=int)
    a=parser.parse_args();tasks=extension_plan()
    if a.suite!='all':tasks=[t for t in tasks if t['scenario'] in a.suite.split(',') or t['group'] in a.suite.split(',')]
    if a.max_seeds:tasks=[t for t in tasks if t['seed']<42+a.max_seeds]
    (ROOT/'results'/'completion_plan.json').write_text(json.dumps({'tasks':extension_plan(),'code_hash':CODE_HASH,
        'method_config':CONFIG,'propensity_features':PROPENSITY_FEATURES,'index_signs':HARDSHIP_SIGNS.tolist()},indent=2))
    start=time.time();fail=[];print('START',len(tasks),'cohorts',a.workers,'workers',CODE_HASH,flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(one,t):t for t in tasks}
        for i,f in enumerate(as_completed(pending),1):
            try:r=f.result()
            except Exception:
                fail.append({'task':pending[f],'traceback':traceback.format_exc()});print('FAILED',fail[-1],flush=True);continue
            if i%10==0 or i==len(tasks):print('PROGRESS',i,'/',len(tasks),'elapsed',round(time.time()-start),r,flush=True)
    if fail:
        (ROOT/'results'/'completion_failures.json').write_text(json.dumps(fail,indent=2));raise RuntimeError(f'{len(fail)} failed')
    print('COMPLETE',len(tasks),'seconds',round(time.time()-start),flush=True)
