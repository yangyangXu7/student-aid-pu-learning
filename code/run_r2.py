"""Run fixed, paired simulation cohorts. No test-based tuning.
Resume is supported only when code/config hashes match.
"""
import os
for _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_v]='1'
from pathlib import Path
import argparse,hashlib,json,time,platform,sys,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import generator_r2 as gen
from models_r2 import fit_model

ROOT=Path(__file__).resolve().parents[1]
GROUPS={'consumption_level':['monthly_total_spend','avg_meal_price','supermarket_spend'],
        'consumption_structure':['canteen_ratio','snack_drink_ratio','low_price_window_ratio'],
        'behavioral_regularity':['meals_per_day','breakfast_rate','weekend_on_campus'],
        'noncard_records':['library_visits','dorm_night_out','score_rank_pct']}
CODE_HASH=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')) if p.name in ['generator_r2.py','models_r2.py','run_r2.py','pu_original.py','synth_original.py'])).hexdigest()

def plan():
    tasks=[]
    def add(name,n=100,params=None,models=None,kind='hgb',drop=None,spy_frac=.15,quantile=5,group='main'):
        for sd in range(42,42+n):tasks.append(dict(scenario=name,seed=sd,params=params or {},models=models or ['Naive','CostSensitive','ElkanNoto','Spy','TrueLabel'],kind=kind,drop=drop or [],spy_frac=spy_frac,quantile=quantile,group=group))
    add('baseline')
    for b in [0.,.3,.62,.9]:
        add('beta_'+str(b),params=dict(label_intercept=.55+.55*b,label_slope=b,label_lo=.10,label_hi=.92),group='beta')
    for m in ['logistic','threshold','heterogeneous','reverse','nonmonotonic','unobserved']:
        add(m,params={'mechanism':m},group='mechanism')
    for param,vals in [('poor_rate',[.05,.25]),('frugal_rate',[0.,.25]),('overlap',[.5,1.]),('noise_scale',[1.5,2.]),('false_positive_rate',[.01,.03])]:
        for v in vals:add(param+'_'+str(v),n=50,params={param:v},group='stress')
    add('alternative_factors',params={'generator':'factors'},group='alternative')
    for g,drop in GROUPS.items():add('drop_'+g,models=['Spy'],drop=drop,group='ablation')
    add('logistic_learner',kind='logistic',group='base_learner')
    for f in [.05,.15,.30]:
        for q in [1,5,10]:
            if f==.15 and q==5:continue
            add('spy_f'+str(f)+'_q'+str(q),n=20,models=['Spy'],spy_frac=f,quantile=q,group='spy_grid')
    return tasks

def calc_metrics(y,s,scores,ids):
    hid=(y==1)&(s==0);u=s==0
    # R2 ties resolved by original generated student index, never by truth.
    order=np.lexsort((ids,-scores));uid=np.flatnonzero(u);uo=uid[np.lexsort((ids[u],-scores[u]))]
    row={'AP_full':float(average_precision_score(y,scores)),
         'AP_U':float(average_precision_score(y[u],scores[u])),
         'n_test':len(y),'n_U':int(u.sum()),'n_hidden':int(hid.sum()),
         'n_positive':int(y.sum()),'n_labeled_test':int(s.sum())}
    legacy=np.argsort(-scores)
    for pct in [5,10,15]:
        k=int(len(y)*pct/100);sel=order[:k]
        row.update({f'P_full_{pct}':float(y[sel].mean()) if k else None,
                    f'R_full_{pct}':float(y[sel].sum()/y.sum()) if y.sum() else None,
                    f'HR_full_{pct}':float(hid[sel].sum()/hid.sum()) if hid.sum() else None,
                    f'HR_legacy_{pct}':float(hid[legacy[:k]].sum()/hid.sum()) if hid.sum() else None})
    curve=[]
    for pct in range(1,31):
        k=int(u.sum()*pct/100);sel=uo[:k];tp=int(y[sel].sum())
        pr=tp/k if k else None;rec=tp/int(hid.sum()) if hid.sum() else None
        curve.append({'capacity_pct':pct,'k':k,'tp':tp,'P_U':pr,'HR_U':rec})
        if pct in [5,10,15]:row.update({f'k_U_{pct}':k,f'P_U_{pct}':pr,f'HR_U_{pct}':rec})
    return row,curve

def one(task):
    start=time.time();sc=task['scenario'];sd=task['seed'];folder=ROOT/'results'/'cohorts'/sc;folder.mkdir(parents=True,exist_ok=True)
    out=folder/f'{sd}.json';pred=folder/f'{sd}.npz'
    if out.exists() and pred.exists():
        old=json.loads(out.read_text())
        if old['code_hash']!=CODE_HASH or old['task']!=task:raise RuntimeError('Refusing stale cached result '+str(out))
        return sc,sd,'cached',old['seconds']
    df=gen.generate(seed=sd,**task['params']);features=[f for f in gen.FEATURES if f not in task['drop']]
    X=df[features].to_numpy();y=df.y_true.to_numpy();s=df.labeled.to_numpy()
    # Keep original stratification labels and split RNG exactly.
    strata=[f'{a}{b}' for a,b in zip(y,s)]
    tr,te=train_test_split(np.arange(len(y)),test_size=.30,random_state=7,stratify=strata)
    rows=[];curves=[];infos={};scores=[];names=[]
    for name in task['models']:
        labels=y[tr] if name=='TrueLabel' else s[tr]
        fitted,info=fit_model(X[tr],labels,name,kind=task['kind'],seed=0,
                              spy_frac=task['spy_frac'],quantile=task['quantile'],y_audit=y[tr])
        p=fitted(X[te]);r,c=calc_metrics(y[te],s[te],p,te)
        r['model']=name;rows.append(r)
        curves.extend(dict(model=name,**item) for item in c)
        infos[name]=info;scores.append(p);names.append(name)
    # A transparent simple ranking baseline needs no labels or fitted parameters.
    if not task['drop']:
        p=-X[te,features.index('monthly_total_spend')]
        r,c=calc_metrics(y[te],s[te],p,te);r['model']='LowSpending';rows.append(r)
        curves.extend(dict(model='LowSpending',**item) for item in c)
        scores.append(p);names.append('LowSpending')
    data_hash=hashlib.sha256(np.ascontiguousarray(df[gen.FEATURES+['y_true','labeled']].to_numpy()).tobytes()).hexdigest()
    np.savez_compressed(pred,ids=te,y=y[te],s=s[te],models=np.array(names),scores=np.array(scores),
                        q_individual=df.q_individual.to_numpy()[te])
    if sc=='baseline':
        np.savez_compressed(folder/f'{sd}_data.npz',X=X,y=y,s=s,train_ids=tr,test_ids=te,features=np.array(features))
    result={'task':task,'code_hash':CODE_HASH,'data_sha256':data_hash,'rows':rows,'curves':curves,'method_info':infos,
            'seconds':time.time()-start,'meta':{'n':len(y),'positive_n':int(y.sum()),'label_n':int(s.sum()),
                'c_true_train':float(s[tr][y[tr]==1].mean()),'label_purity_train':float(y[tr][s[tr]==1].mean()),
                'q_positive_mean':float(df.q_individual[y==1].mean())}}
    tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(out)
    return sc,sd,'done',result['seconds']

def main():
    a=argparse.ArgumentParser();a.add_argument('--suite',default='all');a.add_argument('--workers',type=int,default=6);a.add_argument('--max-seeds',type=int);args=a.parse_args()
    tasks=plan()
    if args.suite!='all':tasks=[t for t in tasks if t['scenario'] in args.suite.split(',') or t['group'] in args.suite.split(',')]
    if args.max_seeds:tasks=[t for t in tasks if t['seed']<42+args.max_seeds]
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/'plan.json').write_text(json.dumps(plan(),indent=2))
    start=time.time();print('START',len(tasks),'tasks','workers',args.workers,'code',CODE_HASH,flush=True)
    failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,t):t for t in tasks}
        for i,fut in enumerate(as_completed(futures),1):
            try:r=fut.result()
            except Exception:
                failures.append({'task':futures[fut],'traceback':traceback.format_exc()});print('FAILED',failures[-1],flush=True);continue
            if i%20==0 or i==len(tasks):print('PROGRESS',i,'/',len(tasks),'elapsed_s',round(time.time()-start),r,flush=True)
    if failures:
        (ROOT/'results'/('failures_'+args.suite.replace(',','_')+'.json')).write_text(json.dumps(failures,indent=2));raise RuntimeError(str(len(failures))+' failed')
    print('COMPLETE',len(tasks),'seconds',time.time()-start,flush=True)
if __name__=='__main__':main()
