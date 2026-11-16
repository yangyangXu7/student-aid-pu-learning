"""NEW audit for the supplied original simulator, not reconstruction of R1 code.
Compute E[q(T)|Y=1,X] by Gauss-Jacobi quadrature. Account for clipping mass.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,argparse
import numpy as np
from scipy.special import roots_jacobi,logsumexp,expit,logit
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,average_precision_score
import generator_r2 as gen
from models_r2 import fit_model
from run_r2 import calc_metrics,ROOT

def density(x,mu,sd,lo,hi):
    # Interior density versus point mass at a clipping bound.
    z=(x-mu)/sd
    ll=norm.logpdf(z)-np.log(sd)
    if lo is not None:ll=np.where(x<=lo,norm.logcdf((lo-mu)/sd),ll)
    if hi is not None:ll=np.where(x>=hi,norm.logsf((hi-mu)/sd),ll)
    return ll

def known_conditionals(X,feats,quadrature=64,prior=.13,frugal_rate=.1):
    nodes,w=roots_jacobi(quadrature,.8,1.2);t=(nodes+1)/2;w=w/w.sum()
    lp=np.zeros((len(X),len(t)));lr=np.zeros(len(X));lf=np.zeros(len(X))
    for name,pt,pa,rich,frug,sd,lo,hi in gen.SPEC:
        x=X[:,feats.index(name)]
        lp+=density(x[:,None],t*pt+(1-t)*pa,sd,lo,hi)
        lr+=density(x,rich,sd,lo,hi);lf+=density(x,frug,sd,lo,hi)
    # Given meal count/price and supermarket spend, total spending contributes
    # the same conditional N(0,40^2) factor to each class; noise_a/b do too.
    joint=lp+np.log(w);lpos=logsumexp(joint,axis=1)
    e=np.exp(logsumexp(joint+np.log(np.clip(.88-.62*t,.12,.88)),axis=1)-lpos)
    lneg=np.logaddexp(np.log(1-frugal_rate)+lr,np.log(frugal_rate)+lf)
    eta=expit(np.log(prior/(1-prior))+lpos-lneg)
    g=eta*e
    eta_u=eta*(1-e)/np.maximum(1-g,1e-15)
    return eta,e,g,eta_u

def run(seed):
    start=time.time();folder=ROOT/'results'/'propensity_audit';folder.mkdir(exist_ok=True)
    path=folder/f'{seed}.json'
    if path.exists():return seed,'cached'
    z=np.load(ROOT/'results'/'cohorts'/'baseline'/f'{seed}_data.npz')
    X,y,s,tr,te,feats=(z[k] for k in ['X','y','s','train_ids','test_ids','features']);feats=feats.tolist()
    eta,e,g,eta_u=known_conditionals(X[te],feats)
    orig=np.load(ROOT/'results'/'cohorts'/'baseline'/f'{seed}.npz')
    naive=orig['scores'][orig['models'].tolist().index('Naive')]
    # Calibrated and uncalibrated numerators share the SAME fitted 80% model.
    it,iv=train_test_split(np.arange(len(tr)),test_size=.2,random_state=19,stratify=s[tr])
    f,_=fit_model(X[tr[it]],s[tr[it]],'Naive');gcaltrain=np.clip(f(X[tr[iv]]),1e-6,1-1e-6)
    platt=LogisticRegression(C=1e6,max_iter=1000).fit(logit(gcaltrain).reshape(-1,1),s[tr[iv]])
    raw=np.clip(f(X[te]),1e-6,1-1e-6);cal=platt.predict_proba(logit(raw).reshape(-1,1))[:,1]
    e_safe=np.maximum(e,1e-6)
    methods={'NaiveFullTrain':naive,'NaiveFullTrain_div_eX_unclipped':naive/e_safe,
             'NaiveFullTrain_div_eX_clipped':np.clip(naive/e_safe,0,1),
             'SplitNumeratorRaw':raw,'SplitNumeratorRaw_div_eX':raw/e_safe,
             'SplitNumeratorCalibrated':cal,'SplitNumeratorCalibrated_div_eX':cal/e_safe,
             'KnownPosterior_eta':eta,'KnownPosterior_U':eta_u,'KnownNumerator_div_eX':g/e_safe}
    rows=[];curves=[]
    for name,sc in methods.items():
        r,c=calc_metrics(y[te],s[te],sc,te);r['model']=name;rows.append(r)
        curves.extend(dict(model=name,**v) for v in c)
    info={'seed':seed,'rows':rows,'curves':curves,'n':len(te),
          'max_identity_error':float(np.max(np.abs(g/e_safe-eta))),
          'eX_min':float(e.min()),'eX_max':float(e.max()),
          'raw_brier_S':float(brier_score_loss(s[te],raw)),
          'cal_brier_S':float(brier_score_loss(s[te],cal)),
          'population_g_brier_S':float(brier_score_loss(s[te],g)),
          'raw_logloss_S':float(log_loss(s[te],raw)),
          'cal_logloss_S':float(log_loss(s[te],cal)),
          'naive_div_eX_gt_one_fraction':float((naive/e_safe>1).mean()),
          'seconds':time.time()-start}
    if seed==42:
        eta2,e2,_,_=known_conditionals(X[te],feats,quadrature=128)
        info['quadrature_64_vs_128_max_eta_diff']=float(np.max(np.abs(eta-eta2)))
        info['quadrature_64_vs_128_max_eX_diff']=float(np.max(np.abs(e-e2)))
    np.savez_compressed(folder/f'{seed}.npz',ids=te,y=y[te],s=s[te],eX=e,eta=eta,eta_U=eta_u,g_true=g,
                        scores=np.array(list(methods.values())),models=np.array(list(methods)))
    path.write_text(json.dumps(info,indent=2));return seed,round(info['seconds'],2)

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor,as_completed
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=100);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(run,sd) for sd in range(42,42+a.seeds)]),1):
            r=f.result()
            if i%10==0 or i==a.seeds:print(i,'/',a.seeds,r,flush=True)
