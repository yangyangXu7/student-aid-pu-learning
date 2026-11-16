"""Independent numerical and saved-result checks for the R2 completion."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from models_propensity_r2 import design,fit_soft,logistic_loss_gradient,expectation,observed_ll
from run_r2 import ROOT,calc_metrics

def numerical():
    rng=np.random.default_rng(7801);X=rng.normal(size=(180,7));A=design(X)
    target=rng.uniform(.05,.95,len(X));w=rng.uniform(.1,2,len(X));theta=rng.normal(size=A.shape[1])
    error=check_grad(lambda b:logistic_loss_gradient(b,A,target,w)[0],
                     lambda b:logistic_loss_gradient(b,A,target,w)[1],theta)
    assert error<1e-4,error
    got=fit_soft(A,target,w=w)
    expected=LogisticRegression(C=1,tol=1e-10,max_iter=10000).fit(
        np.vstack([X,X]),np.r_[np.ones(len(X)),np.zeros(len(X))],
        sample_weight=np.r_[w*target,w*(1-target)])
    soft_error=float(np.max(np.abs(expit(A@got)-expected.predict_proba(X)[:,1])))
    assert soft_error<1e-5,soft_error
    f=expit(A@theta);e=rng.uniform(.05,.95,len(X));s=rng.binomial(1,f*e)
    r=expectation(f,e,s);explicit=np.where(s==1,1,f*(1-e)/(f*(1-e)+1-f))
    assert np.max(np.abs(r-explicit))<1e-13
    # The Adam gradient derived from the E step equals observed-data NLL gradient.
    b=rng.normal(size=A.shape[1]);joint=np.r_[theta,b]
    def nll(t):return -observed_ll(expit(A@t[:A.shape[1]]),expit(A@t[A.shape[1]:]),s)
    def grad(t):
        pp=expit(A@t[:A.shape[1]]);ee=expit(A@t[A.shape[1]:]);rr=expectation(pp,ee,s)
        return np.r_[A.T@(pp-rr)/len(s),A.T@(rr*ee-s)/len(s)]
    joint_error=check_grad(nll,grad,joint)
    assert joint_error<1e-6,joint_error
    return {'gradient_fd_error':float(error),'soft_vs_duplicated_sklearn_max_probability_error':soft_error,
            'joint_likelihood_gradient_fd_error':float(joint_error)}

def artifacts():
    plan=json.loads((ROOT/'results'/'completion_plan.json').read_text());count=0;max_error=0.;nonconv=[];allfinite=True
    for task in plan['tasks']:
        sc,sd=task['scenario'],task['seed'];folder=ROOT/'results'/'completion'/sc
        r=json.loads((folder/f'{sd}.json').read_text());z=np.load(folder/f'{sd}.npz')
        original=json.loads((ROOT/'results'/'cohorts'/sc/f'{sd}.json').read_text())
        old=np.load(ROOT/'results'/'cohorts'/sc/f'{sd}.npz')
        assert r['task']==task and r['code_hash']==plan['code_hash']
        assert r['data_sha256']==original['data_sha256']
        for k in ['ids','y','s']:assert np.array_equal(z[k],old[k])
        assert np.isfinite(z['scores']).all() and np.isfinite(z['propensities']).all()
        assert ((z['scores']>=0)&(z['scores']<=1)).all()
        names=z['models'].tolist();rows={a['model']:a for a in r['rows']}
        for name,scor in zip(names,z['scores']):
            assert abs(average_precision_score(z['y'],scor)-rows[name]['AP_full'])<1e-13
            # Recompute exact shortlisted sets independently of calc_metrics.
            idx=np.flatnonzero(z['s']==0);order=idx[np.lexsort((z['ids'][idx],-scor[idx]))]
            for pct in [5,10,15]:
                k=int(len(idx)*pct/100);tp=z['y'][order[:k]].sum()
                vals={'P_U_'+str(pct):tp/k,'HR_U_'+str(pct):tp/z['y'][idx].sum()}
                for m,v in vals.items():
                    error=abs(float(v)-rows[name][m]);max_error=max(max_error,error);assert error<1e-13
        for m in ['SAR_EM','R1_IndexEM','LBE_LF']:
            if not r['method_info'][m]['converged']:nonconv.append({'scenario':sc,'seed':sd,'model':m,'stop':r['method_info'][m]['stop']})
        if sc=='baseline':
            # Independent prior run with the same logistic pipeline must agree.
            zz=np.load(ROOT/'results'/'cohorts'/'logistic_learner'/f'{sd}.npz')
            for m in ['Naive','CostSensitive','ElkanNoto','Spy','TrueLabel']:
                assert np.array_equal(z['scores'][names.index(m+'_LR')],zz['scores'][zz['models'].tolist().index(m)])
        count+=1
    return {'cohorts_checked':count,'prediction_arrays_checked':count*11,
            'max_recomputed_metric_error':max_error,'nonconverged':nonconv,'all_prediction_arrays_finite':allfinite}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--numerical-only',action='store_true');a=p.parse_args()
    out={'status':'PASS','numerical':numerical()}
    if not a.numerical_only:out['artifacts']=artifacts()
    path=ROOT/'results'/('completion_numerical_verification.json' if a.numerical_only else 'completion_verification.json')
    path.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='artifacts'}))
    if 'artifacts' in out:print('verified',out['artifacts']['cohorts_checked'],'cohorts; nonconverged',len(out['artifacts']['nonconverged']))
