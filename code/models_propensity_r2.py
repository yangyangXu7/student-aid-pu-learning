"""Explicit R2 reimplementations; these are NOT the missing R1 source.

SAR_EM: Bekker et al. SAR-PU EM, restricted propensity features and no final
inverse-propensity refit (the authors' refit_classifier=False option).
R1_IndexEM: R1's described one-dimensional propensity restriction; it must NOT
be presented as Gong et al.'s canonical LBE algorithm.
LBE_LF: Gong et al. logistic likelihood and initialization, full-batch Adam
generalized EM. Every E-step uses only X and observed S.
"""
import os
for _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_v]='1'
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

PROPENSITY_FEATURES=['monthly_total_spend','avg_meal_price','meals_per_day',
 'breakfast_rate','weekend_on_campus','canteen_ratio','snack_drink_ratio',
 'low_price_window_ratio','supermarket_spend']
HARDSHIP_SIGNS=np.array([-1.,-1.,1.,1.,1.,1.,-1.,1.,-1.])
CONFIG={'SAR_EM':{'max_iter':500,'window':10,'slope_tol':1e-4,'q_tol':1e-4,'C':1.},
 'R1_IndexEM':{'max_iter':500,'window':10,'slope_tol':1e-4,'q_tol':1e-4,'C':1.},
 'LBE_LF':{'max_iter':20000,'min_iter':1000,'window':100,'ll_tol':1e-7,
  'gradient_tol':1e-5,'learning_rate':.001,'beta1':.9,'beta2':.999,'epsilon':1e-8,
  'initial_C':1.,'l2':0.}}

def design(x):return np.column_stack([np.ones(len(x)),x])

def logistic_loss_gradient(theta,A,t,w=None,l2=1.):
    if w is None:w=np.ones(len(t))
    z=A@theta
    loss=np.sum(w*(np.logaddexp(0,z)-t*z))+.5*l2*np.dot(theta[1:],theta[1:])
    gradient=A.T@(w*(expit(z)-t));gradient[1:]+=l2*theta[1:]
    return loss,gradient

def fit_soft(A,t,w=None,theta=None,l2=1.):
    # Fractional response is exactly equivalent to duplicating both labels
    # with weights t and 1-t, without doubling the matrix allocation.
    if theta is None:theta=np.zeros(A.shape[1])
    res=minimize(logistic_loss_gradient,theta,args=(A,t,w,l2),jac=True,
                 method='L-BFGS-B',options={'maxiter':200,'ftol':1e-12,'gtol':1e-7,'maxls':40})
    if not np.isfinite(res.fun) or (not res.success and np.linalg.norm(res.jac,np.inf)>1e-3):
        raise RuntimeError('Logistic M-step failed: '+str(res.message))
    return res.x

def expectation(f,e,s):
    return s+(1-s)*f*(1-e)/np.maximum(1-f*e,1e-15)

def observed_ll(f,e,s):
    g=np.clip(f*e,1e-15,1-1e-15)
    return float(np.mean(s*np.log(g)+(1-s)*np.log1p(-g)))

def expected_ll(f,e,s):
    r=expectation(f,e,s)
    return float(np.mean(s*np.log(np.maximum(f*e,1e-15))+(1-s)*(
      r*np.log(np.maximum(f*(1-e),1e-15))+(1-r)*np.log(np.maximum(1-f,1e-15)))))

def fit_propensity(X,s,features,method,overrides=None):
    if method not in CONFIG:raise ValueError(method)
    cfg=dict(CONFIG[method]);cfg.update(overrides or {})
    scaler=StandardScaler().fit(X);z=scaler.transform(X);A=design(z)
    ix=[features.index(f) for f in PROPENSITY_FEATURES]
    if method=='LBE_LF':B=A
    elif method=='R1_IndexEM':B=design((z[:,ix]@HARDSHIP_SIGNS/len(ix))[:,None])
    else:B=design(z[:,ix])
    history=[];qhist=[];ehist=[];maxstep=0;converged=False
    if method=='LBE_LF':
        theta=fit_soft(A,s,l2=1/cfg['initial_C']);gamma=np.zeros(B.shape[1])
        m1=np.zeros_like(theta);v1=m1.copy();m2=np.zeros_like(gamma);v2=m2.copy()
        f=expit(A@theta);e=expit(B@gamma);history.append(observed_ll(f,e,s))
        for it in range(1,cfg['max_iter']+1):
            r=expectation(f,e,s)
            # Negative expected complete-data log likelihood, normalized by n.
            gf=A.T@(f-r)/len(s);ge=B.T@(r*e-s)/len(s)
            m1=cfg['beta1']*m1+(1-cfg['beta1'])*gf;v1=cfg['beta2']*v1+(1-cfg['beta2'])*gf**2
            m2=cfg['beta1']*m2+(1-cfg['beta1'])*ge;v2=cfg['beta2']*v2+(1-cfg['beta2'])*ge**2
            bc1=1-cfg['beta1']**it;bc2=1-cfg['beta2']**it
            theta-=cfg['learning_rate']*(m1/bc1)/(np.sqrt(v1/bc2)+cfg['epsilon'])
            gamma-=cfg['learning_rate']*(m2/bc1)/(np.sqrt(v2/bc2)+cfg['epsilon'])
            f=expit(A@theta);e=expit(B@gamma);history.append(observed_ll(f,e,s))
            if it>=cfg['min_iter'] and it%cfg['window']==0:
                grad=max(float(np.max(np.abs(gf))),float(np.max(np.abs(ge))))
                improvement=abs(history[-1]-history[-1-cfg['window']])
                if improvement<cfg['ll_tol'] and grad<cfg['gradient_tol']:
                    converged=True;break
        r=expectation(f,e,s)
        grad=max(float(np.max(np.abs(A.T@(f-r)/len(s)))),float(np.max(np.abs(B.T@(r*e-s)/len(s)))))
        stop={'gradient_inf':grad,'window_ll_change':abs(history[-1]-history[-1-min(cfg['window'],it)])}
    else:
        prop=float(s.mean());w=s*(1-prop)+(1-s)*prop
        theta=fit_soft(A,s,w=w,l2=1/cfg['C']);f=expit(A@theta)
        gamma=fit_soft(B,s,w=s+(1-s)*f,l2=1/cfg['C']);e=expit(B@gamma)
        history.append(observed_ll(f,e,s));qhist.append(expected_ll(f,e,s))
        for it in range(1,cfg['max_iter']+1):
            r=expectation(f,e,s)
            gamma=fit_soft(B,s,w=r,theta=gamma,l2=1/cfg['C'])
            theta=fit_soft(A,r,theta=theta,l2=1/cfg['C'])
            f=expit(A@theta);e=expit(B@gamma)
            history.append(observed_ll(f,e,s));qhist.append(expected_ll(f,e,s));ehist.append(e[s==0].copy())
            if len(ehist)>cfg['window']:ehist.pop(0)
            if it>cfg['window']+1:
                tx=np.arange(cfg['window'])-(cfg['window']-1)/2
                slope=float(np.mean(np.abs(tx@np.array(ehist)/(tx@tx))))
                qchange=max(qhist[-cfg['window']:])-qhist[-cfg['window']]
                if slope<cfg['slope_tol'] and qchange<cfg['q_tol']:
                    converged=True;break
        stop={'mean_abs_propensity_slope':slope,'window_expected_ll_gain':qchange}
    info={'config':cfg,'iterations':it,'converged':bool(converged),'stop':stop,
          'initial_observed_ll':history[0],'final_observed_ll':history[-1],
          'observed_ll_history':history[::max(1,len(history)//250)],
          'n_ll_decreases_gt_1e_8':int((np.diff(history)<-1e-8).sum()),
          'class_coef':theta.tolist(),'propensity_coef':gamma.tolist(),
          'train_mean_class_probability':float(f.mean()),
          'train_mean_propensity':float(e.mean()),
          'propensity_features':features if method=='LBE_LF' else PROPENSITY_FEATURES,
          'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist()}
    def predict(xx):
        zz=scaler.transform(xx);aa=design(zz)
        if method=='LBE_LF':bb=aa
        elif method=='R1_IndexEM':bb=design((zz[:,ix]@HARDSHIP_SIGNS/len(ix))[:,None])
        else:bb=design(zz[:,ix])
        return expit(aa@theta),expit(bb@gamma)
    return predict,info
