"""Instrumented original algorithms plus a standardized logistic alternative.
Original HGB factory/settings and original training calls retained.
"""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pu_original as orig

def factory(kind='hgb',seed=0):
    if kind=='hgb':return orig._hgb(seed)
    if kind=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=seed))
    raise ValueError(kind)

def fit(m,X,y,w=None):
    if w is None:return m.fit(X,y)
    return m.fit(X,y,**({'logisticregression__sample_weight':w} if hasattr(m,'steps') else {'sample_weight':w}))

def fit_model(X,s,method,kind='hgb',seed=0,spy_frac=.15,quantile=5,y_audit=None):
    if len(np.unique(s))!=2:raise ValueError('Both observed classes required')
    info={}
    if method in ['Naive','TrueLabel']:
        m=fit(factory(kind,seed),X,s)
    elif method=='CostSensitive':
        w=np.where(s==1,(s==0).sum()/(s==1).sum(),1.)
        m=fit(factory(kind,seed),X,s,w)
    elif method=='Spy':
        if not 0<spy_frac<1 or not 0<=quantile<=100:raise ValueError('Invalid Spy settings')
        rng=np.random.default_rng(seed);pos=np.flatnonzero(s==1)
        ns=max(int(len(pos)*spy_frac),1)
        if ns>=len(pos):raise ValueError('Spies exhaust labeled positives')
        spies=rng.choice(pos,ns,replace=False);st=s.copy();st[spies]=0
        m1=fit(factory(kind,seed),X,st);sc=m1.predict_proba(X)[:,1]
        thr0=float(np.percentile(sc[spies],quantile,method='linear'))
        thr=thr0;rn=(s==0)&(sc<thr);fallback=rn.sum()<50
        if fallback:
            thr=float(np.percentile(sc[s==0],30,method='linear'));rn=(s==0)&(sc<thr)
        if not rn.any():raise ValueError('No RN even after original strict-threshold fallback')
        mask=(s==1)|rn;m=fit(factory(kind,seed),X[mask],s[mask])
        info={'n_spies':ns,'n_reliable_neg':int(rn.sum()),'n_unlabeled_train':int((s==0).sum()),
              'rn_retained_fraction':float(rn.sum()/(s==0).sum()),'spy_threshold':thr,
              'initial_spy_threshold':thr0,'fallback':bool(fallback)}
        if y_audit is not None:
            # Read only AFTER model fitting; never affects fit, threshold or choice.
            hidden=(s==0)&(y_audit==1)
            info.update(rn_purity=float((y_audit[rn]==0).mean()),
                        hidden_in_rn_fraction=float((rn&hidden).sum()/hidden.sum()) if hidden.any() else None)
    elif method=='ElkanNoto':
        rng=np.random.default_rng(seed);pos=np.flatnonzero(s==1)
        hold=rng.choice(pos,max(int(len(pos)*.2),1),replace=False);tr=np.ones(len(s),bool);tr[hold]=False
        g=fit(factory(kind,seed),X[tr],s[tr]);c=float(np.clip(g.predict_proba(X[hold])[:,1].mean(),1e-3,1-1e-3))
        gu=np.clip(g.predict_proba(X)[:,1],1e-6,1-1e-6)
        w=np.clip((1-c)/c*gu/(1-gu),0,1);u=s==0
        X2=np.vstack([X[s==1],X[u],X[u]])
        y2=np.r_[np.ones((s==1).sum()),np.ones(u.sum()),np.zeros(u.sum())]
        w2=np.r_[np.ones((s==1).sum()),w[u],1-w[u]]
        m=fit(factory(kind,seed),X2,y2,w2);info={'c_hat':c,'c_holdout_n':len(hold),'weight_at_one_fraction':float((w[u]==1).mean())}
    else:raise ValueError(method)
    return lambda z:m.predict_proba(z)[:,1],info
