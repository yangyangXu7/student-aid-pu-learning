"""R2 simulator: original RNG stream preserved at defaults; extensions are new.
All coefficients are stipulated, not estimates from student records.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.optimize import brentq
from scipy.special import roots_jacobi
from synth_original import FEATURES

# name, poor typical, poor atypical, rich, frugal, sd, lower, upper
SPEC = [
('avg_meal_price',5.2,8.5,10.5,7.5,1.2,2,None),
('meals_per_day',2.7,2.0,1.6,2.3,.35,.3,3.5),
('breakfast_rate',.78,.45,.28,.50,.10,0,1),
('weekend_on_campus',.85,.60,.42,.55,.12,0,1),
('canteen_ratio',.88,.70,.55,.72,.08,0,1),
('snack_drink_ratio',.05,.15,.30,.15,.06,0,1),
('low_price_window_ratio',.55,.30,.12,.30,.10,0,1),
('supermarket_spend',60,150,380,160,60,0,None),
('library_visits',14,10,8,10,4,0,None),
('dorm_night_out',1.5,3.0,5.5,3.0,1.2,0,None),
('score_rank_pct',.44,.50,.52,.50,.18,0,1),
]
_nodes,_weights=roots_jacobi(128,.8,1.2)
_T=(_nodes+1)/2; _W=_weights/_weights.sum()
LOGISTIC_A=brentq(lambda a: (_W*expit(a-4*_T)).sum()-.55,-10,10)
THRESHOLD=float(beta_dist.ppf(2/3,2.2,1.8))

def propensity(t, mechanism='linear_inverse', label_intercept=.88,label_slope=.62,
               label_lo=.12,label_hi=.88, college=None, latent=None):
    if mechanism=='linear_inverse':
        return np.clip(label_intercept-label_slope*t,label_lo,label_hi)
    if mechanism=='scar':return np.full_like(t,.55)
    if mechanism=='logistic':return expit(LOGISTIC_A-4*t)
    if mechanism=='threshold':return np.where(t>THRESHOLD,.20,.725)
    if mechanism=='heterogeneous':return np.where(college==0,np.clip(.891-.62*t,.12,.92),.55)
    if mechanism=='reverse':return np.clip(.209+.62*t,.12,.92)
    if mechanism=='nonmonotonic':return .55+.30*(np.cos(2*np.pi*t)-(_W*np.cos(2*np.pi*_T)).sum())
    if mechanism=='unobserved':
        # latent affects features as well as S; absent from model inputs.
        return expit(LOGISTIC_A-4*t+1.5*latent)
    raise ValueError(mechanism)

def generate(n=12000,poor_rate=.13,frugal_rate=.10,seed=42,
             mechanism='linear_inverse',noise_scale=1.,overlap=0.,
             label_intercept=.88,label_slope=.62,label_lo=.12,label_hi=.88,
             false_positive_rate=0., generator='original'):
    if generator!='original':return generate_factors(n,poor_rate,seed)
    rng=np.random.default_rng(seed)
    poor=rng.random(n)<poor_rate
    t=rng.beta(2.2,1.8,n)
    frugal=(~poor)&(rng.random(n)<frugal_rate)
    # Independent extra stream avoids shifting baseline feature or label draws.
    aux=np.random.default_rng(seed+1000003)
    college=aux.integers(0,2,n); latent=aux.normal(size=n)
    columns={}
    for name,pt,pa,rich,frug,sd,lo,hi in SPEC:
        frug=(1-overlap)*frug+overlap*(.55*pt+.45*pa)
        mu=np.where(poor,t*pt+(1-t)*pa,np.where(frugal,frug,rich))
        x=mu+rng.normal(0,sd*noise_scale,n)
        if mechanism=='unobserved':x=x+.5*sd*latent
        columns[name]=np.clip(x,lo,hi)
    columns['monthly_total_spend']=(columns['meals_per_day']*30*columns['avg_meal_price']+
                                   columns['supermarket_spend']+rng.normal(0,40*noise_scale,n))
    columns['noise_a']=rng.normal(0,1,n);columns['noise_b']=rng.normal(0,1,n)
    df=pd.DataFrame(columns)[FEATURES]
    q=propensity(t,mechanism,label_intercept,label_slope,label_lo,label_hi,college,latent)
    if (q<0).any() or (q>1).any():raise ValueError('Invalid propensity')
    s=poor & (rng.random(n)<q)
    if false_positive_rate:s|=(~poor)&(aux.random(n)<false_positive_rate)
    df['y_true']=poor.astype(int);df['labeled']=s.astype(int)
    df['typicality']=np.where(poor,t,np.nan)
    df['latent_typicality_all']=t;df['q_individual']=q;df['college']=college
    df['frugal']=frugal.astype(int)
    return df

# New fully specified alternative, inspired by the R1 description but NOT its missing code.
FACTOR_CORR=np.array([[1,-.28,.36],[-.28,1,.18],[.36,.18,1]])
FACTOR_LOADINGS=np.array([[1,-.15,.10],[-.2,.15,.6],[-.2,.4,.4],[-.15,.15,.6],
                         [-.5,.1,.3],[.6,-.1,-.2],[-.6,.1,.1],[.8,0,-.1],
                         [-.1,.8,.3],[.4,-.3,-.4],[.15,-.7,.1]])

def generate_factors(n,poor_rate,seed):
    from scipy.stats import norm
    rng=np.random.default_rng(seed)
    z=rng.multivariate_normal(np.zeros(3),FACTOR_CORR,n)
    college=rng.integers(0,2,n)
    ses=z[:,0]+.25*(2*college-1)
    cutoff=brentq(lambda a:.5*norm.cdf(a-.25)+.5*norm.cdf(a+.25)-poor_rate,-5,5)
    y=ses<cutoff
    t=1-norm.cdf(z[:,0]) # defined on all cases, excluded from features
    columns={}
    for j,(name,pt,pa,rich,frug,sd,lo,hi) in enumerate(SPEC):
        mu=(.55*pt+.45*pa+rich)/2
        scale=max(abs(rich-(.55*pt+.45*pa))*.65,sd)
        v=mu+scale*(z@FACTOR_LOADINGS[j]+.18*(2*college-1))
        v+=rng.standard_t(5,n)*sd*np.sqrt(3/5)
        columns[name]=np.clip(v,lo,hi)
    columns['monthly_total_spend']=(columns['meals_per_day']*30*columns['avg_meal_price']+
                                   columns['supermarket_spend']+rng.standard_t(5,n)*40*np.sqrt(3/5))
    columns['noise_a']=rng.normal(size=n);columns['noise_b']=rng.normal(size=n)
    q=np.clip(.88-.62*t,.12,.88)
    s=y&(rng.random(n)<q)
    df=pd.DataFrame(columns)[FEATURES]
    for k,v in {'y_true':y.astype(int),'labeled':s.astype(int),'typicality':np.where(y,t,np.nan),
                'latent_typicality_all':t,'q_individual':q,'college':college,'frugal':np.zeros(n,int)}.items():df[k]=v
    return df
