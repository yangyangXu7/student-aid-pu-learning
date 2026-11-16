"""Audit denominator choices and a prospective sampling design in SIMULATION.
No actual student audit, real fairness claim, or recovery of missing R1 code.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import brier_score_loss,log_loss
from run_r2 import ROOT,calc_metrics
from generator_r2 import generate,FEATURES

def run():
    folder=ROOT/'results'/'denominator_audit';folder.mkdir(exist_ok=True)
    sampling=[]
    for seed in range(42,142):
        base=np.load(ROOT/'results'/'cohorts'/'baseline'/f'{seed}.npz')
        z=np.load(ROOT/'results'/'propensity_audit'/f'{seed}.npz')
        df=generate(seed=seed);ids=z['ids'];y=z['y'];s=z['s'];e=z['eX']
        assert np.array_equal(ids,base['ids'])
        q=df.q_individual.to_numpy()[ids]
        get=lambda m:z['scores'][z['models'].tolist().index(m)]
        raw=get('NaiveFullTrain');split_raw=get('SplitNumeratorRaw');cal=get('SplitNumeratorCalibrated')
        # q(T) exists for all generated rows, but is NOT conditional e_X(X).
        # The next two denominators use true Y and are invalid practical procedures.
        q_neg1=np.where(y==1,q,1.);q_neg05=np.where(y==1,q,.5)
        methods={'Raw_div_latent_q_all':raw/q,'Cal_div_latent_q_all':cal/q,
                 'Raw_div_latent_q_all_clipped':np.clip(raw/q,0,1),
                 'INVALID_truth_dependent_negative_e1':raw/q_neg1,
                 'INVALID_truth_dependent_negative_e05':raw/q_neg05,
                 'Raw_div_conditional_eX':raw/e,'Cal_div_conditional_eX':cal/e,
                 'Cal_div_conditional_eX_clipped':np.clip(cal/e,0,1),
                 'Known_g_div_latent_q_all':z['g_true']/q,
                 'Known_g_div_conditional_eX':z['g_true']/e}
        for low in [.05,.1,.2,.4]:methods['Raw_div_eX_lower_'+str(low)]=raw/np.maximum(e,low)
        for upper in [.9,.99,1.]:methods['Raw_div_eX_output_cap_'+str(upper)]=np.minimum(raw/e,upper)
        rows=[]
        for m,p in methods.items():
            r,_=calc_metrics(y,s,p,ids);r['model']=m;rows.append(r)
        result={'seed':seed,'rows':rows,'negative_case_convention':{
            'conditional_eX':'Integrate q(T) against T|Y=1,X for every X; no test Y is used.',
            'latent_q_all':'q(T_i) is defined on every simulated row. It is not e_X(X_i).',
            'INVALID':'Uses true test Y to assign a constant denominator to negatives; diagnostic only.'},
            'q_all_min':float(q.min()),'q_all_max':float(q.max()),
            'eX_min':float(e.min()),'eX_max':float(e.max()),
            'mean_abs_q_minus_e_on_positives':float(np.mean(np.abs(q[y==1]-e[y==1]))),
            'identity_max_error':float(np.max(np.abs(z['g_true']/e-z['eta'])))}
        (folder/f'{seed}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        np.savez_compressed(folder/f'{seed}.npz',ids=ids,y=y,s=s,models=np.array(list(methods)),
                            scores=np.array(list(methods.values())),q_all=q,eX=e)
        # Sampling-design sensitivity: verify all top-10% U students and a SRS
        # without replacement from the remaining U students, 1,000 draws each.
        p=base['scores'][base['models'].tolist().index('Spy')];u=np.flatnonzero(s==0)
        order=u[np.lexsort((ids[u],-p[u]))];k=int(len(u)*.1)
        tp=int(y[order[:k]].sum());fn=int(y[order[k:]].sum());rest=len(u)-k
        for n_audit in [50,100,200,400]:
            rng=np.random.default_rng(np.random.SeedSequence([78004,seed,n_audit]))
            observed_fn=rng.hypergeometric(fn,rest-fn,n_audit,size=1000)
            missed_hat=rest*observed_fn/n_audit;recall_hat=tp/(tp+missed_hat)
            sampling.append({'seed':seed,'nonselected_sample':n_audit,'repeats':1000,
                'n_U':len(u),'shortlist_n':k,'true_shortlist_positives':tp,'true_missed_positives':fn,
                'true_recall':tp/(tp+fn),'missed_hat_mean':float(missed_hat.mean()),
                'missed_hat_bias':float(missed_hat.mean()-fn),'missed_hat_rmse':float(np.sqrt(np.mean((missed_hat-fn)**2))),
                'recall_hat_mean':float(recall_hat.mean()),'recall_hat_bias':float(recall_hat.mean()-tp/(tp+fn)),
                'recall_hat_rmse':float(np.sqrt(np.mean((recall_hat-tp/(tp+fn))**2))),
                'probability_observe_no_missed_exact':float(hypergeom.pmf(0,rest,fn,n_audit)),
                'fraction_observe_no_missed_MC':float((observed_fn==0).mean()),
                'nonselected_inclusion_probability':n_audit/rest})
    pd.DataFrame(sampling).to_csv(ROOT/'results'/'tables'/'prospective_sampling_simulation_raw.csv',index=False)
    print('Completed denominator audit (100 cohorts, 17 scores) and sampling-design simulation (400,000 draws).',flush=True)

def estimated_propensities():
    rows=[]
    for seed in range(42,142):
        z=np.load(ROOT/'results'/'completion'/'baseline'/f'{seed}.npz')
        a=np.load(ROOT/'results'/'propensity_audit'/f'{seed}.npz')
        y,s,e=a['y'],a['s'],a['eX'];assert np.array_equal(z['ids'],a['ids'])
        for name,eh in zip(z['propensity_models'],z['propensities']):
            f=z['scores'][z['models'].tolist().index(name)];g=f*eh
            rows.append({'seed':seed,'model':name,'class_brier_Y':brier_score_loss(y,f),
                'numerator_brier_S':brier_score_loss(s,g),'numerator_logloss_S':log_loss(s,g),
                'propensity_rmse_truepositives':float(np.sqrt(np.mean((eh[y==1]-e[y==1])**2))),
                'class_mean':float(f.mean()),'propensity_mean_truepositives':float(eh[y==1].mean()),
                'true_conditional_propensity_mean_truepositives':float(e[y==1].mean())})
    pd.DataFrame(rows).to_csv(ROOT/'results'/'tables'/'estimated_propensity_quality.csv',index=False)
    print('Completed fitted-propensity audit.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fitted-only',action='store_true');a=p.parse_args()
    estimated_propensities() if a.fitted_only else run()
