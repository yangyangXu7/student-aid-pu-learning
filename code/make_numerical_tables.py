"""Recreate numerical paper tables and Spy-grid summaries without plotting.
Numerical projections extracted from the original mixed plotting/report scripts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from run_r2 import ROOT
T=ROOT/'results/tables'
LABEL={'Naive':'Naive','CostSensitive':'Cost sensitive','ElkanNoto':'Elkan-Noto','Spy':'Spy','TrueLabel':'True-label reference','LowSpending':'Low spending'}
MODELS=list(LABEL)
MECH=['beta_0.0','baseline','logistic','threshold','heterogeneous','reverse','nonmonotonic','unobserved']
MLABEL=['SCAR','Linear inverse','Logistic','Threshold','Two-college','Reverse','Nonmonotonic','Unobserved factor']
METRICS=pd.read_csv(T/'summary_metrics.csv');PAIRS=pd.read_csv(T/'paired_method_differences.csv')
RAW=pd.read_csv(T/'cohort_metrics.csv')
def val(s,m,metric):return METRICS[(METRICS.scenario==s)&(METRICS.model==m)&(METRICS.metric==metric)].iloc[0]
def pair(s,metric,a='Spy',b='Naive'):return PAIRS[(PAIRS.scenario==s)&(PAIRS.method_a==a)&(PAIRS.method_b==b)&(PAIRS.metric==metric)].iloc[0]
def ms(s,m,k):
    r=val(s,m,k);return f'{r["mean"]:.4f} ± {r.sd:.4f}'
def ci(r,scale=1):return f'{r.difference*scale:+.4f} [{r.ci_low*scale:+.4f}, {r.ci_high*scale:+.4f}]'

gridrows=[]
for f in [.05,.15,.30]:
    for q in [1,5,10]:
        sc='baseline' if (f==.15 and q==5) else 'spy_f'+str(f)+'_q'+str(q)
        r=RAW[(RAW.scenario==sc)&(RAW.seed<62)&(RAW.model=='Spy')]
        info=pd.read_csv(T/'method_diagnostics_raw.csv');di=info[(info.scenario==sc)&(info.seed<62)&(info.model=='Spy')]
        gridrows.append(dict(spy_fraction=f,quantile=q,n=len(r),HR_U_5=r.HR_U_5.mean(),HR_U_10=r.HR_U_10.mean(),AP_U=r.AP_U.mean(),RN_fraction=di.rn_retained_fraction.mean()))
grid=pd.DataFrame(gridrows);grid.to_csv(T/'spy_grid_summary.csv',index=False)

def write_latex(frame,path,caption,label):
    def esc(v):
        s=str(v).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
        return s.replace('±',r'$\pm$')
    lines=[r'\begin{table}[ht]',r'\centering',r'\caption{'+esc(caption)+'}',r'\label{'+label+'}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\hline']
    lines.append(' & '.join(esc(c) for c in frame.columns)+r' \\')
    lines.append(r'\hline')
    for row in frame.itertuples(index=False,name=None):lines.append(' & '.join(esc(v) for v in row)+r' \\')
    lines.extend([r'\hline',r'\end{tabular}}',r'\end{table}'])
    path.write_text('\n'.join(lines)+'\n')


head=[]
for m in MODELS:
    head.append({'method':LABEL[m],**{k:ms('baseline',m,k) for k in ['AP_full','AP_U','P_U_10','HR_U_5','HR_U_10','HR_U_15']}})
headline=pd.DataFrame(head);headline.to_csv(T/'main_results_for_paper.csv',index=False)
write_latex(headline,T/'main_results_for_paper.tex',caption='Synthetic baseline, 100 matched cohorts. Values are mean and sample standard deviation. U denotes unlabeled test students. Shortlist percentages use the size of U as denominator.',label='tab:r2main')
mechrows=[]
for s,label in zip(MECH,MLABEL):
    r=pair(s,'HR_U_10');ra=pair(s,'AP_full')
    mechrows.append({'mechanism':label,'n':int(r.n),'Naive_HR_U10':ms(s,'Naive','HR_U_10'),'Spy_HR_U10':ms(s,'Spy','HR_U_10'),'paired_HR_U10_difference_95CI':ci(r),'paired_AP_full_difference_95CI':ci(ra)})
mecht=pd.DataFrame(mechrows);mecht.to_csv(T/'mechanism_results_for_paper.csv',index=False)
write_latex(mecht,T/'mechanism_results_for_paper.tex',caption='Fully specified R2 mechanisms. Paired percentile confidence intervals use 10,000 cohort bootstrap resamples; intervals are unadjusted.',label='tab:r2mechanisms')


NAMES={'Naive':'Naive HGB','CostSensitive':'Weighted HGB','ElkanNoto':'Elkan-Noto HGB','Spy':'Spy HGB',
 'TrueLabel':'True-label HGB','LowSpending':'Low spending','Naive_LR':'Naive logistic',
 'CostSensitive_LR':'Weighted logistic','ElkanNoto_LR':'Elkan-Noto logistic','Spy_LR':'Spy logistic',
 'TrueLabel_LR':'True-label logistic','SAR_EM':'SAR-EM','R1_IndexEM':'Index-EM (R1-style)','LBE_LF':'LBE-LF'}
PRIMARY=list(NAMES)

def tex(frame,path,caption,label):
    def esc(x):return str(x).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('±',r'$\pm$')
    lines=[r'\begin{table}[ht]',r'\centering',r'\caption{'+esc(caption)+'}',r'\label{'+label+'}',
      r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\hline',
      ' & '.join(esc(c) for c in frame.columns)+r' \\',r'\hline']
    lines+=[' & '.join(esc(v) for v in row)+r' \\' for row in frame.itertuples(index=False,name=None)]
    lines+=[r'\hline',r'\end{tabular}}',r'\end{table}'];path.write_text('\n'.join(lines)+'\n')


ag=pd.read_csv(T/'complete_summary_metrics.csv');pa=pd.read_csv(T/'complete_paired_differences.csv')
def val(sc,m,k):return ag[(ag.scenario==sc)&(ag.model==m)&(ag.metric==k)].iloc[0]
def pair(sc,a,b,k):return pa[(pa.scenario==sc)&(pa.method_a==a)&(pa.method_b==b)&(pa.metric==k)].iloc[0]
def ms(sc,m,k):
    r=val(sc,m,k);return f'{r["mean"]:.4f} ± {r.sd:.4f}'
def ci(r):return f'{r.difference:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]'

mainrows=[{'Method':NAMES[m],**{k:ms('baseline',m,k) for k in ['AP_full','AP_U','P_U_5','P_U_10','P_U_15','HR_U_5','HR_U_10','HR_U_15']}} for m in PRIMARY]
mainframe=pd.DataFrame(mainrows);mainframe.to_csv(T/'complete_main_for_paper.csv',index=False)
tex(mainframe,T/'complete_main_for_paper.tex','Baseline simulation, 100 paired cohorts. Entries are mean and sample SD. Screening is within U. All added EM estimators are explicit R2 reimplementations.','tab:r2_complete_main')
refrows=[{'Method':NAMES[m],'Reference':NAMES[b],**{k:ci(pair('baseline',m,b,k)) for k in ['AP_full','AP_U','HR_U_5','HR_U_10','HR_U_15']}} for m in ['SAR_EM','R1_IndexEM','LBE_LF'] for b in ['Spy','TrueLabel','TrueLabel_LR']]
refs=pd.DataFrame(refrows);refs.to_csv(T/'complete_reference_contrasts_for_paper.csv',index=False)
tex(refs,T/'complete_reference_contrasts_for_paper.tex','Mean paired differences and unadjusted percentile 95 percent confidence intervals, 10,000 resamples of the 100 baseline cohorts.','tab:r2_reference_contrasts')
mechrows=[]
for sc,lab in zip(MECH+['alternative_factors'],MLABEL+['Factor DGP']):
    for m in ['Spy','SAR_EM','R1_IndexEM','LBE_LF','TrueLabel','TrueLabel_LR']:
        r=pair(sc,'Spy','Naive','AP_U') if m=='Spy' else (pair(sc,m,'TrueLabel_LR','AP_U') if m in ['SAR_EM','R1_IndexEM','LBE_LF'] else None)
        mechrows.append({'Scenario':lab,'Method':NAMES[m],'n':100,'AP_full':ms(sc,m,'AP_full'),
            'AP_U':ms(sc,m,'AP_U'),'HR_U_10':ms(sc,m,'HR_U_10'),
            'Paired_AP_U':ci(r) if r is not None else '',
            'Contrast_reference':'Naive HGB' if m=='Spy' else ('True-label logistic' if r is not None else '')})
pd.DataFrame(mechrows).to_csv(T/'complete_mechanisms_for_paper.csv',index=False)
print("Numerical paper tables and Spy-grid summary written.")
