"""Select acquisition hyperparameters solely on Manchester through 2020."""
import os
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).resolve().parents[1]/'cache/numba'))
import json,time
import numpy as np
import pandas as pd
from site_data import make_site
from acquisition_controller import run_site,run_site_fixed
from fixed_schedules import fixed_site_mask
from calibration_engine import metrics
from study_data import ROOT


def summarize(fleet,output,policy,setting,budget=48):
    rows=[]
    pred=np.maximum(output['pred'],0)
    for i,m in enumerate(fleet['metadata']):
        z=metrics(fleet['y'],pred[i],fleet['score_masks'][i])
        rows.append(dict(**{k:m[k] for k in ['site','deployment','instrument','family']},
                         policy=policy,setting=setting,budget=budget,**z))
    return rows


def main():
    f=make_site('Manchester',stage='development')
    print('fleet',len(f['models']),len(f['y']),flush=True)
    rows=[]; access=[]
    for phase in [0.,.25,.5,.75]:
        q,_=fixed_site_mask(len(f['y']),int(min(f['ready'])),f['initialization_mask'],phase=phase)
        out=run_site_fixed(f['models'],f['y'],f['ready'],f['initialization_mask'],q)
        rows+=summarize(f,out,'periodic',phase)
        access.append(dict(policy='periodic',setting=phase,runtime=out['runtime_seconds'],queries=out['additional_attempts']))
        print('periodic',phase,np.mean([r['rmse'] for r in rows[-len(f['models']):]]),out['runtime_seconds'],flush=True)
    for policy in ['risk','information_gain','uncertainty','risk_independent','risk_fixed','window_uncertainty']:
        lengths=[6,12,24] if policy in ['risk_fixed','window_uncertainty'] else [12]
        horizons=[24,48,168] if policy!='window_uncertainty' else [48]
        for H in horizons:
            for L in lengths:
                for scale in [.75,1.,1.25]:
                    out=run_site(f['models'],f['y'],f['ready'],f['initialization_mask'],policy=policy,threshold_scale=scale,fixed_length=L,horizon=H)
                    setting=f'scale={scale},L={L},H={H}'
                    z=summarize(f,out,policy,setting); rows+=z
                    access.append(dict(policy=policy,setting=setting,runtime=out['runtime_seconds'],queries=out['additional_attempts'],forced=int(out['forced_query'].sum()),starts=int(out['starts'].sum())))
                    print(policy,setting,np.mean([r['rmse'] for r in z]),out['runtime_seconds'],flush=True)
                    pd.DataFrame(rows).to_csv(ROOT/'results/development_policy_metrics.csv',index=False)
                    pd.DataFrame(access).to_csv(ROOT/'results/development_policy_access.csv',index=False)
    df=pd.DataFrame(rows)
    sm=df.groupby(['policy','setting'],sort=False)[['rmse','mae']].mean().sort_values('rmse')
    sm.to_csv(ROOT/'results/development_policy_summary.csv')
    print(sm.to_string(),flush=True)

if __name__=='__main__': main()
