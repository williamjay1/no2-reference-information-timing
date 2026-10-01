"""Documented response to an observed development-domain failure."""
import os
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).resolve().parents[1]/'cache/numba'))
import numpy as np
import pandas as pd
from calibration_engine import initialize, periodic_mask, run_fixed, metrics, guard_features
from study_data import read_data,deployment_arrays,eligible,ROOT

def main():
    data=read_data()
    data=data[(data.site=='Manchester')&(data.timestamp<'2021-01-01')]
    rows=[]
    for key,g in data.groupby('deployment'):
        arr=deployment_arrays(g,weather=True); ok,score=eligible(arr)
        if not ok: continue
        mask=periodic_mask(len(arr['y']),336,48,starts=4,phase=.25)
        for width in [8.,12.,24.,float('inf')]:
            X,guard=guard_features(arr['X'],width=width)
            for q in [1e-5,1e-4,1e-3]:
                model=initialize(X,arr['y'],336,q_scale=q)
                pred,_=run_fixed(model,arr['y'],mask)
                z=metrics(arr['y'],np.maximum(pred,0),score)
                rows.append(dict(deployment=key,width=width,q_scale=q,
                                 clipped=int(np.sum(np.isfinite(X[:,0])&(X[:,0]!=arr['X'][:,0]))),**z))
        print(key,flush=True)
    df=pd.DataFrame(rows); df.to_csv(ROOT/'results/development_guard_grid.csv',index=False)
    sm=df.groupby(['width','q_scale'])[['rmse','mae']].mean().sort_values('rmse')
    sm.to_csv(ROOT/'results/development_guard_summary.csv')
    print(sm.to_string(),flush=True)
if __name__=='__main__': main()
