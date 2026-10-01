"""Development-only common-engine grid. No 2021+ outcomes are inspected."""
import os
os.environ.setdefault('NUMBA_CACHE_DIR',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'cache/numba'))
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from calibration_engine import initialize, periodic_mask, run_fixed, metrics
from study_data import read_data, deployment_arrays, eligible, ROOT


def main():
    data = read_data()
    data = data[(data.site=='Manchester') & (data.timestamp < '2021-01-01')]
    rows, records = [], []
    for key, g in data.groupby('deployment'):
        for weather in [False,True]:
            arr = deployment_arrays(g, weather=weather)
            ok, score = eligible(arr)
            if not ok:
                continue
            if not weather:
                records.append(dict(deployment=key,n=len(arr['y']),scored=int(score.sum())))
            mask = periodic_mask(len(arr['y']),336,budget=48,starts=4,phase=.25)
            for q in [0.,1e-6,1e-5,1e-4,1e-3]:
                for ar in [False,True]:
                    model = initialize(arr['X'],arr['y'],336,q_scale=q,use_ar=ar)
                    pred,_=run_fixed(model,arr['y'],mask)
                    z=metrics(arr['y'],np.maximum(pred,0.),score)
                    rows.append(dict(deployment=key,weather=weather,q_scale=q,ar=ar,rho=model['rho'],**z))
            print(key,weather,len(arr['y']),int(score.sum()),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(ROOT/'results/development_engine_grid.csv',index=False)
    summary=frame.groupby(['weather','q_scale','ar']).agg(rmse=('rmse','mean'),mae=('mae','mean'),n=('deployment','count')).sort_values('rmse')
    summary.to_csv(ROOT/'results/development_engine_summary.csv')
    (ROOT/'results/development_deployments.json').write_text(json.dumps(records,indent=2))
    print(summary.to_string(),flush=True)

if __name__=='__main__':
    main()
