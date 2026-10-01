"""Meaningful causal-update and calendar-time checks on synthetic arrays only."""
import os
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).resolve().parents[1]/'cache/numba'))
import json
import numpy as np
from calibration_engine import initialize, periodic_mask, run_fixed
from study_data import ROOT

def main():
    rng=np.random.default_rng(92)
    n=1400
    x=10+5*np.sin(np.arange(n)/20)+rng.normal(size=n)
    X=np.c_[x,15+2*np.sin(np.arange(n)/120),60+10*np.sin(np.arange(n)/80)]
    y=1.2*x+2+np.cumsum(rng.normal(scale=.015,size=n))+rng.normal(size=n)
    mask=periodic_mask(n,336,48,starts=4,phase=.25)
    model=initialize(X,y,336)
    p,_=run_fixed(model,y,mask)
    hidden=y.copy(); hidden[(np.arange(n)>=336)&~mask]=99999.
    ph,_=run_fixed(model,hidden,mask)
    assert np.allclose(p,ph,equal_nan=True)
    t=int(np.flatnonzero(mask)[0]); changed=y.copy(); changed[t]+=100
    pc,_=run_fixed(model,changed,mask)
    assert np.allclose(p[:t+1],pc[:t+1],equal_nan=True)
    assert np.nanmax(np.abs(p[t+1:]-pc[t+1:]))>0
    future=X.copy(); future[1000:]+=1000.
    fm=initialize(future,y,336)
    pf,_=run_fixed(fm,y,mask)
    assert np.allclose(p[:1000],pf[:1000],equal_nan=True)
    Xm=X.copy(); Xm[500:600,0]=np.nan
    mm=initialize(Xm,y,336); pm,_=run_fixed(mm,y,mask)
    assert np.isnan(pm[500:600]).all()
    result=dict(synthetic_only=True,hidden_target_invariance=True,
                current_target_does_not_change_own_prediction=True,
                future_feature_invariance=True,missing_primary_sensor_no_prediction=True)
    (ROOT/'results/core_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__': main()
