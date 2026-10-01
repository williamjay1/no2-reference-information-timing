"""Read-only independent synthetic audit of the sixth-review state replay.

An intentionally slow chronological refilter constructs each output using only
references released strictly before it. This is implementation verification,
not new empirical evidence of physical drift or AR(2) benefits.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import revision6_model_sensitivity as candidate
from calibration_engine import initialize

def refilter(phi, valid, y, initmask, query, m0, P0, F, Q, r, ready, delay, pulsed):
    n=len(y);pred=np.full(n,np.nan);var=np.full(n,np.nan)
    h=np.r_[np.zeros(4),1.,0.] if len(m0)==6 else np.r_[np.zeros(4),1.]
    for t in range(ready+delay,n):
        if not valid[t]:continue
        m=m0.copy();P=P0.copy()
        for event in range(ready,t+1):
            q=Q.copy()
            if pulsed:q[:4,:4]*=168. if (event-ready+1)%168==0 else 0.
            m=F@m;P=F@P@F.T+q
            if event+delay<t and (query[event] or initmask[event]) and valid[event] and np.isfinite(y[event]):
                h[:4]=phi[event];ph=P@h;s=r+h@ph
                m+=ph*(y[event]-h@m)/s;P-=np.outer(ph,ph)/s;P=.5*(P+P.T)
        h[:4]=phi[t];pred[t]=h@m;var[t]=h@P@h+r
    return pred,var

def main():
    rng=np.random.default_rng(20261001);n=420;ready=96
    X=rng.normal(size=(n,3));X[:,0]+=20.;y=1.3*X[:,0]+rng.normal(size=n)
    model=initialize(X,y,ready,q_scale=.001)
    phi=model['phi'];valid=model['valid'].copy();valid[109:114]=False
    init=np.arange(n)<ready;init[190:196]=True
    query=np.zeros(n,bool);query[100:112]=True;query[130:142]=True;query[290:302]=True
    F1=np.diag(model['F']);Q1=np.diag(model['Q'])
    B=np.asarray([[.55,.22],[1.,0.]])
    P2=np.zeros((6,6));P2[:4,:4]=model['P'][:4,:4]
    P2[4:,4:]=solve_discrete_lyapunov(B,np.diag([.3,0.]))
    F2=np.eye(6);F2[4:,4:]=B
    Q2=np.zeros((6,6));Q2[:4,:4]=Q1[:4,:4];Q2[4,4]=.3
    m2=np.r_[model['m'][:4],0.,0.]
    records=[]
    for kind,m,P,F,Q in (('ar1',model['m'],model['P'],F1,Q1),('ar2',m2,P2,F2,Q2)):
        for pulsed in (False,True):
            for delay in (0,24,72,168):
                p,v=candidate.replay(phi,valid,y,init,query,m,P,F,Q,model['r'],ready,delay,pulsed)
                expected_p,expected_v=refilter(phi,valid,y,init,query,m,P,F,Q,model['r'],ready,delay,pulsed)
                np.testing.assert_allclose(p,expected_p,equal_nan=True,rtol=2e-12,atol=2e-12)
                np.testing.assert_allclose(v,expected_v,equal_nan=True,rtol=2e-12,atol=2e-12)
                assert np.nanmin(v)>0.
                records.append(dict(kind=kind,pulsed=pulsed,delay=delay,
                    prediction_max_abs_difference=float(np.nanmax(np.abs(p-expected_p))),
                    variance_max_abs_difference=float(np.nanmax(np.abs(v-expected_v)))))
    # An AR(2) companion with a zero second-lag coefficient reduces to AR(1).
    rho=model['rho'];V=model['P'][-1,-1]
    Fzero=np.eye(6);Fzero[4:,4:]=[[rho,0.],[1.,0.]]
    Pzero=P2.copy();Pzero[4:,4:]=[[V,rho*V],[rho*V,V]]
    Qzero=Q2.copy();Qzero[4,4]=Q1[-1,-1]
    for delay in (0,24,72,168):
        p1,v1=candidate.replay(phi,valid,y,init,query,model['m'],model['P'],F1,Q1,model['r'],ready,delay,False)
        p2,v2=candidate.replay(phi,valid,y,init,query,m2,Pzero,Fzero,Qzero,model['r'],ready,delay,False)
        np.testing.assert_allclose(p1,p2,equal_nan=True,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(v1,v2,equal_nan=True,rtol=2e-12,atol=2e-12)
    path=candidate.OUT/'independent_synthetic_audit.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(status='PASS',kind='synthetic implementation checks, not empirical sensor evidence',
        checked_script='revision6_model_sensitivity.py',
        independent_reference='fresh chronological Gaussian refilter for every product with release < output time',
        ar2_second_lag_zero_matches_ar1=True,records=records),indent=2),encoding='utf-8')
    print('INDEPENDENT SENSITIVITY AUDIT PASS',len(records),'cases',flush=True)
    print('MAX PREDICTION',max(row['prediction_max_abs_difference'] for row in records),
          'MAX VARIANCE',max(row['variance_max_abs_difference'] for row in records),flush=True)

if __name__=='__main__':main()
