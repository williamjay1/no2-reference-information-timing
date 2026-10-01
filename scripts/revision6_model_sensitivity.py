"""Prespecified correction-model sensitivities on fixed, equal-budget calendars.

These experiments test corrected products under changed state assumptions.
They do not optimize an AR(2) acquisition planner or infer actual electrode drift.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
for k,v in {'NUMBA_CACHE_DIR':ROOT/'cache/numba','TEMP':ROOT/'temp','TMP':ROOT/'temp'}.items():
    os.environ.setdefault(k,str(v))
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,copy,json,time
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from numba import njit
from site_data import make_site
from revision_access import CONFIG,macro
import revision_access_controller as original
from statistical_analysis import load_run,build_site_bundle,metric_tables

OUT=ROOT/'results/revision6/model_sensitivity'
EV=ROOT/'evidence/revision6'
VARIANTS=('ar1','ar2','drift_low','drift_high','drift_pulsed')

@njit(cache=True)
def replay(phi,valid,y,initmask,query,m0,P0,F,Q,r,ready,delay,pulsed):
    n=len(y);p=len(m0)
    pred=np.full(n,np.nan);var=np.full(n,np.nan)
    m=m0.copy();P=P0.copy();A=np.eye(p);S=np.zeros((p,p))
    for _ in range(delay):
        S=F@S@F.T+Q;A=F@A
    h=np.zeros(p);h[-2 if p==6 else -1]=1.
    for t in range(ready+delay,n):
        u=t-delay
        q=Q.copy()
        if pulsed:
            factor=168. if (u-ready+1)%168==0 else 0.
            for k in range(4):q[k,k]*=factor
        m=F@m;P=F@P@F.T+q
        if valid[t]:
            h[:4]=phi[t]
            mean=A@m
            cov=A@P@A.T+S
            # F is identity on coefficient states. Count scheduled pulses
            # exactly; the residual forecast noise is still the precomputed S.
            if pulsed and delay:
                pulses=(u+delay-ready+1)//168-(u-ready+1)//168
                for k in range(4):cov[k,k]+=(168*pulses-delay)*Q[k,k]
            pred[t]=h@mean;var[t]=h@cov@h+r
        if (query[u] or initmask[u]) and valid[u] and np.isfinite(y[u]):
            h[:4]=phi[u];ph=P@h;s=r+h@ph
            m+=ph*(y[u]-h@m)/s;P-=np.outer(ph,ph)/s;P=.5*(P+P.T)
    return pred,var

def parameterize(model,f,index,variant):
    m=model['m'].copy();P=model['P'].copy();F=np.diag(model['F']);Q=np.diag(model['Q']);r=model['r']
    details={}
    if variant=='ar2':
        end=int(f['ready'][index]);start=max(0,end-336)
        e=f['y'][start:end]-model['phi'][start:end]@m[:4]
        ok=model['valid'][start:end]&np.isfinite(e)
        good=ok[2:]&ok[1:-1]&ok[:-2]
        coeff=np.array([model['rho'],0.])
        if good.sum()>=24:
            x=np.column_stack([e[1:-1],e[:-2]])[good];target=e[2:][good]
            coeff=np.linalg.solve(x.T@x+np.eye(2),x.T@target)
        # Only commissioning data set the stable two-lag process. Scale both
        # coefficients toward zero until the companion spectral radius <= .95.
        for _ in range(1000):
            B=np.array([[coeff[0],coeff[1]],[1.,0.]])
            if np.max(np.abs(np.linalg.eigvals(B)))<=.95:break
            coeff*=.98
        C=solve_discrete_lyapunov(B,np.diag([1.,0.]))
        innovation=model['variance']*.8/C[0,0]
        residual_cov=C*innovation
        m=np.r_[m[:4],0.,0.];Pnew=np.zeros((6,6));Pnew[:4,:4]=P[:4,:4];Pnew[4:,4:]=residual_cov;P=Pnew
        F=np.eye(6);F[4:,4:]=B
        Qnew=np.zeros((6,6));Qnew[:4,:4]=Q[:4,:4];Qnew[4,4]=innovation;Q=Qnew
        details=dict(ar2_coefficients=coeff.tolist(),commissioning_triples=int(good.sum()),
            spectral_radius=float(np.max(np.abs(np.linalg.eigvals(B)))),marginal_residual_variance=float(residual_cov[0,0]))
    elif variant=='drift_low':Q[:4,:4]*=.1
    elif variant=='drift_high':Q[:4,:4]*=10.
    return m,P,F,Q,r,details

def self_test():
    from calibration_engine import initialize
    rng=np.random.default_rng(610);n=520;ready=96
    X=rng.normal(size=(n,3));X[:,0]+=20;y=X[:,0]*1.3+rng.normal(size=n)
    model=initialize(X,y,ready,q_scale=.001);init=np.arange(n)<ready
    query=np.zeros(n,bool);query[100:112]=True;query[240:252]=True;init[300:306]=True
    for delay in (0,24,72,168):
        result=original.run_site_fixed([model],y,[ready],init,query,delay=delay)
        p,v=replay(model['phi'],model['valid'],y,init,query,model['m'],model['P'],np.diag(model['F']),np.diag(model['Q']),model['r'],ready,delay,False)
        np.testing.assert_allclose(p,result['pred'][0],equal_nan=True,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(v,result['var'][0],equal_nan=True,rtol=2e-12,atol=2e-12)
        changed=y.copy();changed[180:]+=1000
        pc,_=replay(model['phi'],model['valid'],changed,init,query,model['m'],model['P'],np.diag(model['F']),np.diag(model['Q']),model['r'],ready,delay,False)
        np.testing.assert_allclose(pc[:180+delay+1],p[:180+delay+1],equal_nan=True,rtol=0,atol=0)
    # Independent hour-by-hour propagation checks the pulsed-Q forecast sum.
    F=np.diag(model['F']);Q=np.diag(model['Q']);P=model['P'].copy()
    for u in (ready,ready+95,ready+166,ready+170):
        for delay in (24,72,168):
            expected=P.copy()
            for step in range(1,delay+1):
                q=Q.copy()
                factor=168. if (u+step-ready+1)%168==0 else 0.
                q[:4,:4]*=factor
                expected=F@expected@F.T+q
            A=np.linalg.matrix_power(F,delay);S=np.zeros_like(P)
            for _ in range(delay):S=F@S@F.T+Q
            actual=A@P@A.T+S
            pulses=(u+delay-ready+1)//168-(u-ready+1)//168
            for k in range(4):actual[k,k]+=(168*pulses-delay)*Q[k,k]
            np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
    (OUT/'self_test.json').write_text(json.dumps(dict(status='PASS',ar1_fixed_delay_prediction_and_variance_parity=True,
        late_target_cannot_change_past_output=True,pulsed_forecast_independent_hourly_parity=True,
        delays=[0,24,72,168],synthetic_only=True),indent=2),encoding='utf-8')
    print('MODEL SENSITIVITY SELF TEST PASS',flush=True)

def run_all():
    rows=[];parameters=[]
    for site in ('Manchester','London','York'):
        f=make_site(site,**CONFIG['calibrator'])
        for delay in (0,72):
            policies=('periodic','original') if delay==0 else ('periodic','original','aware_retained')
            for policy in policies:
                base=load_run(ROOT/'results/revision6/fixed_delay'/f'{site}_B48_{policy}_d{delay}_shared1.npz')
                for variant in VARIANTS:
                    name=f'{variant}_{policy}_d{delay}'
                    path=OUT/f'{site}_B48_{name}.npz'
                    if not path.exists():
                        pred=np.full_like(base.pred,np.nan);var=pred.copy();started=time.perf_counter()
                        for i,model in enumerate(f['models']):
                            m,P,F,Q,r,detail=parameterize(model,f,i,variant)
                            pred[i],var[i]=replay(model['phi'],model['valid'],f['y'],f['initialization_mask'],base.query,
                                m,P,F,Q,r,int(f['ready'][i]),delay,variant=='drift_pulsed')
                            if variant=='ar2':parameters.append(dict(site=site,segment=i,**detail))
                        pred=np.maximum(pred,0.)
                        assert np.isfinite(pred[base.score_masks]).all()
                        if variant=='ar1':np.testing.assert_allclose(pred,base.pred,equal_nan=True,rtol=2e-11,atol=2e-11)
                        np.savez_compressed(path,pred=pred,var=var,y=base.y,times=np.asarray(base.times.astype(str),dtype='U19'),
                            score_masks=base.score_masks,query=base.query,initmask=base.initmask,metadata_json=json.dumps(base.metadata),
                            revision_provenance_json=json.dumps(dict(variant=variant,delay=delay,policy=policy,
                                acquisition='transferred AR1 calendar; no alternative-model optimization',runtime_seconds=time.perf_counter()-started)))
                    run=load_run(path);rows.append(dict(site=site,delay=delay,policy=policy,variant=variant,
                        rmse=macro(run.pred,run.y,run.score_masks,run.metadata)))
                    print('MODEL',json.dumps(rows[-1]),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'site_metrics.csv',index=False)
    if parameters:pd.DataFrame(parameters).drop_duplicates(['site','segment']).to_csv(OUT/'ar2_commissioning_parameters.csv',index=False)
    all_rows=[]
    for delay in (0,72):
        policies=('periodic','original') if delay==0 else ('periodic','original','aware_retained')
        for variant in VARIANTS:
            methods=tuple(f'{variant}_{policy}_d{delay}' for policy in policies)
            bundles=[build_site_bundle({m:load_run(OUT/f'{site}_B48_{m}.npz') for m in methods},methods=methods)
                for site in ('Manchester','London','York')]
            metrics=metric_tables(bundles)['aggregate_metrics'];metrics['variant']=variant;metrics['delay']=delay
            all_rows.append(metrics)
    pd.concat(all_rows,ignore_index=True).to_csv(OUT/'aggregate_metrics.csv',index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True);EV.mkdir(parents=True,exist_ok=True)
    spec=dict(stage='exploratory prespecified sixth-revision sensitivity; evaluation records previously seen',
        variants=list(VARIANTS),delays=[0,72],support='common ready+168',calendars='frozen AR1 acquisition; no alternative-model parameter search',
        ar2='two-lag ridge fit on commissioning residuals; roots projected within .95; marginal residual variance preserved at .8 commissioning variance',
        drift_low_high='coefficient process variance multiplied by .1 or 10; residual unchanged',
        drift_pulsed='zero coefficient process variance between 168h boundaries and 168x at the boundary; same cumulative variance per complete block; residual unchanged',
        claims='robustness of correction and transferred-calendar contrasts; not optimized AR2 acquisition or actual physical change points')
    p=EV/'model_sensitivity_protocol.json'
    if p.exists():assert json.loads(p.read_text())==spec
    else:p.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    self_test();run_all()

if __name__=='__main__':main()
