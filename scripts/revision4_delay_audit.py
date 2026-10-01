"""Independent fourth-revision delay checks and untuned-planner sensitivity."""
from pathlib import Path
import os,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'cache/numba'))
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from revision4_delay import OUT,CONFIG,make_site,run
import revision4_delay_controller as c
from statistical_analysis import load_run,build_site_bundle,paired_calendar_bootstrap,metric_tables

def independent_utility_check():
    # Three complete historical scenarios, queued paid and free events, known
    # validity, and a future commissioning mask. Pure NumPy matrix operations
    # implement a second covariance engine; no shared Kalman helper is used.
    rng=np.random.default_rng(404);n=800;t=500;D=72;H=168;p=5
    phi=rng.normal(size=(1,n,p-1));valid=np.ones((1,n),bool);ready=np.array([50])
    A=rng.normal(size=(p,p));P=(A@A.T)[None]
    F=np.array([[1.,1.,1.,1.,.8]]);Q=np.array([[.01,.01,.01,.01,.2]])
    R=np.array([.4]);weights=np.ones(1);im=np.zeros(n,bool);query=np.zeros(n,bool)
    query[440:452]=True;im[470:476]=True;im[510:517]=True
    lengths=np.array([12])
    got=c._delay_utilities(phi,valid,ready,P,F,Q,R,weights,im,query,t,lengths,H,D)[0]
    expected=0.;Fd=np.diag(F[0]);scale=np.diag(F[0]**D)
    def update(P,h):
        ph=P@h;return P-np.outer(ph,ph)/(R[0]+h@ph)
    for lag in (H+24,H+48,H+120):
        origin=t-lag;base=P[0].copy();alt=base.copy();paid=0
        for j in range(H+1):
            if j>0:
                base=Fd@base@Fd.T+np.diag(Q[0]);alt=Fd@alt@Fd.T+np.diag(Q[0])
            hf=np.r_[phi[0,origin+j] if j>0 else phi[0,t],1.]
            if j>0:expected+=max(float(hf@scale@(base-alt)@scale.T@hf),0.)/(H*3)
            e=t-D+j
            h=np.r_[phi[0,e] if e<t else phi[0,origin-D+j] if e>t else phi[0,t],1.]
            common=im[e] or (e<t and query[e])
            acquired=e>=t and not im[e] and paid<12
            if common:base=update(base,h)
            if common or acquired:alt=update(alt,h)
            if acquired:paid+=1
    np.testing.assert_allclose(got,expected,rtol=1e-10,atol=1e-10)
    return dict(pending_and_shared_setup_matrix_rollout=True,independent_utility=expected)

def sensitivity():
    plan=dict(status='Post-main-result sensitivity, not used for model selection',purpose='Isolate the delay-aware utility while retaining the original immediate horizon/threshold',horizon=168,threshold_scale=1.25,delay=72,all_sites=True)
    path=OUT/'untuned_sensitivity_protocol.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for site in CONFIG['sites']:
        dest=OUT/f'{site}_B48_delay_aware72_untuned.npz'
        if dest.exists():continue
        f=make_site(site,**CONFIG['calibrator']);z=run(c,f,168,1.25)
        with np.load(OUT/f'{site}_B48_delay_aware72.npz',allow_pickle=False) as src:
            pack={k:src[k].copy() for k in src.files if k in ('y','times','metadata_json','warning_mask','quality_flag_mask','y_duplicate','duplicate_valid','score_masks','initmask')}
        pack.update(pred=np.maximum(z['pred'],0.),query=z['query'])
        for k in ('var','start_mask','forced_query','chosen_lengths','candidate_scores','block_spent','block_target','budget_shortfall'):pack[k]=z[k]
        pack['access_json']=json.dumps(dict(additional_attempts=z['additional_attempts'],initialization_attempts=z['initialization_attempts'],runtime_seconds=z['runtime_seconds']))
        pack['revision_provenance_json']=json.dumps(plan)
        np.savez_compressed(dest,**pack)
    methods=('periodic_delay0','risk_fixed_delay0','periodic_delay72','risk_fixed_delay72','delay_aware72_untuned','delay_aware72')
    bundles=[]
    for site in CONFIG['sites']:
        runs={m:load_run((OUT if m.startswith('delay_aware') else ROOT/'results/revision/access_v2')/f'{site}_B48_{m}.npz') for m in methods}
        bundles.append(build_site_bundle(runs,methods=methods))
    for name,frame in metric_tables(bundles).items():frame.to_csv(OUT/f'extended_{name}.csv',index=False)
    contrasts={'untuned_aware_minus_original72':{4:1,3:-1},'untuned_aware_minus_periodic72':{4:1,2:-1},'selected_minus_untuned':{5:1,4:-1},'selected_advantage_change_vs_immediate':{5:1,2:-1,1:-1,0:1}}
    rows=[]
    for days,reps in ((7,1000),(14,300)):
        result=paired_calendar_bootstrap(bundles,n_replicates=reps,block_days=days,seed=20261001+days)
        for name,co in contrasts.items():
            vec=np.array([co.get(i,0) for i in range(len(methods))]);samples=result['primary']@vec;lo,hi=np.quantile(samples,[.025,.975])
            rows.append(dict(contrast=name,difference=float(result['observed_primary']@vec),ci_low=lo,ci_high=hi,block_days=days))
    pd.DataFrame(rows).to_csv(OUT/'extended_contrasts.csv',index=False)
    return bundles

def numerical_audit(bundles):
    checks=independent_utility_check();details=[]
    for site in CONFIG['sites']:
        f=make_site(site,**CONFIG['calibrator'])
        selected=json.loads((OUT/'selected.json').read_text())
        with np.load(OUT/f'{site}_B48_delay_aware72.npz',allow_pickle=False) as z:
            source={k:z[k].copy() for k in z.files}
        rerun=run(c,f,selected['horizon'],selected['threshold_scale'])
        for key in ('pred','query','candidate_scores'):
            expected=np.maximum(rerun[key],0.) if key=='pred' else rerun[key]
            np.testing.assert_allclose(source[key],expected,rtol=0,atol=0,equal_nan=True)
        # A fixed calendar is a second implementation of delayed state replay.
        fixed=c.run_site_fixed(f['models'],f['y'],f['ready'],f['initialization_mask'],source['query'],delay=72)
        np.testing.assert_allclose(source['pred'],np.maximum(fixed['pred'],0.),rtol=0,atol=0,equal_nan=True)
        assert np.array_equal(rerun['block_spent'],rerun['block_feasible']) and (rerun['starts']<=4).all()
        details.append(dict(site=site,exact_from_data_rerun=True,fixed_calendar_prediction_parity=True,additional_attempts=int(source['query'].sum()),forced_attempts=int(source['forced_query'].sum()),starts=int(source['start_mask'].sum()),finite_scored_predictions=bool(np.isfinite(source['pred'][source['score_masks']]).all())))
    table=pd.read_csv(OUT/'extended_aggregate_metrics.csv')
    for b in bundles:
        for k,m in enumerate(b['methods']):
            for i,ident in enumerate(b['instruments']):
                mask=b['active'][i];val=float(np.sqrt(b['se'][k,i,mask].mean()))
                entry=pd.read_csv(OUT/'extended_site_instrument_metrics.csv')
                saved=entry.loc[entry.site.eq(b['site'])&entry.instrument.eq(ident)&entry.method.eq(m),'rmse'].iloc[0]
                np.testing.assert_allclose(val,saved,rtol=1e-13,atol=1e-13)
    checks.update(status='PASS',sites=details,instrument_hours=int(sum(b['active'].sum() for b in bundles)),all_extended_instrument_metrics_reaggregated=True,controller_sha256=hashlib.sha256((ROOT/'scripts/revision4_delay_controller.py').read_bytes()).hexdigest())
    (OUT/'independent_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':
    numerical_audit(sensitivity())
