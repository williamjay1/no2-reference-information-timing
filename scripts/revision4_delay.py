"""Exploratory delay-aware reference-window redesign, fourth revision.

All tuning uses pre-2021 Manchester. Original artifacts remain immutable.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
for key,value in {'NUMBA_CACHE_DIR':ROOT/'cache/numba','MPLCONFIGDIR':ROOT/'temp/mpl','TEMP':ROOT/'temp','TMP':ROOT/'temp'}.items():
    os.environ.setdefault(key,str(value))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,copy,hashlib,json,time
import numpy as np
import pandas as pd
from numba import njit
from site_data import make_site
from calibration_engine import initialize
import revision_access_controller as prior
from revision_access import CONFIG,macro
from statistical_analysis import load_run,build_site_bundle,paired_calendar_bootstrap,metric_tables

OUT=ROOT/'results/revision4/delay'
EVIDENCE=ROOT/'evidence'

UTILITY=r'''
@njit(cache=True)
def _delay_utilities(phi, valid, ready, anchors_P, Fs, Qs, rs, weights,
                     initmask, query, t, lengths, horizon, delay):
    # Event state is t-delay BEFORE the reference arriving after output t.
    # Pending requests are common to both paths. A candidate event at t+k
    # first affects the prediction at t+k+delay+1.
    nd,n,features=phi.shape
    p=features+1
    utilities=np.zeros(len(lengths))
    active_weight=0.
    lag0=max(72,horizon+24)
    lags=(lag0,lag0+24,lag0+96)
    for d in range(nd):
        if t-delay<ready[d] or not valid[d,t]:
            continue
        active_weight+=weights[d]
        F=Fs[d];Q=Qs[d]
        scales=F**delay
        scenarios=0
        for lag in lags:
            origin=t-lag
            if origin-delay>=0 and origin+horizon<t:
                scenarios+=1
        nscenarios=max(scenarios,1)
        for scenario in range(nscenarios):
            origin=-1;seen=0
            for lag in lags:
                candidate=t-lag
                if candidate-delay>=0 and candidate+horizon<t:
                    if seen==scenario:origin=candidate
                    seen+=1
            base=anchors_P[d].copy()
            alternatives=np.empty((len(lengths),p,p))
            counts=np.zeros(len(lengths),np.int64)
            for a in range(len(lengths)):alternatives[a]=base
            h=np.ones(p);hf=np.ones(p)
            for j in range(horizon+1):
                if j>0:
                    _cov_advance(base,F,Q)
                    for a in range(len(lengths)):_cov_advance(alternatives[a],F,Q)
                # Forecast covariance F^delay P (F^delay)' plus common Q.
                # Q cancels in the variance difference. Future contexts use
                # complete historical analogs, never future sensor values.
                for v in range(features):
                    idx=origin+j
                    hf[v]=phi[d,idx,v] if j>0 and origin>=0 and valid[d,idx] else phi[d,t,v]
                hs=hf*scales
                vb=_quadratic(base,hs)
                if j>0:
                    for a in range(len(lengths)):
                        utilities[a]+=weights[d]*max(vb-_quadratic(alternatives[a],hs),0.)/(horizon*nscenarios)
                event=t-delay+j
                if event<ready[d]:continue
                free=event<n and initmask[event]
                pending=event<t and event>=0 and query[event]
                if event<t:
                    # Recorded past feature validity is known, unlike its
                    # still-withheld reference concentration.
                    if event<0 or not valid[d,event]:continue
                    for v in range(features):h[v]=phi[d,event,v]
                else:
                    idx=origin-delay+j
                    for v in range(features):
                        h[v]=phi[d,idx,v] if event>t and origin>=0 and valid[d,idx] else phi[d,t,v]
                common=free or pending
                for a in range(len(lengths)):
                    paid=event>=t and not free and counts[a]<lengths[a]
                    if common or paid:_cov_observe(alternatives[a],h,rs[d])
                    if paid:counts[a]+=1
                if common:_cov_observe(base,h,rs[d])
    if active_weight:utilities/=active_weight
    return utilities
'''

def write_controller():
    source=(ROOT/'scripts/revision_access_controller.py').read_text(encoding='utf-8')
    old='utilities = _window_utilities(phi, valid, planning_ready, Ps, Fs,\n                                                      Qs, rs, weights, planning_initmask, t, lengths,\n                                                      horizon, policy)'
    new='utilities = (_window_utilities(phi, valid, planning_ready, Ps, Fs, Qs, rs, weights, planning_initmask, t, lengths, horizon, policy) if delay == 0 else _delay_utilities(phi, valid, ready, anchors_P, Fs, Qs, rs, weights, planning_initmask, query, t, lengths, horizon, delay))'
    assert source.count(old)==1
    source=source.replace(old,new)
    source=source.replace('@njit(cache=True)\ndef _past_quantile',UTILITY+'\n\n@njit(cache=True)\ndef _past_quantile',1)
    source='"""Delay-aware covariance redesign; generated from immutable prior controller.\nIncludes pending references and actual future arrival order in the planner.\n"""\n'+source[source.index('from __future__'):]
    path=ROOT/'scripts/revision4_delay_controller.py'
    if path.exists():assert path.read_text(encoding='utf-8')==source
    else:path.write_text(source,encoding='utf-8')
    return path

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(stage='exploratory fourth revision after previous evaluation results were known',delay_hours=72,
        tuning='Pre-2021 Manchester only; lowest development macro physical-instrument RMSE, ties lexicographic horizon/threshold',
        candidates=[dict(horizon=h,threshold_scale=s) for h in (168,240) for s in (1.,1.25)],
        controls='Same calibrator, 12h commitments, B48 per 672h, four starts, common catch-up and shared commissioning',
        objective='Mean predicted variance reduction over H future predictions, conditional on pending requests and 72h arrivals; three strictly historical context analogs or current-context fallback',
        planner_assumptions='Attempted references assumed usable; past device validity known; future reference/device availability never consulted; future external commissioning calendar known',
        score_support='Original all-delay common supplied masks, ready+168; 18 instruments and 59674 instrument-hours',
        uncertainty=dict(days=[7,14],replicates=[1000,300],seed_rule='20261001+days',scope='conditional realized errors, same sites and calendar quarters, no policy retraining'),
        limits='No new untouched dataset; exploratory extension. Longer horizon and threshold are both tuned only in development. No electrode-identifiability claim.',
        old_controller_sha256=hashlib.sha256((ROOT/'scripts/revision_access_controller.py').read_bytes()).hexdigest())
    p=EVIDENCE/'revision4_delay_protocol.json'
    if p.exists():assert json.loads(p.read_text())==design
    else:p.write_text(json.dumps(design,indent=2),encoding='utf-8')
    return design

def self_test(controller):
    rng=np.random.default_rng(4);n=560;ready=96;delay=72
    X=rng.normal(size=(n,3));X[:,0]+=20
    y=X[:,0]*1.5+rng.normal(size=n)
    model=initialize(X,y,ready,q_scale=.001)
    init=np.arange(n)<ready
    args=dict(budget=24,block=224,max_starts=4,policy='risk_fixed',horizon=168,threshold_scale=1.25,fixed_length=12)
    zero=controller.run_site([model],y,[ready],init,delay=0,**args)
    old=prior.run_site([model],y,[ready],init,delay=0,**args)
    for key in ('pred','query','candidate_scores'):
        np.testing.assert_allclose(zero[key],old[key],equal_nan=True,rtol=0,atol=0)
    result=controller.run_site([model],y,[ready],init,delay=delay,**args)
    hidden=y.copy();hidden[~(result['query']|init)]=rng.normal(100,30,size=(~(result['query']|init)).sum())
    hidden_run=controller.run_site([model],hidden,[ready],init,delay=delay,**args)
    for key in ('pred','query','candidate_scores'):
        np.testing.assert_allclose(result[key],hidden_run[key],equal_nan=True,rtol=0,atol=0)
    # Entire earlier trajectory invariant to future sensor and future labels.
    changed=copy.deepcopy(model);at=300
    changed['phi'][at+1:]*=20
    future_y=y.copy();future_y[at+1:]+=100
    future=controller.run_site([changed],future_y,[ready],init,delay=delay,**args)
    for key in ('pred','query','candidate_scores'):
        np.testing.assert_allclose(result[key][...,:at+1],future[key][...,:at+1],equal_nan=True,rtol=0,atol=0)
    # Utility equals independent delayed event-covariance replay, current
    # fallback context, with no pending/free targets and one candidate.
    from calibration_engine import _advance,_observe
    packed=prior._prepare([model],y,[ready],init)
    phi,valid,rd,ms,Ps,Fs,Qs,rs,yy,im=packed
    t=170;H=168;L=12;P=Ps[0].copy()
    observed=controller._delay_utilities(phi,valid,rd,Ps,Fs,Qs,rs,np.ones(1),np.zeros(n,bool),np.zeros(n,bool),t,np.array([L]),H,delay)[0]
    base=P.copy();alt=P.copy();m=np.zeros(len(Fs[0]));h=np.r_[phi[0,t],1.];expected=0.
    for j in range(H+1):
        if j>0:
            _advance(m,base,Fs[0],Qs[0]);_advance(m,alt,Fs[0],Qs[0])
        pb=base.copy();pa=alt.copy()
        for k in range(delay):
            _advance(m,pb,Fs[0],Qs[0]);_advance(m,pa,Fs[0],Qs[0])
        if j>0:expected+=max(float(h@pb@h-h@pa@h),0.)/H
        if delay<=j<delay+L:_observe(m,alt,h,0.,rs[0])
    np.testing.assert_allclose(observed,expected,rtol=1e-10,atol=1e-10)
    assert np.array_equal(result['block_spent'],result['block_feasible']) and max(result['starts'])<=4
    checks=dict(status='PASS',zero_delay_exact_prior_parity=True,hidden_target_invariance=True,future_sensor_and_target_prefix_invariance=True,independent_covariance_rollout=True,quota_and_start_cap=True,utility_value=float(observed))
    (OUT/'self_test.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks),flush=True)

def run(controller,f,h,s):
    return controller.run_site(f['models'],f['y'],f['ready'],f['initialization_mask'],budget=48,block=672,max_starts=4,policy='risk_fixed',horizon=h,threshold_scale=s,fixed_length=12,delay=72,shared_setup=True)

def development(controller,design):
    path=OUT/'selected.json'
    if path.exists():return json.loads(path.read_text())
    f=make_site('Manchester',stage='development',**CONFIG['calibrator'])
    mask=f['score_masks']&(np.arange(len(f['y']))[None,:]>=f['ready'][:,None]+168)
    rows=[]
    for c in design['candidates']:
        z=run(controller,f,c['horizon'],c['threshold_scale'])
        row=dict(**c,rmse=macro(np.maximum(z['pred'],0.),f['y'],mask,f['metadata']),runtime_seconds=z['runtime_seconds'],attempts=int(z['query'].sum()))
        rows.append(row);print('DEVELOPMENT',json.dumps(row),flush=True)
        np.savez_compressed(OUT/f"development_H{c['horizon']}_T{c['threshold_scale']}.npz",pred=np.maximum(z['pred'],0.),query=z['query'],score_masks=mask,y=f['y'],times=f['times'].astype(str),metadata_json=json.dumps(f['metadata']))
    pd.DataFrame(rows).to_csv(OUT/'development_candidates.csv',index=False)
    selected=min(rows,key=lambda r:(r['rmse'],r['horizon'],r['threshold_scale']))
    selected.update(selection_data_end_exclusive='2021-01-01',test_results_used_for_selection=False,exploratory_revision=True)
    path.write_text(json.dumps(selected,indent=2),encoding='utf-8')
    return selected

def evaluate(controller,selected):
    summaries=[]
    for site in CONFIG['sites']:
        path=OUT/f'{site}_B48_delay_aware72.npz'
        if path.exists():continue
        f=make_site(site,**CONFIG['calibrator'])
        z=run(controller,f,selected['horizon'],selected['threshold_scale'])
        with np.load(ROOT/'results/revision/access_v2'/f'{site}_B48_periodic_delay72.npz',allow_pickle=False) as src:
            pack={k:src[k].copy() for k in src.files if k in ('y','times','metadata_json','warning_mask','quality_flag_mask','y_duplicate','duplicate_valid','score_masks')}
        np.testing.assert_array_equal(pack['y'],f['y'])
        pack.update(pred=np.maximum(z['pred'],0.),query=z['query'],initmask=f['initialization_mask'])
        for k in ('var','start_mask','forced_query','chosen_lengths','candidate_scores','block_spent','block_target','budget_shortfall'):pack[k]=z[k]
        pack['revision_provenance_json']=json.dumps(dict(selected=selected,controller_sha256=hashlib.sha256((ROOT/'scripts/revision4_delay_controller.py').read_bytes()).hexdigest()))
        pack['access_json']=json.dumps(dict(initialization_attempts=z['initialization_attempts'],additional_attempts=z['additional_attempts'],runtime_seconds=z['runtime_seconds']))
        np.savez_compressed(path,**pack)
        row=dict(site=site,rmse=macro(pack['pred'],f['y'],pack['score_masks'],f['metadata']),runtime_seconds=z['runtime_seconds'],attempts=z['additional_attempts'])
        summaries.append(row);print('EVALUATION',json.dumps(row),flush=True)
    if summaries:pd.DataFrame(summaries).to_csv(OUT/'site_runtime.csv',index=False)

def analyze():
    methods=('periodic_delay72','risk_fixed_delay72','delay_aware72')
    bundles=[]
    for site in CONFIG['sites']:
        runs={m:load_run((OUT if m=='delay_aware72' else ROOT/'results/revision/access_v2')/f'{site}_B48_{m}.npz') for m in methods}
        bundles.append(build_site_bundle(runs,methods=methods))
    tables=metric_tables(bundles)
    for name,frame in tables.items():frame.to_csv(OUT/f'{name}.csv',index=False)
    rows=[];diag=[]
    assert sum(b['active'].sum() for b in bundles)==59674
    for days,reps in ((7,1000),(14,300)):
        result=paired_calendar_bootstrap(bundles,n_replicates=reps,block_days=days,seed=20261001+days)
        diag.append(result['diagnostics'])
        for a,b in ((2,0),(2,1),(1,0)):
            sample=result['primary'][:,a]-result['primary'][:,b]
            ci=np.quantile(sample,[.025,.975])
            rows.append(dict(method=methods[a],comparator=methods[b],difference=float(result['observed_primary'][a]-result['observed_primary'][b]),ci_low=float(ci[0]),ci_high=float(ci[1]),block_days=days))
    pd.DataFrame(rows).to_csv(OUT/'contrasts.csv',index=False)
    (OUT/'bootstrap_diagnostics.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['test','run','analyze'],default='run');args=ap.parse_args()
    design=freeze();write_controller()
    import revision4_delay_controller as controller
    if args.stage=='test':self_test(controller)
    elif args.stage=='analyze':analyze()
    else:
        self_test(controller);selected=development(controller,design);evaluate(controller,selected);analyze()
