"""Causal closed-loop request decisions under a misspecified delivery contract.

Actual random releases drive the estimator, while an explicitly approximate
planner works with a fixed 72-hour contract. This is not distribution-aware
optimal planning. Original controllers and streams remain immutable.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
for key,value in {'NUMBA_CACHE_DIR':ROOT/'cache/numba','MPLCONFIGDIR':ROOT/'temp/mpl',
                  'TEMP':ROOT/'temp','TMP':ROOT/'temp'}.items():
    os.environ.setdefault(key,str(value))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse,json,hashlib
from time import perf_counter
import numpy as np
import pandas as pd
from numba import njit
from calibration_engine import _advance,_observe,initialize
import revision_access_controller as original
from revision4_delay_controller import _delay_utilities
from revision_access_controller import _window_utilities,_past_quantile
from revision6_random_delivery import replay,DELAYS,PROBABILITIES,SEEDS
from site_data import make_site
from revision_access import CONFIG
from statistical_analysis import load_run,build_site_bundle,metric_tables

OUT=ROOT/'results/revision6/online_delivery_corrected'
EV=ROOT/'evidence/revision6'
SITES=('Manchester','London','York')
POLICIES=('periodic','original_online','aware_online')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def potential_schedule(delays):
    """Schedule all potential events; only requested/commissioning events release."""
    n=len(delays);events=np.arange(n,dtype=np.int64);arrival=events+delays
    keep=arrival<n;events,arrival=events[keep],arrival[keep]
    order=np.lexsort((events,arrival));events,arrival=events[order],arrival[order]
    offsets=np.r_[0,np.cumsum(np.bincount(arrival,minlength=n))].astype(np.int64)
    return events,offsets


@njit(cache=True)
def _closed_loop(phi,valid,ready,m0,P0,Fs,Qs,rs,y,initmask,weights,
                 event_order,release_offsets,max_delay,block_budgets,
                 remaining_slots,origin,block,max_starts,horizon,threshold_scale,
                 fixed_length,assumed_delay,aware,availability_gate):
    nd,n,features=phi.shape;p=features+1
    width=max(max_delay,assumed_delay)+2
    means=np.zeros((nd,width,p));covs=np.zeros((nd,width,p,p))
    ms,Ps=m0.copy(),P0.copy()
    received=np.zeros(n,np.bool_);success=np.zeros(n,np.bool_)
    pred=np.full((nd,n),np.nan);variance=pred.copy()
    query=np.zeros(n,np.bool_);start_mask=query.copy();forced=query.copy()
    chosen_lengths=np.zeros(n,np.int64);candidate_scores=np.full(n,np.nan)
    spent=np.zeros(len(block_budgets),np.int64);starts=spent.copy()
    anchors=np.zeros((nd,p,p));h=np.ones(p)
    # Availability is determined by the declared contract upper bound, never
    # by looking up an as-yet-unobserved realized commissioning delay.
    planning_ready=ready+availability_gate
    working_ready=ready+max(0,availability_gate-assumed_delay)
    remaining_window=0;forced_window=False;cooldown=False;replay_steps=0
    for t in range(n):
        active=0
        for d in range(nd):
            if t<ready[d]:continue
            _advance(ms[d],Ps[d],Fs[d],Qs[d])
            if valid[d,t] and t>=planning_ready[d]:
                h[:-1]=phi[d,t]
                # These predictions use the actual release history, not the
                # working fixed-delay contract used by the request planner.
                pred[d,t]=h@ms[d]
                variance[d,t]=h@Ps[d]@h+rs[d]
                if t>=planning_ready[d]:active+=1
            means[d,t%width]=ms[d];covs[d,t%width]=Ps[d]
        if t>=origin:
            bindex=(t-origin)//block
            if (t-origin)%block==0:
                remaining_window=0;forced_window=False;cooldown=False
            remaining_budget=block_budgets[bindex]-spent[bindex]
            available_slots=remaining_slots[t]
            if not initmask[t] and remaining_budget>0:
                execute=False
                must_catch_up=remaining_budget>=available_slots
                if remaining_window>0:
                    execute=True
                elif must_catch_up:
                    remaining_window=remaining_budget;forced_window=True
                    chosen_lengths[t]=remaining_window
                    if starts[bindex]<max_starts:
                        start_mask[t]=True;starts[bindex]+=1
                    execute=True
                elif cooldown:
                    cooldown=False
                elif starts[bindex]<max_starts and active:
                    lengths=np.array([min(fixed_length,remaining_budget)])
                    if aware:
                        anchor=t-assumed_delay
                        # The anchor is a forward-filter prior at event time
                        # t-72 using only received event<=anchor-1 labels.
                        # Later received events are intentionally omitted here
                        # and represented as common working-contract events in
                        # _delay_utilities. No backwards smoother is claimed.
                        for d in range(nd):
                            if anchor<ready[d]:
                                anchors[d]=P0[d]
                            else:
                                if anchor==ready[d]:anchors[d]=P0[d]
                                else:anchors[d]=covs[d,(anchor-1)%width]
                                dummy=np.zeros(p)
                                _advance(dummy,anchors[d],Fs[d],Qs[d])
                        utility=_delay_utilities(phi,valid,working_ready,anchors,Fs,Qs,rs,weights,
                            initmask,query,t,lengths,horizon,assumed_delay)[0]
                    else:
                        # Original instantaneous objective, now using the
                        # current posterior obtained from actual releases.
                        utility=_window_utilities(phi,valid,planning_ready,Ps,Fs,Qs,rs,
                            weights,initmask,t,lengths,horizon,4)[0]
                    startup_shadow=block_budgets[bindex]/max(max_starts,1)
                    best=utility/(lengths[0]+startup_shadow)
                    candidate_scores[t]=best
                    rate=min(.95,max(.001,remaining_budget/max(available_slots,1)))
                    threshold=_past_quantile(candidate_scores,t,336,1.-rate,best)
                    if best>=threshold_scale*threshold and best>0.:
                        ell=lengths[0]
                        if starts[bindex]==max_starts-1:
                            ell=remaining_budget;forced_window=ell!=lengths[0]
                        else:forced_window=False
                        remaining_window=min(ell,available_slots)
                        chosen_lengths[t]=remaining_window;start_mask[t]=True
                        starts[bindex]+=1;execute=True
                if execute:
                    query[t]=True;forced[t]=forced_window;spent[bindex]+=1
                    remaining_window-=1
                    if remaining_window==0:cooldown=True;forced_window=False
        # Inspect only this hour's release list, after this hour's decision.
        # Future event lags/labels never enter any request score or decision.
        earliest=n
        for k in range(release_offsets[t],release_offsets[t+1]):
            u=event_order[k]
            if not (query[u] or initmask[u]):continue
            received[u]=True;success[u]=np.isfinite(y[u])
            if success[u]:earliest=min(earliest,u)
        if earliest==n:continue
        for d in range(nd):
            start=max(earliest,ready[d])
            if start>t:continue
            if start==ready[d]:m,P=m0[d].copy(),P0[d].copy()
            else:m,P=means[d,(start-1)%width].copy(),covs[d,(start-1)%width].copy()
            for u in range(start,t+1):
                _advance(m,P,Fs[d],Qs[d])
                if received[u] and valid[d,u] and np.isfinite(y[u]):
                    h[:-1]=phi[d,u];_observe(m,P,h,y[u],rs[d])
                means[d,u%width]=m;covs[d,u%width]=P;replay_steps+=1
            ms[d],Ps[d]=m,P
    return pred,variance,query,success,starts,start_mask,forced,chosen_lengths,candidate_scores,spent,replay_steps


def run_online(models,y,ready,initmask,delays,policy='aware_online',budget=48,
               block=672,max_starts=4,horizon=240,threshold=1.25,assumed=72,availability=168):
    started=perf_counter();packed=original._prepare(models,y,ready,initmask)
    n=len(y);initmask=packed[-1];origin=int(min(packed[2]))
    nb=(n-origin+block-1)//block;targets=np.zeros(nb,np.int64);feasible=targets.copy()
    slots=np.zeros(n,np.int64)
    for k in range(nb):
        left=origin+k*block;right=min(n,left+block)
        targets[k]=int(np.floor(budget*(right-left)/block))
        available=(~initmask[left:right]).astype(np.int64)
        slots[left:right]=np.cumsum(available[::-1])[::-1]
        feasible[k]=min(targets[k],int(available.sum()))
    events,offsets=potential_schedule(delays)
    weights=np.array([m.get('weight',1.) for m in models])
    assert np.min(delays)>=0 and np.max(delays)<=168
    assert availability>=assumed
    output=_closed_loop(*packed,weights,events,offsets,168,feasible,
        slots,origin,block,max_starts,horizon,threshold,12,assumed,policy=='aware_online',availability)
    names=('pred','var','query','success','starts','start_mask','forced_query',
           'chosen_lengths','candidate_scores','block_spent','replay_steps')
    z=dict(zip(names,output))
    assert np.array_equal(z['block_spent'],feasible)
    assert np.max(z['starts'])<=max_starts and not np.any(z['query']&initmask)
    z.update(block_target=targets,block_feasible=feasible,budget_shortfall=targets-z['block_spent'],
             runtime_seconds=perf_counter()-started,event_delays=np.asarray(delays))
    return z


def self_test():
    rng=np.random.default_rng(20261017);n=700;ready=96
    X=rng.normal(size=(n,3));X[:,0]+=20;y=1.5*X[:,0]+rng.normal(size=n)
    model=initialize(X,y,ready,q_scale=.001)
    initmask=np.arange(n)<ready;initmask[260:280]=True
    settings=dict(budget=24,block=224,max_starts=4,horizon=240,threshold=1.25,assumed=72)
    z=run_online([model],y,[ready],initmask,np.full(n,72),availability=72,**settings)
    import revision4_delay_controller as fixed
    old=fixed.run_site([model],y,[ready],initmask,budget=24,block=224,max_starts=4,
        horizon=240,threshold_scale=1.25,fixed_length=12,policy='risk_fixed',delay=72,shared_setup=True)
    np.testing.assert_array_equal(z['query'],old['query'])
    max_error=0.
    for key in ('pred','var'):
        a,b=z[key][:,ready+72:],old[key][:,ready+72:]
        np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-11,equal_nan=True)
        max_error=max(max_error,float(np.nanmax(np.abs(a-b))))
    np.testing.assert_allclose(z['candidate_scores'],old['candidate_scores'],rtol=2e-11,atol=2e-11,equal_nan=True)
    for policy in ('original_online','aware_online'):
        delays=rng.choice(DELAYS,n,p=PROBABILITIES)
        base=run_online([model],y,[ready],initmask,delays,policy=policy,**settings)
        hidden=y.copy();hidden[~(base['query']|initmask)]+=1e4
        changed=run_online([model],hidden,[ready],initmask,delays,policy=policy,**settings)
        for key in ('pred','var','query','candidate_scores'):
            np.testing.assert_allclose(base[key],changed[key],rtol=0,atol=0,equal_nan=True)
        cut=330;future_y=y.copy();future_y[cut+1:]+=1000
        future_delays=delays.copy();future_delays[cut+1:]=168-future_delays[cut+1:]
        changed=run_online([model],future_y,[ready],initmask,future_delays,policy=policy,**settings)
        for key in ('pred','var','query','candidate_scores'):
            np.testing.assert_allclose(base[key][...,:cut+1],changed[key][...,:cut+1],rtol=0,atol=0,equal_nan=True)
        u=int(np.flatnonzero(base['query']&(np.arange(n)+delays<n-1))[0])
        altered=y.copy();altered[u]+=100
        changed=run_online([model],altered,[ready],initmask,delays,policy=policy,**settings)
        release=u+int(delays[u])
        for key in ('pred','var','query','candidate_scores'):
            np.testing.assert_allclose(base[key][...,:release+1],changed[key][...,:release+1],rtol=0,atol=0,equal_nan=True)
        # The exact OOSM estimator agrees with the independently validated
        # fixed-calendar implementation when handed its realized own queries.
        replayed=replay([model],y,[ready],initmask,base['query'],delays)
        for key in ('pred','var'):
            np.testing.assert_allclose(base[key][:,ready+168:],replayed[key][:,ready+168:],rtol=0,atol=0,equal_nan=True)
        assert not np.any(base['query'][:ready+168])
        assert np.isnan(base['pred'][:,:ready+168]).all()
    # A second instrument's fitted prior must not affect a ready instrument's
    # requests before that second prior becomes available. Merely gating the
    # active counter without gating every utility contributor would fail this.
    second=initialize(X,y,200,q_scale=.001)
    two=run_online([model,second],y,[ready,200],initmask,delays,**settings)
    altered_second={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in second.items()}
    altered_second['m']+=5000;altered_second['P']*=100
    altered_two=run_online([model,altered_second],y,[ready,200],initmask,delays,**settings)
    np.testing.assert_array_equal(two['query'][:200+168],altered_two['query'][:200+168])
    np.testing.assert_allclose(two['candidate_scores'][:200+168],altered_two['candidate_scores'][:200+168],
                               rtol=0,atol=0,equal_nan=True)
    value=dict(status='PASS',constant72_aware_query_parity=True,
        constant72_aware_prediction_variance_score_parity=True,maximum_constant72_error=max_error,
        hidden_nonrequested_label_invariance=True,future_label_and_future_delay_prefix_invariance=True,
        current_release_cannot_change_current_output_or_decision=True,quota_and_start_cap=True,
        realized_online_estimator_equals_independent_oosm_replay=True,
        random_fit_availability_gate_hours=168,no_fitted_prior_used_before_available=True,
        unavailable_second_device_prior_cannot_change_requests=True,synthetic_self_tests_only=True)
    write_json(OUT/'self_test.json',value);print('SELF TEST',json.dumps(value),flush=True)


def protocol():
    OUT.mkdir(parents=True,exist_ok=True);EV.mkdir(parents=True,exist_ok=True)
    value=dict(stage='retrospective sixth-revision real-data replay with simulated delivery',
        estimator='Exact chronological refiltering of bounded out-of-sequence releases; all outputs precede current-hour releases',
        random_contract=dict(delays_hours=DELAYS.tolist(),probabilities=PROBABILITIES.tolist(),expected_delay_hours=72.,seeds=list(SEEDS)),
        shared_setup=True,assumed_delay_hours=72,aware_horizon=240,naive_horizon=168,threshold=1.25,
        budget=48,block=672,max_starts=4,fixed_window_length=12,
        support='Original masks intersected with t>=ready+168; same18 instruments/59674 instrument-hours',
        policy_design='Decisions recomputed each hour from actual delivered history; no transferred adaptive query masks',
        aware_working_model='Conservative forward-filter anchor at t-72 prior; received later-event information omitted from anchor and replayed as common working-contract events. Older overdue unreceived events before anchor omitted. Fixed future72h lag assumed.',
        naive_working_model='Instantaneous original variance-reduction objective applied to current legal posterior',
        commissioning='Random-contract fitted priors and every model utility contributor become available only at ready+168, based on the stated contract upper bound. Constant72 validation uses ready+72. No future realized commissioning lags determine eligibility.',
        prior_diagnostic='Pre-fix ready+72 random-gate streams are retained in online_delivery and explicitly fail the commissioning availability audit; not final scientific evidence.',
        limitations=['Misspecified conservative fixed-delay planning, not optimal or exact distribution-aware planning.',
            'Simulated release contracts are not observed field arrival logs.',
            'Conditional three-seed variability is not deployment/generalization uncertainty.',
            'Replay is retrospective, with no fresh-site or prospective confirmation.'],
        source_sha256={name:sha(ROOT/'scripts'/name) for name in ('revision_access_controller.py','revision4_delay_controller.py','calibration_engine.py','revision6_random_delivery.py')})
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==value
    else:write_json(path,value)
    return value


def evaluate():
    rows=[];parity=[]
    for site_index,site in enumerate(SITES):
        f=make_site(site,**CONFIG['calibrator']);n=len(f['y'])
        mask=f['score_masks']&(np.arange(n)[None,:]>=f['ready'][:,None]+168)
        # Same physical constant-contract algorithm is tested on real data.
        constant=run_online(f['models'],f['y'],f['ready'],f['initialization_mask'],np.full(n,72),availability=72)
        with np.load(ROOT/'results/revision6/fixed_delay'/f'{site}_B48_aware_retained_d72_shared1.npz') as a:
            np.testing.assert_array_equal(constant['query'],a['query'])
            for key in ('pred','var'):
                v=np.maximum(constant[key],0.) if key=='pred' else constant[key]
                np.testing.assert_allclose(v[mask],a[key][mask],rtol=2e-10,atol=2e-10)
            np.testing.assert_allclose(constant['candidate_scores'],a['candidate_scores'],rtol=2e-10,atol=2e-10,equal_nan=True)
            max_error=float(np.max(np.abs(np.maximum(constant['pred'],0.)[mask]-a['pred'][mask])))
        parity.append(dict(site=site,status='PASS',query_exact=True,maximum_prediction_error=max_error))
        with np.load(ROOT/'results/main'/f'{site}_B48_periodic.npz') as a:periodic=a['query'].copy()
        for seed in SEEDS:
            delays=np.random.default_rng(np.random.SeedSequence([seed,site_index])).choice(DELAYS,n,p=PROBABILITIES)
            for policy in POLICIES:
                path=OUT/f'{site}_B48_{policy}_random{seed}.npz'
                if not path.exists():
                    if policy=='periodic':
                        z=replay(f['models'],f['y'],f['ready'],f['initialization_mask'],periodic,delays)
                    else:
                        z=run_online(f['models'],f['y'],f['ready'],f['initialization_mask'],delays,
                                     policy=policy,horizon=240 if policy=='aware_online' else 168,availability=168)
                    pred=np.maximum(z['pred'],0.);assert np.isfinite(pred[mask]).all()
                    diagnostics=dict(runtime_seconds=z['runtime_seconds'],replay_steps=int(z['replay_steps']),
                                     additional_attempts=int(z['query'].sum()))
                    assert int(z['query'].sum())==int(periodic.sum())
                    metadata=dict(site=site,policy=policy,seed=seed,assumed_delay=72,
                        scientific_type='causal closed-loop with ready+168 commissioning availability under simulated random releases; approximate working planner',
                        commissioning_availability_gate_hours=168,
                        runner_sha256=sha(__file__))
                    pack=dict(pred=pred,var=z['var'],y=f['y'],times=np.asarray(f['times'].astype(str),dtype='U19'),
                        query=z['query'],initmask=f['initialization_mask'],score_masks=mask,full_score_masks=f['score_masks'],
                        ready_indices=f['ready'],metadata_json=json.dumps(f['metadata']),event_delay=delays,
                        release_time=np.where(z['query']|f['initialization_mask'],np.arange(n)+delays,-1),
                        revision_provenance_json=json.dumps(metadata),delivery_diagnostics_json=json.dumps(diagnostics))
                    for key in ('candidate_scores','start_mask','forced_query','chosen_lengths','block_spent',
                                'block_target','budget_shortfall','starts'):
                        if key in z:pack[key]=z[key]
                    np.savez_compressed(path,**pack)
                else:
                    with np.load(path) as saved:
                        assert json.loads(saved['revision_provenance_json'].item())['runner_sha256']==sha(__file__)
                        diagnostics=json.loads(saved['delivery_diagnostics_json'].item())
                rows.append(dict(site=site,policy=policy,seed=seed,**diagnostics))
                print('ONLINE',json.dumps(rows[-1]),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'runtime_diagnostics.csv',index=False)
    write_json(OUT/'constant72_real_parity.json',dict(status='PASS',checks=parity))


def summarize():
    aggregate=[];instruments=[]
    for seed in SEEDS:
        methods=tuple(f'{p}_random{seed}' for p in POLICIES);bundles=[]
        for site in SITES:
            runs={m:load_run(OUT/f'{site}_B48_{m}.npz') for m in methods}
            bundles.append(build_site_bundle(runs,methods=methods))
        assert sum(b['active'].sum() for b in bundles)==59674
        tables=metric_tables(bundles)
        for key,destination in (('aggregate_metrics',aggregate),('instrument_metrics',instruments)):
            frame=tables[key].copy();frame['seed']=seed
            frame['policy']=frame.method.str.replace(f'_random{seed}','',regex=False);destination.append(frame)
    a=pd.concat(aggregate,ignore_index=True);a.to_csv(OUT/'aggregate_metrics.csv',index=False)
    pd.concat(instruments,ignore_index=True).to_csv(OUT/'instrument_metrics.csv',index=False)
    a.groupby('policy').macro_instrument_rmse.agg(['mean','min','max']).reset_index().to_csv(OUT/'seed_ranges.csv',index=False)
    value=dict(status='PASS',site_streams=27,instruments=18,instrument_hours=59674,
        actual_causal_online_replanning=True,precise_distribution_aware_planner=False,
        delivery_observed_in_field=False,seeds=list(SEEDS),commissioning_availability_gate_hours=168,
        pre_fix_streams_are_not_final_evidence=True)
    write_json(OUT/'summary.json',value)
    text=("# Causal online planning under random reference releases\n\n"
        "This retrospective QUANT replay recomputes both request calendars each hour from the actual delivered "
        "history. No archived adaptive calendar is transferred. Releases are simulated per event with lags "
        "0/24/72/168 hours and probabilities .20/.20/.35/.25 (expected72 hours), three fixed paired seeds. "
        "The estimator handles out-of-sequence arrivals by event insertion and exact chronological refiltering "
        "of a bounded suffix. Predictions and request decisions precede current-hour releases. Fitted priors "
        "and every model contribution to the utility are withheld until ready+168, using the stated contract "
        "upper bound rather than future realized delays. Constant72 validation separately uses ready+72. "
        "Shared post-ready commissioning events follow the same simulated release law.\n\n"
        "The retained-aware planner uses H240, threshold1.25, and a working fixed72-hour contract. Its anchor "
        "is a conservative forward-filter prior at event time t−72, conditioned only on received earlier-event "
        "labels. Later-event information is omitted from that anchor and represented as common events by the "
        "working-contract rollout. Older overdue unreceived events before the anchor are omitted. This is "
        "an explicitly misspecified approximate planner, not a distribution-aware optimal controller or exact "
        "random-arrival utility. The original objective uses current legitimate covariance with H168. "
        "Both obey B48/672h, four starts and12-hour nominal windows.\n\n"
        "Constant72-hour aware prediction, query and score parity was checked against the frozen controller, "
        "including three real site streams. Synthetic checks cover hidden nonrequested labels, future labels "
        "and future delay draws, same-hour release causality, quotas and estimator equivalence to the separately "
        "validated out-of-order fixed-calendar replay. Synthetic tests are not empirical study evidence.\n\n"
        "All conditions score the same18 instruments and59,674 instrument-hours after ready+168. Three-seed "
        "ranges describe conditional simulation variability, not deployment confidence intervals or fresh-site "
        "confirmation. No field arrival logs or prospective service validation are claimed. The pre-fix "
        "ready+72 randomized-gate outputs remain in online_delivery as failed availability diagnostics, "
        "with their original producer snapshotted under evidence/revision6/precommissioning_fix.\n\n"
        "## Per-seed aggregate RMSE, ppb\n\n"+a[['seed','policy','macro_instrument_rmse']].to_csv(index=False))
    (EV/'online_delivery_corrected.md').write_text(text,encoding='utf-8');print('COMPLETE',json.dumps(value),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['test','run','summarize','all'],default='all')
    args=ap.parse_args();protocol()
    if args.stage in ('test','all'):self_test()
    if args.stage in ('run','all'):evaluate()
    if args.stage in ('summarize','all'):summarize()


if __name__=='__main__':main()
