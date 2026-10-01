"""Post-review diagnostics; never mutates frozen predictions or primary tables.

Definitions are written before execution. All new observations are retrospective
revision analyses, not an untouched confirmatory test or measured cost savings.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'cache/numba')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import itertools
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from statistical_analysis import load_run, build_site_bundle, paired_calendar_bootstrap

OUT = ROOT/'results/revision/diagnostics'
MAIN = ROOT/'results/main'
SITES = ['Manchester', 'London', 'York']
POLICIES = ['periodic','risk','risk_fixed','risk_independent','information_gain',
            'uncertainty','window_uncertainty','random1','random2','random3']
FOCUS = ['periodic','risk','risk_fixed','gp_stream','hgb_stream','noAR','reference_all','init_only']
DESIGN = dict(version=1, date='2026-10-01', status='post-review retrospective supplemental analysis',
    cost_lambda_hours=[0,6,24,72],
    episode_definition='Maximal continuous true runs in initmask OR additional query, including attempted but unsuccessful hours; not physical visits.',
    analytical_rmse_targets_ppb=[3,4,5,6,8],
    target_interpretation='Illustrative analytical targets fixed before this supplemental computation, informed by known original results; not regulatory limits or preregistration.',
    tested_budgets=[12,24,48], interpolation=False,
    age_bin_edges_hours=[0,24,72,168],
    age_reference='Strictly prior usable requested reference for this channel; valid sensor and finite archived reference, including own initialization.',
    age_episode='Hours after a completed merged requested episode; ongoing request hours reported separately. Ending at final record is right-censored.',
    age_interpretation='Conditional descriptive errors; schedule-specific bin support is not randomized or necessarily matched.',
    minimum_scored_hours=dict(day=18,week=126,month=168),
    temporal_definition='Calendar day, Monday-Sunday calendar week, calendar month; thresholds are display/summary eligibility, not altered primary scoring.',
    evidence_groups='Four physical Zep instruments are long-record evidence; 14 non-Zep Manchester instruments are short-record multi-product evidence.',
    random_seed_ids=list(range(1001,1031)),
    random_seed_rule='numpy SeedSequence([20261001, seed_id, site_index]).generate_state(1)[0]; identical site seed across budgets, all 30 seed IDs retained.',
    random_method='Existing fixed_site_mask randomized preferred episode positions and frozen common AR engine; not uniform over all feasible calendars.',
    bootstrap=dict(days=[7,14,28],replicates=[1000,300,300],base_seed=20261001,
                   scope='Paired circular calendar blocks within observed site quarters; realized errors, no refitting or geographic population inference.'),
    availability='Retained cohort, observed deployment spans; report initialization-inclusive and post-initialization denominators, preserve inactive replacement gaps.',
    flags='Existing score-only flag sensitivities are not replay filtering; this script does not represent them as refitting.')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def resolve_result_path(recorded):
    """Relocate a recorded result path into the current extracted project root."""
    raw=str(recorded).replace('\\','/')
    candidate=Path(raw)
    if not candidate.is_absolute() and not (len(raw)>1 and raw[1]==':'):
        candidate=ROOT/candidate
    if candidate.resolve().is_relative_to(ROOT.resolve()):
        return candidate
    normalized='/'+raw.lstrip('/')
    marker=normalized.lower().rfind('/results/')
    if marker<0:
        raise ValueError(f'Recorded result is outside this project and lacks a results/ suffix: {recorded}')
    suffix=Path(normalized[marker+1:])
    if '..' in suffix.parts:
        raise ValueError(f'Unsafe recorded result suffix: {recorded}')
    mapped=ROOT/suffix
    assert mapped.resolve().is_relative_to(ROOT.resolve())
    return mapped

def write_csv(rows, name):
    frame = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT/(name+'.csv'),index=False)
    return frame

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'analysis_design.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        assert old['design']==DESIGN, 'Do not silently change supplemental design'
    else:
        path.write_text(json.dumps(dict(design=DESIGN,original_freeze_sha256=sha(ROOT/'evidence/FREEZE.json'),
            original_statistics_sha256=sha(ROOT/'results/statistics_summary.json')),indent=2),encoding='utf-8')

def episodes(mask):
    """Inclusive starts/exclusive ends of contiguous requested clock hours."""
    z=np.r_[False,np.asarray(mask,bool),False].astype(np.int8)
    return np.flatnonzero(np.diff(z)==1),np.flatnonzero(np.diff(z)==-1)

def stats(a,prefix):
    a=np.asarray(a,dtype=float)
    return {prefix+'_n':int(len(a)),prefix+'_mean':float(np.mean(a)) if len(a) else np.nan,
        prefix+'_median':float(np.median(a)) if len(a) else np.nan,
        prefix+'_p90':float(np.quantile(a,.9)) if len(a) else np.nan,
        prefix+'_max':float(np.max(a)) if len(a) else np.nan}

def request_costs():
    rows=[]; events=[]; calendars={}
    for path in sorted(MAIN.glob('*.npz')):
        site,bstr,method=path.stem.split('_',2); b=int(bstr[1:])
        with np.load(path,allow_pickle=False) as z:
            q=z['query']; init=z['initmask']; allq=q|init; starts,ends=episodes(allq)
            paidstart,paidend=episodes(q)
            rawstarts=int(z['start_mask'].sum()) if 'start_mask' in z else np.nan
            forced=z['forced_query'] if 'forced_query' in z else np.zeros(len(q),bool)
            calendar_source=path.name
            inherited={'bias_ewma_risk':'risk','ewrls_risk':'risk','hgb_stream_risk':'risk',
                **{m:'periodic' for m in ['bias_ewma','ewrls','gp_stream','hgb_stream','ridge_stream']}}
            if method in inherited and ('forced_query' not in z or 'start_mask' not in z):
                source=MAIN/f'{site}_B{b}_{inherited[method]}.npz'
                with np.load(source,allow_pickle=False) as calendar:
                    assert np.array_equal(q,calendar['query']) and np.array_equal(init,calendar['initmask'])
                    forced=calendar['forced_query'] if 'forced_query' in calendar else np.zeros(len(q),bool)
                    rawstarts=int(calendar['start_mask'].sum()) if 'start_mask' in calendar else np.nan
                calendar_source=source.name
            t=pd.DatetimeIndex(pd.to_datetime(z['times'].astype(str)))
            base=dict(site=site,budget=b,method=method,setup_hours=int(init.sum()),paid_hours=int(q.sum()),
                total_attempted_hours=int(allq.sum()),merged_service_episodes=len(starts),
                paid_contiguous_runs=len(paidstart),accounting_starts=rawstarts,
                calendar_diagnostics_source=calendar_source,
                paid_finite_reference=int((q&np.isfinite(z['y'])).sum()),forced_paid_hours=int(forced.sum()),
                forced_paid_fraction=float(forced.sum()/q.sum()) if q.any() else np.nan)
            base.update(stats(ends-starts,'episode_length_hours'))
            base.update(stats(starts[1:]-ends[:-1],'unrequested_gap_hours'))
            for lam in DESIGN['cost_lambda_hours']:
                rows.append(base|dict(lambda_hours=lam,equivalent_hours=int(allq.sum())+lam*len(starts)))
            if method in POLICIES:
                calendars[(site,b,method)]=(q,init)
                si,se=episodes(init)
                for k,(left,right) in enumerate(zip(starts,ends)):
                    earlier=se[se<=left]; later=si[si>=right]
                    events.append(dict(site=site,budget=b,method=method,episode=k,start=str(t[left]),
                        end=str(t[right-1]),start_hour=int(t[left].hour),length_hours=int(right-left),
                        setup_hours=int(init[left:right].sum()),paid_hours=int(q[left:right].sum()),
                        forced_hours=int(forced[left:right].sum()),
                        hours_since_prior_setup_end=int(left-earlier[-1]+1) if len(earlier) else np.nan,
                        hours_to_next_setup_start=int(later[0]-right+1) if len(later) else np.nan,
                        overlaps_setup=bool(init[left:right].any())))
    cost=write_csv(rows,'reference_cost_by_site')
    agg=cost.groupby(['budget','method','lambda_hours'],as_index=False)[['setup_hours','paid_hours','total_attempted_hours','merged_service_episodes','equivalent_hours','forced_paid_hours']].sum()
    write_csv(agg,'reference_cost_aggregate'); write_csv(events,'service_episodes')
    joins=[]; distance=[]
    for site,b in itertools.product(SITES,DESIGN['tested_budgets']):
        for a,c in itertools.combinations(POLICIES,2):
            qa,ia=calendars[(site,b,a)]; qc,ic=calendars[(site,b,c)]
            for scope,x,y in [('paid',qa,qc),('all_requests',qa|ia,qc|ic)]:
                union=int((x|y).sum()); inter=int((x&y).sum())
                joins.append(dict(site=site,budget=b,method=a,comparator=c,scope=scope,
                    overlap_hours=inter,union_hours=union,jaccard=inter/union if union else 1.,
                    symmetric_difference_hours=int((x^y).sum())))
        for method in POLICIES:
            q,init=calendars[(site,b,method)]; iset=np.flatnonzero(init); starts,_=episodes(q)
            for start in starts:
                near=int(np.min(np.abs(iset-start))) if len(iset) else np.nan
                distance.append(dict(site=site,budget=b,method=method,paid_run_start_index=int(start),
                    distance_to_any_setup_hour=near,within_24h=near<=24,within_72h=near<=72,within_168h=near<=168))
    write_csv(joins,'calendar_overlap');write_csv(distance,'paid_start_setup_distance')
    ev=pd.DataFrame(events)
    write_csv(ev.groupby(['site','budget','method','start_hour'],as_index=False).size(),'episode_start_hour')
    A=pd.read_csv(ROOT/'results/tables/aggregate_metrics.csv'); A=A[A.reference.eq('primary')]
    curves=A.merge(agg,on=['budget','method'],how='inner')
    write_csv(curves,'error_total_access_curves')
    targets=[]
    for (method,lam),g in curves.groupby(['method','lambda_hours']):
        # Only methods with every tested budget enter minimum-tested-budget comparisons.
        if method in ('init_only','reference_all'):continue
        if set(g.budget)!=set(DESIGN['tested_budgets']):continue
        for endpoint in ['macro_instrument_rmse','site_balanced_rmse']:
            for goal in DESIGN['analytical_rmse_targets_ppb']:
                ok=g[g[endpoint]<=goal].sort_values(['equivalent_hours','budget'])
                row=dict(method=method,lambda_hours=lam,endpoint=endpoint,target_ppb=goal,
                    attained=bool(len(ok)),budget=np.nan,equivalent_hours=np.nan,total_attempted_hours=np.nan,rmse=np.nan)
                if len(ok):
                    z=ok.iloc[0];row.update(budget=int(z.budget),equivalent_hours=int(z.equivalent_hours),total_attempted_hours=int(z.total_attempted_hours),rmse=z[endpoint])
                targets.append(row)
    write_csv(targets,'minimum_tested_budget')
    return cost

def span_masks(run):
    ticks=np.arange(len(run.y)); full=[];post=[]
    for m in run.metadata:
        start=run.times.get_indexer([pd.Timestamp(m['start'])])[0]
        end=run.times.get_indexer([pd.Timestamp(m['end'])])[0]
        ready=run.times.get_indexer([pd.Timestamp(m['ready'])])[0]
        assert min(start,end,ready)>=0
        full.append((ticks>=start)&(ticks<=end));post.append((ticks>=ready)&(ticks<=end))
    return np.asarray(full),np.asarray(post)

def sensor_valid(run,data):
    rows=[]
    for m in run.metadata:
        g=data[data.deployment.eq(m['deployment'])].set_index('timestamp').reindex(run.times)
        rows.append(np.isfinite(g.x.to_numpy(float)))
    return np.asarray(rows)

def availability(data):
    rows=[];gaps=[];coverage=[]
    for site in SITES:
        run=load_run(MAIN/f'{site}_B48_periodic.npz'); valid=sensor_valid(run,data)
        full,post=span_masks(run)
        for ins in sorted({m['instrument'] for m in run.metadata}):
            ix=[i for i,m in enumerate(run.metadata) if m['instrument']==ins]
            for domain,spans in [('deployment_including_initialization',full),('post_initialization',post)]:
                active=spans[ix].any(axis=0); sv=(valid[ix]&spans[ix]).any(axis=0)
                n=int(active.sum()); ref=np.isfinite(run.y); paired=sv&ref
                row=dict(site=site,instrument=ins,family=run.metadata[ix[0]]['family'],domain=domain,
                    active_calendar_hours=n,sensor_available_hours=int((sv&active).sum()),
                    reference_available_hours=int((ref&active).sum()),paired_available_hours=int((paired&active).sum()),
                    scored_hours=int(run.score_masks[ix].any(axis=0).sum()))
                for label,mask in [('sensor',sv),('reference',ref),('paired',paired)]:
                    row[label+'_availability_fraction']=float((mask&active).sum()/n)
                    gs,ge=episodes(active&~mask)
                    row.update(stats(ge-gs,label+'_gap_hours'))
                    for left,right in zip(gs,ge):
                        gaps.append(dict(site=site,instrument=ins,domain=domain,signal=label,
                            start=str(run.times[left]),end=str(run.times[right-1]),gap_hours=int(right-left),
                            left_boundary=left==0 or not active[left-1],
                            right_boundary=right==len(active) or not active[min(right,len(active)-1)]))
                rows.append(row)
                if domain=='post_initialization':
                    scored=run.score_masks[ix].any(axis=0)
                    for freq,label,minimum in [('D','day',18),('W-SUN','week',126),('M','month',168)]:
                        pp=run.times.to_period(freq)
                        for period in pp[active].unique():
                            use=np.asarray(pp==period)&active
                            coverage.append(dict(site=site,instrument=ins,period_type=label,period=str(period),
                                active_calendar_hours=int(use.sum()),scored_hours=int((use&scored).sum()),
                                eligible_coverage=int((use&scored).sum())>=minimum,minimum_scored_hours=minimum))
    write_csv(rows,'availability');write_csv(gaps,'availability_gaps');write_csv(coverage,'period_calendar_coverage')

def prior_age(mask):
    ix=np.arange(len(mask));latest=np.maximum.accumulate(np.where(mask,ix,-10**8))
    before=np.r_[-10**8,latest[:-1]]
    return np.where(before>=0,ix-before,np.inf)

def losses_for_masks(run,masks):
    out=[]
    for ins in sorted({m['instrument'] for m in run.metadata}):
        ix=[i for i,m in enumerate(run.metadata) if m['instrument']==ins]
        mask=masks[ix];n=mask.sum(axis=0);active=n>0;hours=int(active.sum())
        if not hours: continue
        e=np.where(mask,run.pred[ix]-run.y[None,:],0.)
        se=(e*e).sum(axis=0)/np.maximum(n,1);ae=np.abs(e).sum(axis=0)/np.maximum(n,1)
        out.append(dict(site=run.site,budget=run.budget,method=run.method,instrument=ins,
            family=run.metadata[ix[0]]['family'],evidence_group='long_Zep' if run.metadata[ix[0]]['family']=='Zep' else 'short_multi_product',
            instrument_hours=hours,channel_observations=int(mask.sum()),rmse=float(np.sqrt(se.sum()/hours)),mae=float(ae.sum()/hours)))
    return out

def age_and_temporal(data):
    ages=[];periods=[];instrument=[]
    for site in SITES:
        template=load_run(MAIN/f'{site}_B48_periodic.npz');valid=sensor_valid(template,data)
        full,post=span_masks(template)
        for budget in DESIGN['tested_budgets']:
            for method in FOCUS:
                run=load_run(MAIN/f'{site}_B{budget}_{method}.npz');request=run.query|run.initmask
                usable=valid & np.isfinite(run.y)[None,:]&request[None,:]&full
                # Each channel's own initialization fit can use its initial pairs;
                # after readiness, a requested finite pair is a usable update.
                lag=np.asarray([prior_age(z) for z in usable])
                starts,ends=episodes(request); completed=np.zeros(len(request),bool)
                completed[ends[ends<len(request)]-1]=True
                episode_lag=np.broadcast_to(prior_age(completed),(len(valid),len(request)))
                edges=[0,24,72,168,np.inf];names=['0_24','24_72','72_168','168_plus']
                for axis,age in [('prior_usable_reference',lag),('completed_service_episode_end',episode_lag)]:
                    for scope in ['off_request','all_scored']:
                        for lo,hi,label in zip(edges[:-1],edges[1:],names):
                            selected=(age>=lo)&(age<hi)&np.isfinite(age)
                            if scope=='off_request':selected&=~request[None,:]
                            for row in losses_for_masks(run,run.score_masks&selected):
                                ages.append(row|dict(age_basis=axis,scope=scope,age_bin_hours=label))
                        missing=run.score_masks&~np.isfinite(age)
                        if scope=='off_request':missing&=~request[None,:]
                        for row in losses_for_masks(run,missing):ages.append(row|dict(age_basis=axis,scope=scope,age_bin_hours='no_prior_event'))
                if budget!=48:continue
                instrument.extend(losses_for_masks(run,run.score_masks))
                for freq,label,minhours in [('D','day',18),('W-SUN','week',126),('M','month',168)]:
                    ps=run.times.to_period(freq)
                    for period in ps.unique():
                        sel=np.asarray(ps==period)
                        for row in losses_for_masks(run,run.score_masks&sel[None,:]):
                            periods.append(row|dict(period_type=label,period=str(period),eligible_coverage=row['instrument_hours']>=minhours,minimum_scored_hours=minhours))
                print('diagnostic',site,budget,method,flush=True)
    ag=write_csv(ages,'reference_age_instrument_errors')
    keys=['budget','method','age_basis','scope','age_bin_hours','evidence_group']
    summary=ag.groupby(keys,as_index=False).agg(n_instruments=('instrument','size'),instrument_hours=('instrument_hours','sum'),macro_rmse=('rmse','mean'),macro_mae=('mae','mean'))
    write_csv(summary,'reference_age_group_errors')
    pe=write_csv(periods,'period_instrument_errors');ins=write_csv(instrument,'B48_instrument_errors')
    comparison=ins[ins.method.eq('risk_fixed')].merge(ins[ins.method.eq('periodic')],on=['site','instrument','family','evidence_group','budget'],suffixes=('_fixed','_periodic'))
    assert np.array_equal(comparison.instrument_hours_fixed,comparison.instrument_hours_periodic)
    comparison['rmse_difference']=comparison.rmse_fixed-comparison.rmse_periodic
    write_csv(comparison,'B48_fixed_vs_periodic_instruments')
    write_csv(comparison.groupby(['site','evidence_group'],as_index=False).agg(n_instruments=('instrument','size'),instrument_hours=('instrument_hours_fixed','sum'),fixed_rmse=('rmse_fixed','mean'),periodic_rmse=('rmse_periodic','mean'),difference=('rmse_difference','mean')),'B48_fixed_by_site_group')
    target=[]
    for keys,g in pe.groupby(['site','instrument','family','evidence_group','method','period_type']):
        eligible=g[g.eligible_coverage]
        for goal in DESIGN['analytical_rmse_targets_ppb']:
            target.append(dict(zip(['site','instrument','family','evidence_group','method','period_type'],keys))|
                dict(target_ppb=goal,observed_periods=len(g),eligible_periods=len(eligible),
                    target_met_periods=int((eligible.rmse<=goal).sum()),
                    proportion=float((eligible.rmse<=goal).mean()) if len(eligible) else np.nan,
                    scored_hours=int(eligible.instrument_hours.sum())))
    write_csv(target,'period_target_coverage')

def paired_intervals():
    methods=('periodic','risk_fixed','gp_stream','hgb_stream')
    bundles=[build_site_bundle({m:load_run(MAIN/f'{s}_B48_{m}.npz') for m in methods},methods=methods) for s in SITES]
    rows=[];diagnostics=[]
    for label,selected in [('all_sites',bundles)]+[(b['site'],[b]) for b in bundles]:
        for days,reps in zip([7,14,28],[1000,300,300]):
            result=paired_calendar_bootstrap(selected,n_replicates=reps,block_days=days,seed=20261001+days)
            diagnostics.append(dict(analysis_scope=label,**result['diagnostics']))
            for comp in ['periodic','gp_stream','hgb_stream']:
                ia=methods.index('risk_fixed'); ib=methods.index(comp)
                for metric,samples,observed in [('macro_instrument_rmse',result['primary'],result['observed_primary']),('site_balanced_rmse',result['secondary'],result['observed_secondary'])]:
                    lo,hi=np.quantile(samples[:,ia]-samples[:,ib],[.025,.975])
                    rows.append(dict(scope=label,budget=48,method='risk_fixed',comparator=comp,metric=metric,
                        difference=float(observed[ia]-observed[ib]),ci_low=float(lo),ci_high=float(hi),block_days=days,
                        accepted_replicates=result['diagnostics']['accepted_replicates'],seed=result['diagnostics']['seed']))
    write_csv(rows,'B48_fixed_paired_intervals')
    (OUT/'bootstrap_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')

def random_replays():
    from site_data import make_site
    from acquisition_controller import run_site_fixed
    from fixed_schedules import fixed_site_mask
    dest=OUT/'random_predictions';dest.mkdir(exist_ok=True)
    config=json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))
    rows=[]
    for si,site in enumerate(SITES):
        fleet=make_site(site,**config['calibrator']);origin=int(min(fleet['ready']))
        for budget in DESIGN['tested_budgets']:
            expected=load_run(MAIN/f'{site}_B{budget}_periodic.npz')
            for seed in DESIGN['random_seed_ids']:
                actual_seed=int(np.random.SeedSequence([20261001,seed,si]).generate_state(1)[0])
                path=dest/f'{site}_B{budget}_random{seed}.npz'
                if not path.exists():
                    q,a=fixed_site_mask(len(fleet['y']),origin,fleet['initialization_mask'],budget=budget,seed=actual_seed)
                    assert q.sum()==expected.query.sum()
                    out=run_site_fixed(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'],q)
                    pred=np.maximum(out['pred'],0.)
                    assert np.isfinite(pred[fleet['score_masks']]).all()
                    np.savez_compressed(path,pred=pred,y=fleet['y'],times=fleet['times'].to_numpy(dtype='datetime64[s]').astype('U19'),
                        score_masks=fleet['score_masks'],query=q,initmask=fleet['initialization_mask'],
                        metadata_json=json.dumps(fleet['metadata']),seed_id=seed,actual_seed=actual_seed,
                        original_freeze_sha256=sha(ROOT/'evidence/FREEZE.json'),runtime_seconds=out['runtime_seconds'])
                run=load_run(path)
                assert np.array_equal(run.score_masks,expected.score_masks)
                assert run.query.sum()==expected.query.sum()
                for row in losses_for_masks(run,run.score_masks):rows.append(row|dict(seed_id=seed,actual_seed=actual_seed))
            print('random_complete',site,budget,30,flush=True)
    r=write_csv(rows,'random_instrument_errors')
    agg=r.groupby(['budget','seed_id'],as_index=False).agg(rmse=('rmse','mean'),mae=('mae','mean'),n_instruments=('instrument','size'),instrument_hours=('instrument_hours','sum'))
    site=r.groupby(['budget','seed_id','site'],as_index=False).agg(rmse=('rmse','mean'),mae=('mae','mean'))
    sb=site.groupby(['budget','seed_id']).rmse.mean().rename('site_balanced_rmse').reset_index()
    agg=agg.merge(sb,on=['budget','seed_id']);write_csv(agg,'random_seed_aggregate');write_csv(site,'random_seed_site')
    summary=[]
    A=pd.read_csv(ROOT/'results/tables/aggregate_metrics.csv');A=A[A.reference.eq('primary')]
    for budget,g in agg.groupby('budget'):
        for endpoint in ['rmse','mae','site_balanced_rmse']:
            col={'rmse':'macro_instrument_rmse','mae':'macro_instrument_mae','site_balanced_rmse':'site_balanced_rmse'}[endpoint]
            for m in ['periodic','risk_fixed','risk','gp_stream','hgb_stream']:
                point=float(A[(A.budget==budget)&A.method.eq(m)].iloc[0][col])
                summary.append(dict(budget=int(budget),endpoint=endpoint,comparator=m,comparator_error=point,
                    n_seeds=len(g),random_mean=float(g[endpoint].mean()),random_sd=float(g[endpoint].std()),
                    random_min=float(g[endpoint].min()),random_p025=float(g[endpoint].quantile(.025)),
                    random_median=float(g[endpoint].median()),random_p975=float(g[endpoint].quantile(.975)),
                    random_max=float(g[endpoint].max()),fraction_random_better=float((g[endpoint]<point).mean())))
    write_csv(summary,'random_distribution_summary')

def verify_random_files():
    """Check every saved seed, rather than validating only distribution summaries."""
    folder=OUT/'random_predictions';records=[]
    for si,site in enumerate(SITES):
        for budget in DESIGN['tested_budgets']:
            expected=load_run(MAIN/f'{site}_B{budget}_periodic.npz')
            for seed in DESIGN['random_seed_ids']:
                path=folder/f'{site}_B{budget}_random{seed}.npz';run=load_run(path)
                assert run.times.equals(expected.times)
                assert np.array_equal(run.y,expected.y,equal_nan=True)
                assert np.array_equal(run.score_masks,expected.score_masks)
                assert np.array_equal(run.initmask,expected.initmask)
                assert run.query.sum()==expected.query.sum()
                assert [(m['instrument'],m['deployment'],m['start'],m['end']) for m in run.metadata]==[(m['instrument'],m['deployment'],m['start'],m['end']) for m in expected.metadata]
                with np.load(path,allow_pickle=False) as z:
                    assert int(z['seed_id'])==seed
                    assert int(z['actual_seed'])==int(np.random.SeedSequence([20261001,seed,si]).generate_state(1)[0])
                    assert str(z['original_freeze_sha256'])==sha(ROOT/'evidence/FREEZE.json')
                records.append(dict(site=site,budget=budget,seed_id=seed,path=str(path),sha256=sha(path)))
    assert len(records)==270
    (OUT/'random_verification.json').write_text(json.dumps(dict(status='PASS',files=records,
        checks=['all 30 fixed seed IDs, all 3 budgets, all 3 sites','common timestamps, reference, scoring support, initialization and row identities','equal attempted cost','finite nonnegative scored predictions','seed derivation and frozen manifest'],
        random_routine_sources={n:sha(ROOT/'scripts'/n) for n in ['fixed_schedules.py','calibration_engine.py','site_data.py','acquisition_controller.py']}),indent=2),encoding='utf-8')

def random_costs():
    """Report startup scenarios for every seed, without choosing a calendar."""
    verification=json.loads((OUT/'random_verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='PASS' and len(verification['files'])==270
    rows=[]
    for item in verification['files']:
        with np.load(item['path'],allow_pickle=False) as z:
            q=z['query'];init=z['initmask'];starts,ends=episodes(q|init)
            finite=np.isfinite(z['y'])
            for lam in DESIGN['cost_lambda_hours']:
                rows.append(dict(site=item['site'],budget=item['budget'],seed_id=item['seed_id'],lambda_hours=lam,
                    total_attempted_hours=int((q|init).sum()),merged_service_episodes=len(starts),
                    paid_finite_reference=int((q&finite).sum()),total_finite_reference=int(((q|init)&finite).sum()),
                    equivalent_hours=int((q|init).sum())+lam*len(starts)))
    site=write_csv(rows,'random_reference_cost_by_site')
    aggregate=site.groupby(['budget','seed_id','lambda_hours'],as_index=False)[['total_attempted_hours','merged_service_episodes','equivalent_hours','paid_finite_reference','total_finite_reference']].sum()
    assert len(aggregate)==360
    errors=pd.read_csv(OUT/'random_seed_aggregate.csv')
    aggregate=aggregate.merge(errors,on=['budget','seed_id'],validate='many_to_one')
    write_csv(aggregate,'random_reference_cost_by_seed')
    summary=aggregate.groupby(['budget','lambda_hours'],as_index=False).agg(
        seeds=('seed_id','size'),attempted_hours=('total_attempted_hours','first'),
        episodes_min=('merged_service_episodes','min'),episodes_mean=('merged_service_episodes','mean'),episodes_max=('merged_service_episodes','max'),
        paid_finite_reference_min=('paid_finite_reference','min'),paid_finite_reference_mean=('paid_finite_reference','mean'),paid_finite_reference_max=('paid_finite_reference','max'),
        equivalent_hours_min=('equivalent_hours','min'),equivalent_hours_mean=('equivalent_hours','mean'),equivalent_hours_max=('equivalent_hours','max'))
    write_csv(summary,'random_reference_cost_distribution')

def frontier():
    """Display extension: full threshold ladder, never interpolate budgets."""
    curves=pd.read_csv(OUT/'error_total_access_curves.csv'); rows=[]
    for (method,lam),g in curves.groupby(['method','lambda_hours']):
        if method in ('init_only','reference_all'):continue
        if set(g.budget)!=set(DESIGN['tested_budgets']):continue
        for endpoint in ['macro_instrument_rmse','site_balanced_rmse']:
            for goal in np.arange(3.,8.001,.25):
                ok=g[g[endpoint]<=goal].sort_values(['equivalent_hours','budget'])
                row=dict(method=method,lambda_hours=lam,endpoint=endpoint,target_ppb=goal,
                         attained=bool(len(ok)),budget=np.nan,equivalent_hours=np.nan,total_attempted_hours=np.nan)
                if len(ok):
                    z=ok.iloc[0];row.update(budget=int(z.budget),equivalent_hours=int(z.equivalent_hours),total_attempted_hours=int(z.total_attempted_hours))
                rows.append(row)
    write_csv(rows,'minimum_tested_budget_frontier')

def bundle_metrics(bundles,group):
    rows=[]
    for b in bundles:
        for i,ins in enumerate(b['instruments']):
            n=int(b['active'][i].sum())
            if not n:continue
            for j,m in enumerate(b['methods']):
                rows.append(dict(analysis_group=group,site=b['site'],instrument=ins,family=b['families'][i],method=m,
                    instrument_hours=n,channel_observations=int(b['channel_count'][i].sum()),
                    rmse=float(np.sqrt(b['se'][j,i].sum()/n)),mae=float(b['ae'][j,i].sum()/n),bias=float(b['signed'][j,i].sum()/n)))
    ins=pd.DataFrame(rows);agg=[]
    for method,g in ins.groupby('method'):
        assert not g.instrument.duplicated().any(), 'Reconcile cross-site physical identity before macro mean'
        agg.append(dict(analysis_group=group,method=method,n_instruments=len(g),n_sites=g.site.nunique(),
            instrument_hours=int(g.instrument_hours.sum()),channel_observations=int(g.channel_observations.sum()),
            macro_instrument_rmse=float(g.rmse.mean()),macro_instrument_mae=float(g.mae.mean()),macro_instrument_bias=float(g.bias.mean()),
            site_balanced_rmse=float(g.groupby('site').rmse.mean().mean())))
    return ins,pd.DataFrame(agg)

def restricted_bundle(b,rows):
    out=b.copy();rows=np.asarray(rows,dtype=bool)
    for key in ['active','channel_count']:out[key]=b[key][rows]
    for key in ['se','ae','signed']:out[key]=b[key][:,rows]
    for key in ['instruments','families']:out[key]=tuple(np.asarray(b[key])[rows])
    return out

def paired_rows(bundles,pairs,group,days=(7,),reps=(1000,)):
    rows=[];diag=[]
    if not bundles:return rows,diag
    methods=bundles[0]['methods']
    for day,nrep in zip(days,reps):
        r=paired_calendar_bootstrap(bundles,n_replicates=nrep,block_days=day,seed=20261001+day)
        diag.append(dict(analysis_group=group,**r['diagnostics']))
        for left,right in pairs:
            ia=methods.index(left);ib=methods.index(right)
            for metric,samples,obs in [('macro_instrument_rmse',r['primary'],r['observed_primary']),('site_balanced_rmse',r['secondary'],r['observed_secondary'])]:
                lo,hi=np.quantile(samples[:,ia]-samples[:,ib],[.025,.975])
                rows.append(dict(analysis_group=group,method=left,comparator=right,metric=metric,
                    difference=float(obs[ia]-obs[ib]),ci_low=float(lo),ci_high=float(hi),block_days=day,
                    n_instruments=len(r['instruments']),accepted_replicates=r['diagnostics']['accepted_replicates']))
    return rows,diag

def supplemental():
    """Integrate authoritative access_v2; original access directory is ignored."""
    from dataclasses import replace
    from study_data import read_data
    source=ROOT/'results/revision/access_v2'
    plan=dict(source=str(source),status='retrospective post-review integration',
        comparisons='Fixed-minus-periodic within each delay; own-setup matched contrasts; white-matched/retuned versus original periodic; memory versus same-calendar original; filtered replay versus scoring-only.',
        delay_support='Use supplied common ready+168 scored masks, never mix with original wider support.',
        persistence='168+ hours after completed request episode and separately after prior usable reference, off-request only; same reference calendar and exact common scoring mask across each memory comparison; long-Zep and short-multiproduct separately. Empty support is reported.',
        frontier='Additional requested descriptive threshold ladder 3..8 ppb by .25, no budget interpolation; original five targets unchanged.')
    (OUT/'integration_design.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    data=read_data();availability(data)
    for name in ['arrival','setup','components','memory','quality']:
        assert (source/f'{name}_summary.csv').exists(),name
    configurations=[]
    arrivals=[f'{p}_delay{d}' for d in [0,24,72,168] for p in ['periodic','risk_fixed','risk']]
    configurations.append(('arrival',{m:source for m in arrivals},[(f'risk_fixed_delay{d}',f'periodic_delay{d}') for d in [0,24,72,168]]+
        [(f'{p}_delay{d}',f'{p}_delay0') for p in ['periodic','risk_fixed'] for d in [24,72,168]],(7,14),(1000,300)))
    setup=['periodic','risk_fixed','risk']; own=[m+'_own_setup' for m in setup]
    configurations.append(('setup',{**{m:MAIN for m in setup},**{m:source for m in own}},
        [('risk_fixed_own_setup','periodic_own_setup')]+[(m+'_own_setup',m) for m in setup],(7,14),(1000,300)))
    configurations.append(('white_components',{'periodic':MAIN,'noAR':MAIN,'white_matched':source,'white_retuned':source},
        [('white_matched','periodic'),('white_retuned','periodic'),('white_matched','noAR')],(7,),(1000,)))
    memory=[p+'_'+m for p in ['periodic','risk_fixed'] for m in ['sensor_decay','reference_only']]
    configurations.append(('memory',{**{m:MAIN for m in ['periodic','risk_fixed']},**{m:source for m in memory}},
        [(m,m.removesuffix('_sensor_decay').removesuffix('_reference_only')) for m in memory],(7,),(1000,)))
    quality=[p+'_'+m for p in ['periodic','risk_fixed'] for m in ['filtered_replay','clean_scoring_only']]
    configurations.append(('quality',{m:source for m in quality},[(p+'_filtered_replay',p+'_clean_scoring_only') for p in ['periodic','risk_fixed']],(7,),(1000,)))
    allins=[];allagg=[];allci=[];alldiag=[];support=[]
    for group,mapping,pairs,days,reps in configurations:
        methods=tuple(mapping);bundles=[];runs_by_site={}
        for site in SITES:
            runs={m:load_run(directory/f'{site}_B48_{m}.npz') for m,directory in mapping.items()}
            runs_by_site[site]=runs
            bundles.append(build_site_bundle(runs,methods=methods))
        ins,agg=bundle_metrics(bundles,group);allins.append(ins);allagg.append(agg)
        ci,di=paired_rows(bundles,pairs,group,days,reps);allci.extend(ci);alldiag.extend(di)
        if group=='memory':
            for policy in ['periodic','risk_fixed']:
                names=(policy,policy+'_sensor_decay',policy+'_reference_only')
                paired=[(m,policy) for m in names[1:]]
                for age_basis in ['episode_end','prior_usable']:
                    eligible=[]
                    for site in SITES:
                        runs=runs_by_site[site];base=runs[policy];q=base.query|base.initmask
                        if age_basis=='episode_end':
                            starts,ends=episodes(q);last=np.zeros(len(q),bool);last[ends[ends<len(q)]-1]=True
                            age=np.broadcast_to(prior_age(last),base.pred.shape)
                        else:
                            valid=sensor_valid(base,data);full,_=span_masks(base)
                            usable=valid&np.isfinite(base.y)[None,:]&q[None,:]&full
                            age=np.asarray([prior_age(z) for z in usable])
                        gap=(age>=168)&np.isfinite(age)&~q[None,:]
                        restricted={}
                        for m in names:
                            assert np.array_equal(runs[m].query,base.query) and np.array_equal(runs[m].initmask,base.initmask)
                            restricted[m]=replace(runs[m],score_masks=runs[m].score_masks&gap)
                        b=build_site_bundle(restricted,methods=names)
                        eligible.append(restricted_bundle(b,b['active'].any(axis=1)))
                    for label in ['long_Zep','short_multi_product']:
                        ss=[]
                        for b in eligible:
                            keep=np.array([(f=='Zep')==(label=='long_Zep') for f in b['families']])
                            if keep.any():ss.append(restricted_bundle(b,keep))
                        name=f'memory_168h_{age_basis}_{policy}_{label}'
                        support.append(dict(analysis_group=name,available=bool(ss),n_sites=len(ss),
                            n_instruments=sum(len(b['instruments']) for b in ss),
                            instrument_hours=sum(int(b['active'].sum()) for b in ss)))
                        if not ss:continue
                        ii,aa=bundle_metrics(ss,name);allins.append(ii);allagg.append(aa)
                        ci,di=paired_rows(ss,paired,name);allci.extend(ci);alldiag.extend(di)
        print('supplement_group_complete',group,flush=True)
        write_csv(pd.concat(allins,ignore_index=True),'supplement_instrument_metrics')
        write_csv(pd.concat(allagg,ignore_index=True),'supplement_aggregate_metrics')
        write_csv(allci,'supplement_paired_intervals')
        write_csv(support,'memory_168h_support')
        (OUT/'supplement_bootstrap_diagnostics.json').write_text(json.dumps(alldiag,indent=2),encoding='utf-8')
    frontier()

def quality_pair():
    """Append the requested fully filtered fixed-versus-periodic contrast."""
    source=ROOT/'results/revision/access_v2'
    methods=('periodic_filtered_replay','risk_fixed_filtered_replay')
    bundles=[build_site_bundle({m:load_run(source/f'{site}_B48_{m}.npz') for m in methods},methods=methods) for site in SITES]
    assert sum(int(b['active'].sum()) for b in bundles)==61210
    rows,diagnostics=paired_rows(bundles,[(methods[1],methods[0])],'quality',(7,14),(1000,300))
    path=OUT/'supplement_paired_intervals.csv'
    old=pd.read_csv(path);new=pd.DataFrame(rows)
    keys=['analysis_group','method','comparator','metric','block_days']
    oldkeys=set(map(tuple,old[keys].to_numpy()))
    new=new[[tuple(row) not in oldkeys for row in new[keys].to_numpy()]]
    if len(new):
        # Append rather than rewriting the previously published rows.
        new[old.columns].to_csv(path,mode='a',header=False,index=False)
        diagpath=OUT/'supplement_bootstrap_diagnostics.json'
        previous=json.loads(diagpath.read_text(encoding='utf-8'))
        previous.extend([dict(**d,contrast='risk_fixed_filtered_replay-minus-periodic_filtered_replay') for d in diagnostics])
        diagpath.write_text(json.dumps(previous,indent=2),encoding='utf-8')
    print(new.to_string(index=False),flush=True)

def operational_contrasts():
    """Paired differences of model-policy advantages, not CI overlap tests."""
    source=ROOT/'results/revision/access_v2'
    arrivals={f'{p}_delay{d}':source for d in [0,24,72,168] for p in ['periodic','risk_fixed']}
    setup={m:MAIN for m in ['periodic','risk_fixed']}
    setup.update({m+'_own_setup':source for m in ['periodic','risk_fixed']})
    contrasts={'arrival':[(f'delay{d}_advantage_change',{
        f'risk_fixed_delay{d}':1,f'periodic_delay{d}':-1,'risk_fixed_delay0':-1,'periodic_delay0':1}) for d in [24,72,168]],
        'setup':[('own_setup_advantage_change',{
            'risk_fixed_own_setup':1,'periodic_own_setup':-1,'risk_fixed':-1,'periodic':1})]}
    design=dict(status='Additional post-review contrasts requested after initial supplemental estimates',
        definition='[fixed_changed - periodic_changed] - [fixed_original - periodic_original] in macro RMSE; positive means a numerically reduced fixed-policy advantage',
        contrasts=contrasts,bootstrap_days=[7,14],replicates=[1000,300],
        support='Exact common original supplied masks: arrival59674, setup62652 instrument-hours; all four methods per contrast, no source alteration',
        interpretation='Descriptive paired intervals conditional on observed sites, without claiming a difference solely because separate intervals cross zero differently.')
    (OUT/'operational_advantage_change_design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    rows=[];support=[];diagnostics=[]
    for group,mapping,nexpected in [('arrival',arrivals,59674),('setup',setup,62652)]:
        methods=tuple(mapping)
        bundles=[build_site_bundle({m:load_run(path/f'{s}_B48_{m}.npz') for m,path in mapping.items()},methods=methods) for s in SITES]
        ins,agg=bundle_metrics(bundles,group)
        assert agg.instrument_hours.eq(nexpected).all() and agg.n_instruments.eq(18).all()
        for name,coefficients in contrasts[group]:
            ss=ins[ins.method.isin(coefficients)].copy();ss['contrast']=name
            ss['coefficient']=ss.method.map(coefficients);support.append(ss)
        for days,nreps in [(7,1000),(14,300)]:
            result=paired_calendar_bootstrap(bundles,n_replicates=nreps,block_days=days,seed=20261001+days)
            diagnostics.append(dict(analysis_group=group,**result['diagnostics']))
            for name,coefficients in contrasts[group]:
                vector=np.array([coefficients.get(m,0.) for m in methods])
                assert (vector!=0).sum()==4 and vector.sum()==0
                for metric,samples,observed in [('macro_instrument_rmse',result['primary'],result['observed_primary']),
                    ('site_balanced_rmse',result['secondary'],result['observed_secondary'])]:
                    point=float(observed@vector);ci=np.quantile(samples@vector,[.025,.975])
                    rows.append(dict(analysis_group=group,contrast=name,coefficients_json=json.dumps(coefficients,sort_keys=True),
                        metric=metric,difference_of_differences=point,ci_low=float(ci[0]),ci_high=float(ci[1]),
                        block_days=days,n_instruments=18,instrument_hours=nexpected,
                        accepted_replicates=result['diagnostics']['accepted_replicates']))
    frame=write_csv(rows,'operational_advantage_change_intervals')
    write_csv(pd.concat(support,ignore_index=True),'operational_advantage_change_support')
    (OUT/'operational_advantage_change_bootstrap_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    print(frame[frame.metric.eq('macro_instrument_rmse')].to_string(index=False),flush=True)

def matrix_phase():
    """Read completed producer matrices; never fit or choose any model here."""
    from dataclasses import replace
    source=ROOT/'results/revision/competitive'
    names=['frozen','shared_selected'];models=['ar','gp','hgb'];policies=['periodic','risk_fixed','risk']
    expected={(s,m,p) for s in SITES for m in models for p in policies}
    methods=tuple(m+'__'+p for m in models for p in policies)
    pairs=[(m+'__'+p,m+'__periodic') for m in models for p in ['risk_fixed','risk']]
    pairs += [('ar__risk_fixed',m+'__'+p) for m in ['gp','hgb'] for p in ['periodic','risk_fixed']]
    design=dict(status='retrospective revision analysis, no model or calendar selection from test errors',
        matrices=names,methods=methods,comparisons=pairs,bootstrap_days=[7,14],replicates=[1000,300],
        interpretation='Within-model calendar contrasts and fixed AR versus periodic and fixed GP/HGB; conditional on original shared scoring support. No formal factorial interaction test.',
        source='Producer matrix paths and hashes are authoritative; original and shared-development-selected matrices are separate comparisons.')
    (OUT/'matrix_analysis_design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    available=[]
    for name in names:
        path=source/f'matrix_{name}.json'
        if not path.exists():
            print('matrix_not_ready',name,flush=True);continue
        payload=json.loads(path.read_text(encoding='utf-8'));cells=payload['cells']
        assert len(cells)==27 and {(c['site'],c['model'],c['policy']) for c in cells}==expected
        # Verify the actual files in this checkout even when analysis tables are cached.
        resolved={(c['site'],c['model'],c['policy']):resolve_result_path(c['path']) for c in cells}
        for c in cells:
            p=resolved[(c['site'],c['model'],c['policy'])]
            assert sha(p)==c['sha256'],p
        manifest=OUT/f'matrix_{name}_inputs.json'
        products=[OUT/f'matrix_{name}_{suffix}.csv' for suffix in ['instrument_metrics','aggregate_metrics','site_metrics','paired_intervals']]
        if manifest.exists() and all(p.exists() for p in products) and json.loads(manifest.read_text(encoding='utf-8')).get('matrix_sha256')==sha(path):
            print('matrix_already_complete',name,flush=True);available.append(name);continue
        indexed={(c['site'],c['model'],c['policy']):c for c in cells}
        bundles=[]
        for site in SITES:
            runs={}
            for model in models:
                for policy in policies:
                    c=indexed[(site,model,policy)];p=resolved[(site,model,policy)]
                    key=model+'__'+policy;runs[key]=replace(load_run(p),method=key)
            for policy in policies:
                base=runs['ar__'+policy]
                for model in ['gp','hgb']:
                    run=runs[model+'__'+policy]
                    assert np.array_equal(run.query,base.query), (site,model,policy,'schedule mismatch')
                    assert np.array_equal(run.initmask,base.initmask)
            bundles.append(build_site_bundle(runs,methods=methods))
        ins,agg=bundle_metrics(bundles,'matrix_'+name)
        assert agg.n_instruments.eq(18).all() and agg.instrument_hours.eq(62652).all()
        ci,diag=paired_rows(bundles,pairs,'matrix_'+name,(7,14),(1000,300))
        siteagg=ins.groupby(['analysis_group','method','site'],as_index=False).agg(
            rmse=('rmse','mean'),mae=('mae','mean'),bias=('bias','mean'),n_instruments=('instrument','size'),instrument_hours=('instrument_hours','sum'))
        write_csv(ins,'matrix_'+name+'_instrument_metrics');write_csv(agg,'matrix_'+name+'_aggregate_metrics')
        write_csv(siteagg,'matrix_'+name+'_site_metrics');write_csv(ci,'matrix_'+name+'_paired_intervals')
        (OUT/f'matrix_{name}_bootstrap_diagnostics.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
        manifest.write_text(json.dumps(dict(matrix_path=str(path),matrix_sha256=sha(path),cells=cells,
            status='PASS',checks=['27 complete cells; source hashes','same-policy exact query and initialization across models',
                'common reference, timestamps, metadata and 62652 instrument-hour scoring support','all 18 physical instruments; finite scored predictions']),indent=2),encoding='utf-8')
        available.append(name);print('matrix_complete',name,flush=True)
        print(agg[['method','macro_instrument_rmse','macro_instrument_mae','site_balanced_rmse']].to_string(index=False),flush=True)
    return available

def coverage_summary():
    """Expose calendar periods with zero scores alongside conditional accuracy."""
    calendar=pd.read_csv(OUT/'period_calendar_coverage.csv')
    assert not calendar.duplicated(['site','instrument','period_type','period']).any()
    summary=calendar.groupby(['site','instrument','period_type'],as_index=False).agg(
        active_calendar_periods=('period','size'),calendar_eligible_periods=('eligible_coverage','sum'),
        zero_score_periods=('scored_hours',lambda z:int((z==0).sum())),
        active_calendar_hours=('active_calendar_hours','sum'),calendar_scored_hours=('scored_hours','sum'))
    targets=pd.read_csv(OUT/'period_target_coverage.csv')
    joined=targets.merge(summary,on=['site','instrument','period_type'],how='left',validate='many_to_one')
    assert joined.active_calendar_periods.notna().all()
    assert (joined.eligible_periods==joined.calendar_eligible_periods).all()
    joined['eligible_fraction_of_calendar_periods']=joined.eligible_periods/joined.active_calendar_periods
    joined['target_met_fraction_of_calendar_periods']=joined.target_met_periods/joined.active_calendar_periods
    # This combined availability/accuracy fraction is not an assigned error on missing hours.
    write_csv(joined,'period_target_coverage_complete')
    write_csv(summary,'period_calendar_coverage_summary')
    source=ROOT/'results/revision/access_v2'
    files=sorted(source.glob('*.npz'));assert len(files)==75
    record=dict(status='PASS',source='authoritative access_v2 only; superseded access ignored',
        npz_files=[dict(path=str(p),sha256=sha(p)) for p in files],
        summary_files={p.name:sha(p) for p in source.glob('*_summary.csv')},
        original_freeze_sha256=sha(ROOT/'evidence/FREEZE.json'),
        notes='This verification identifies analyzed inputs; it does not independently replay the producer experiments.')
    (OUT/'supplement_inputs_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

def self_test():
    s,e=episodes([1,1,0,1,0,1,1]); assert list(s)==[0,3,5] and list(e)==[2,4,7]
    age=prior_age([1,0,0,1,0]);assert np.isinf(age[0]) and np.array_equal(age[1:],[1,2,3,1])
    assert resolve_result_path('D:/former_checkout/results/main/test.npz')==ROOT/'results/main/test.npz'
    assert resolve_result_path('D:/old/results/archive/results/main/test.npz')==ROOT/'results/main/test.npz'
    assert resolve_result_path('results/main/test.npz')==ROOT/'results/main/test.npz'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['freeze','existing','costs','random','random_costs','supplemental','quality_pair','operational_contrasts','matrix','coverage','verify_random','all'],default='existing')
    args=parser.parse_args();freeze();self_test();start=time.perf_counter()
    with threadpool_limits(limits=2):
        if args.phase=='costs':request_costs();frontier()
        if args.phase in ['existing','all']:
            request_costs(); paired_intervals()
            print('cost_calendar_bootstrap_complete',flush=True)
            from study_data import read_data
            data=read_data();availability(data);age_and_temporal(data)
        if args.phase in ['random','all']:random_replays()
        if args.phase in ['verify_random','all']:verify_random_files()
        if args.phase in ['random_costs','all']:random_costs()
        if args.phase in ['supplemental','all']:supplemental()
        if args.phase in ['quality_pair','all']:quality_pair()
        if args.phase in ['operational_contrasts','all']:operational_contrasts()
        if args.phase in ['matrix','all']:matrix_phase()
        if args.phase in ['coverage','all']:coverage_summary()
    print('COMPLETE',args.phase,'seconds',round(time.perf_counter()-start,2),flush=True)

if __name__=='__main__':main()
