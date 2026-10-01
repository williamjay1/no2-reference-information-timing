"""Enhanced-budget GP acquisition, independent sixth-review sensitivity.

The correction, data support and exact access quotas remain frozen. This
retrospective, non-selected sensitivity increases only the planning budget:
six-hour candidates, 28 quadrature offsets, and three strictly past scenarios.
Existing fourth-review sources and results are never modified.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import revision4_gp_active as base
from statistical_analysis import load_run, build_site_bundle, metric_tables

ROOT = base.ROOT
OUT = ROOT / 'results/revision6/gp_active'
SITES = base.SITES
PARAMS = base.PARAMS.copy()
SETTINGS = dict(base.SETTINGS, candidate_stride=6)
OFFSETS = np.arange(6, 169, 6, dtype=int)
LAGS = (192, 216, 288)
PLAN = dict(stage='Sixth-review retrospective sensitivity; evaluation results were already known',
    settings=SETTINGS, correction_parameters=PARAMS, evaluation_offsets=OFFSETS.tolist(),
    quadrature='Equal right-endpoint six-hour weights across the same 168-hour horizon',
    historical_lags=list(LAGS), historical_scenarios=3,
    scenario_rule='Use each complete lagged 168-hour path ending strictly before the current time. Missing historical features use the current context. If no complete path exists, use one frozen current-context path. Average available scenario utilities; no future feature or target is used.',
    scope='Immediate shared commissioning; same three evaluation sites, eighteen physical instruments, score masks, attempted-hour quotas, maximum starts and correction model as revision4.',
    fixed='No new kernel, threshold, horizon, memory, feature, training or target selection. No test-driven selection between original and enhanced planning settings.',
    approximation='Fixed current GP posterior and kernel; covariance updates integrate hypothetical target values out. Future pair aging and cap removal are not simulated. Historical scenarios do not establish a calibrated distribution of future contexts.',
    comparison=['original 12-hour/9-offset/one-scenario GP planner', 'GP on AR fixed-risk calendar', 'GP on AR variable-risk calendar', 'GP periodic'],
    overlap='Paid-hour Jaccard and asymmetric start-nearness fractions within twelve hours; known commissioning excluded from overlap. Start extraction from stored start_mask when present, otherwise starts of contiguous paid runs.',
    inference='Descriptive computational-budget sensitivity. Existing evaluation has already been observed; this is not an independent confirmatory optimization experiment.')

def write_json(path, value):
    base.write_json(path, value)

def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path=ROOT/'results/revision/competitive/shared_development_selection.json'
    selected=json.loads(selection_path.read_text(encoding='utf-8'))
    assert selected['selection_data_end'] < '2021-01-01'
    assert selected['selected']['gp']['parameters'] == PARAMS
    identity=dict(plan=PLAN, selection_sha256=base.sha(selection_path),
        base_runner_sha256=base.sha(ROOT/'scripts/revision4_gp_active.py'),
        datasets_sha256={name:base.sha(ROOT/'datasets'/name) for name in ('no2_hourly_oob.parquet','reference_no2_ratified.parquet')})
    target=OUT/'plan.json'
    if target.exists():
        assert json.loads(target.read_text(encoding='utf-8'))['identity'] == identity
    else:
        write_json(target,dict(recorded_utc=datetime.now(timezone.utc).isoformat(), identity=identity))
    return base.sha(target)

class GPStream(base.GPStream):
    """Identical correction implementation; richer covariance planning only."""
    def utility(self, t, length, initmask, settings):
        begin=perf_counter(); horizon=settings['horizon']; local_t=t-self.start
        free=np.zeros(horizon+1,bool)
        last=min(len(initmask),t+horizon+1); free[:last-t]=initmask[t:last]
        paid=[]
        for hour in range(horizon+1):
            if not free[hour] and len(paid)<length: paid.append(hour)
        offsets=OFFSETS[OFFSETS<=horizon]
        weights=np.diff(np.r_[0,offsets]).astype(float); weights/=weights.sum()
        hours=np.asarray(sorted(set(paid+np.flatnonzero(free).tolist()+offsets.tolist())),int)
        origins=[local_t-lag for lag in LAGS if local_t-lag>=0 and local_t-lag+horizon<local_t]
        if not origins: origins=[None]
        values=[]
        for origin in origins:
            U=np.repeat(self.Z[local_t][None,:],len(hours),axis=0)
            if origin is not None:
                assert origin+horizon<local_t
                for row,hour in enumerate(hours):
                    index=origin+hour
                    if hour>0 and self.valid[index]: U[row]=self.Z[index]
            covariance=self.covariance(U)
            values.append(base.sequential_variance_reduction(covariance,hours,offsets,weights,free,set(paid),PARAMS['gp_noise']+1e-6))
        self.diag['score_seconds']+=perf_counter()-begin; self.diag['scores']+=1
        self.diag['historical_scenario_evaluations']=self.diag.get('historical_scenario_evaluations',0)+len(origins)
        self.diag['max_score_contexts']=max(self.diag['max_score_contexts'],len(hours))
        return float(np.mean(values))*self.scale**2

def models_from_local(local, source, fleet):
    weights={row['deployment']:model['weight'] for row,model in zip(fleet['metadata'],fleet['models'])}
    return [GPStream(item['X'],source['y'][item['start']:item['start']+336],item['valid'],
        start=item['start'],weight=weights[item['row']['deployment']],name=item['row']['deployment']) for item in local]

def run_site(site, stage='test', smoke_hours=None):
    plan_hash=plan(); begin=perf_counter()
    fleet=base.make_site(site,stage=stage,q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
    source_path=ROOT/'results/main'/f'{site}_B48_periodic.npz' if stage=='test' else ROOT/'results/revision/competitive/development_calendar.npz'
    if stage!='test': assert fleet['times'].max()<pd.Timestamp('2021-01-01')
    source,local=base.frozen._load_source(source_path,fleet)
    output=OUT/('streams' if stage=='test' else 'development')/f'{site}_B48_gp_enhanced.npz'
    provenance=dict(plan_sha256=plan_hash,runner_sha256=base.sha(__file__),source_sha256=base.sha(source_path),stage=stage,parameters=PARAMS)
    if output.exists() and smoke_hours is None:
        with np.load(output,allow_pickle=False) as saved:
            assert json.loads(saved['revision6_provenance_json'].item())==provenance
        load_run(output); print('RESUME',output,flush=True);return
    with threadpool_limits(limits=2):
        models=models_from_local(local,source,fleet)
        n=len(source['y']) if smoke_hours is None else min(len(source['y']),smoke_hours)
        result=base.replay(models,source['y'][:n],source['initmask'][:n].astype(bool),settings=SETTINGS)
    elapsed=perf_counter()-begin; records=[dict(deployment=model.name,**model.diag) for model in models]
    if smoke_hours is not None:
        payload=dict(site=site,stage=stage,hours=n,total_seconds=elapsed,model_count=len(models),
            fit_seconds=sum(row['fit_seconds'] for row in records),score_seconds=sum(row['score_seconds'] for row in records),
            scores=sum(row['scores'] for row in records),scenario_evaluations=sum(row.get('historical_scenario_evaluations',0) for row in records),
            attempts=int(result['query'].sum()),starts=result['starts'].tolist(),models=records)
        write_json(OUT/f'real_smoke_{site}.json',payload);print(json.dumps({key:value for key,value in payload.items() if key!='models'},indent=2),flush=True);return
    saved=base.frozen._shared_arrays(source); saved.update(result)
    saved['pred']=np.maximum(saved['pred'],0.)
    saved['runtime_seconds']=np.asarray(elapsed); saved['runtime_json']=np.asarray(json.dumps(records))
    saved['revision6_provenance_json']=np.asarray(json.dumps(provenance,sort_keys=True))
    finite=np.isfinite(saved['y']); all_access=saved['initmask']|saved['query']
    saved['access_json']=np.asarray(json.dumps(dict(initialization_attempts=int(saved['initmask'].sum()),
        additional_attempts=int(saved['query'].sum()),additional_successes=int((saved['query']&finite).sum()),
        total_attempts=int(all_access.sum()),merged_episodes=int((all_access&~np.r_[False,all_access[:-1]]).sum()),
        forced_paid_hours=int(saved['forced_query'].sum()),runtime_seconds=elapsed)))
    for key in ('y','times','score_masks','initmask','metadata_json'):
        assert np.array_equal(saved[key],source[key],equal_nan=True) if saved[key].dtype.kind in 'fc' else np.array_equal(saved[key],source[key]),key
    assert np.isfinite(saved['pred'][saved['score_masks']]).all()
    assert saved['query'].sum()==source['query'].sum()
    assert np.array_equal(saved['block_spent'],saved['block_feasible']) and np.all(saved['starts']<=4)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle: np.savez_compressed(handle,**saved)
    temporary.replace(output);load_run(output)
    write_json(output.with_suffix('.json'),dict(provenance=provenance,records=records,elapsed_seconds=elapsed))
    print('SAVED',output.name,'seconds',round(elapsed,2),'attempts',int(saved['query'].sum()),flush=True)

def smoke():
    plan(); rng=np.random.default_rng(20261001);n=900;initial=96
    X=rng.normal(size=(n,3));X[:,0]+=10;X[12::49,1]=np.nan;X[29::61,0]=np.nan
    y=1.3*np.nan_to_num(X[:,0])+rng.normal(size=n)
    initmask=np.arange(n)<initial;initmask[500:520]=True
    def execute(features=X, targets=y, external=None):
        model=GPStream(features,targets[:initial],np.isfinite(features[:,0]),initial=initial)
        with threadpool_limits(limits=2): result=base.replay([model],targets,initmask,settings=SETTINGS,external_query=external)
        return result,model
    begin=perf_counter();original,model=execute()
    hidden=~(original['query']|initmask);altered=y.copy();altered[hidden]=1e8
    altered[np.flatnonzero(hidden)[::2]]=np.nan
    changed,_=execute(targets=altered)
    np.testing.assert_array_equal(original['query'],changed['query'])
    np.testing.assert_allclose(original['pred'],changed['pred'],equal_nan=True,rtol=0,atol=0)
    event=int(np.flatnonzero(original['query'])[0]);altered=y.copy();altered[event]+=100
    changed,_=execute(targets=altered)
    np.testing.assert_allclose(original['pred'][:,:event+1],changed['pred'][:,:event+1],equal_nan=True,rtol=0,atol=0)
    np.testing.assert_array_equal(original['query'][:event+1],changed['query'][:event+1])
    cut=396;features=X.copy();features[cut:]+=100
    changed,_=execute(features=features)
    np.testing.assert_array_equal(original['query'][:cut],changed['query'][:cut])
    np.testing.assert_allclose(original['pred'][:,:cut],changed['pred'][:,:cut],equal_nan=True,rtol=0,atol=0)
    expected=base.original.fit_predict_stream_baselines(X,y,initial,original['query']|initmask,
        valid_x=np.isfinite(X[:,0]),model_names=('gp_stream',),**PARAMS)['gp_stream']
    np.testing.assert_allclose(original['pred'][0],np.maximum(expected,0),equal_nan=True,rtol=1e-12,atol=1e-12)
    assert np.array_equal(original['block_spent'],original['block_feasible'])
    assert np.all(original['starts']<=4) and not np.any(original['query']&initmask)
    checks=['Hidden unacquired label values and missingness do not change decisions/predictions',
        'Acquired current label cannot change current prediction or decision',
        'Future feature perturbation cannot change past decisions or predictions',
        'Enhanced-calendar correction matches independent frozen GP replay at 1e-12',
        'Exact attempted-hour budgets, maximum four starts and commissioning exclusion']
    payload=dict(status='PASS',kind='synthetic implementation verification, not empirical sensor evidence',checks=checks,
        elapsed_seconds=perf_counter()-begin,runner_sha256=base.sha(__file__),settings=SETTINGS)
    write_json(OUT/'smoke_checks.json',payload);print(json.dumps(payload,indent=2),flush=True)

def path_for(site, method):
    if method=='gp_enhanced': return OUT/'streams'/f'{site}_B48_gp_enhanced.npz'
    if method=='gp_original': return ROOT/'results/revision4/gp_active/streams'/f'{site}_B48_gp_active.npz'
    policy={'gp_periodic':'periodic','gp_ar_fixed':'risk_fixed','gp_ar_variable':'risk'}[method]
    return ROOT/'results/revision/competitive/shared_selected'/f'{site}_B48_gp_shareddev_{policy}.npz'

def starts_of(path, query):
    with np.load(path,allow_pickle=False) as source:
        if 'start_mask' in source.files: return np.flatnonzero(source['start_mask']), 'stored planned starts'
    return np.flatnonzero(query&~np.r_[False,query[:-1]]), 'starts of contiguous paid runs'

def start_calendar_path(site, method):
    if method in ('gp_ar_fixed','gp_ar_variable'):
        policy={'gp_ar_fixed':'risk_fixed','gp_ar_variable':'risk'}[method]
        return ROOT/'results/main'/f'{site}_B48_{policy}.npz'
    return path_for(site,method)

def summarize():
    plan();methods=('gp_enhanced','gp_original','gp_ar_fixed','gp_ar_variable','gp_periodic')
    bundles=[];overlap=[];runtimes=[];index=[]
    for site in SITES:
        runs={}
        for method in methods:
            path=path_for(site,method);run=load_run(path);run.method=method;runs[method]=run
            index.append(dict(site=site,method=method,path=str(path),sha256=base.sha(path)))
            assert run.query.sum()==runs['gp_enhanced'].query.sum()
            assert np.array_equal(run.score_masks,runs['gp_enhanced'].score_masks)
            with np.load(path,allow_pickle=False) as source:
                if 'runtime_seconds' in source.files:runtime=float(source['runtime_seconds'])
                else:runtime=None
                records=json.loads(source['runtime_json'].item()) if 'runtime_json' in source.files else []
            runtimes.append(dict(site=site,method=method,total_seconds=runtime,
                fit_seconds=sum(row.get('fit_seconds',0.) for row in records),
                score_seconds=sum(row.get('score_seconds',0.) for row in records),
                scores=sum(row.get('scores',0) for row in records),
                scenario_evaluations=sum(row.get('historical_scenario_evaluations',row.get('scores',0)) for row in records)))
        bundles.append(build_site_bundle(runs,methods=methods))
        left=runs['gp_enhanced'];left_starts,left_rule=starts_of(path_for(site,'gp_enhanced'),left.query)
        for method in methods[1:]:
            right=runs[method];start_path=start_calendar_path(site,method)
            with np.load(start_path,allow_pickle=False) as calendar:
                np.testing.assert_array_equal(calendar['query'],right.query)
            right_starts,right_rule=starts_of(start_path,right.query)
            intersection=int((left.query&right.query).sum());union=int((left.query|right.query).sum())
            distances=np.min(np.abs(left_starts[:,None]-right_starts[None,:]),axis=1) if len(left_starts) and len(right_starts) else np.full(len(left_starts),np.inf)
            reverse=np.min(np.abs(right_starts[:,None]-left_starts[None,:]),axis=1) if len(left_starts) and len(right_starts) else np.full(len(right_starts),np.inf)
            overlap.append(dict(site=site,comparison=method,paid_hours=int(left.query.sum()),shared_paid_hours=intersection,
                paid_hour_jaccard=intersection/union if union else 1., enhanced_starts=len(left_starts),comparator_starts=len(right_starts),
                enhanced_starts_within_12h=float(np.mean(distances<=12)) if len(distances) else None,
                comparator_starts_within_12h=float(np.mean(reverse<=12)) if len(reverse) else None,
                enhanced_to_nearest_start_median_hours=float(np.median(distances)) if len(distances) else None,
                enhanced_start_rule=left_rule,comparator_start_rule=right_rule))
    tables=metric_tables(bundles)
    for name,frame in tables.items():frame.to_csv(OUT/f'{name}.csv',index=False)
    pd.DataFrame(overlap).to_csv(OUT/'calendar_overlap.csv',index=False)
    pd.DataFrame(runtimes).to_csv(OUT/'runtime.csv',index=False)
    write_json(OUT/'source_index.json',dict(plan_sha256=plan(),runner_sha256=base.sha(__file__),cells=index))
    print(tables['aggregate_metrics'].to_string(index=False),flush=True)
    print(pd.DataFrame(overlap).to_string(index=False),flush=True)

def verify():
    """Independent frozen chronological correction on every enhanced calendar."""
    records=[]
    for site in SITES:
        fleet=base.make_site(site,stage='test',q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
        source,local=base.frozen._load_source(ROOT/'results/main'/f'{site}_B48_periodic.npz',fleet)
        output=path_for(site,'gp_enhanced')
        with np.load(output,allow_pickle=False) as container:saved={key:container[key].copy() for key in container.files}
        access=saved['query']|saved['initmask'];maximum=0.
        for row,item in enumerate(local):
            left,right=item['start'],item['stop']
            with threadpool_limits(limits=2):
                expected=base.original.fit_predict_stream_baselines(item['X'],source['y'][left:right],336,
                    access[left:right],valid_x=item['valid'],model_names=('gp_stream',),**PARAMS)['gp_stream']
            expected=np.maximum(expected,0.);observed=saved['pred'][row,left:right]
            np.testing.assert_allclose(observed,expected,equal_nan=True,rtol=1e-12,atol=1e-12)
            maximum=max(maximum,float(np.nanmax(np.abs(observed-expected))))
        records.append(dict(site=site,status='PASS',models=len(local),maximum_absolute_prediction_difference=maximum,
            exact_attempted_budget=bool(np.array_equal(saved['block_spent'],saved['block_feasible']))))
        print('VERIFIED',site,maximum,flush=True)
    write_json(OUT/'verification.json',dict(status='PASS',method='independent fixed-mask replay using frozen correction implementation',records=records))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','smoke','real-smoke','run','summarize','verify'))
    parser.add_argument('--site',choices=SITES);parser.add_argument('--hours',type=int,default=1680)
    arguments=parser.parse_args()
    if arguments.action=='plan':print(plan())
    elif arguments.action=='smoke':smoke()
    elif arguments.action=='real-smoke':run_site(arguments.site or 'Manchester',stage='development',smoke_hours=arguments.hours)
    elif arguments.action=='run':
        for site in (arguments.site,) if arguments.site else SITES:run_site(site)
    elif arguments.action=='summarize':summarize()
    elif arguments.action=='verify':verify()
