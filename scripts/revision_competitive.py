"""B4 post-review supplement: complete frozen calendar crossing and shared-site tuning.

Never modifies original scripts, source data, or results. No adaptive calendar is
retuned here. The shared-site parameter search uses pre-2021 Manchester only.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for key in ('TEMP', 'TMP'):
    os.environ[key] = str(ROOT/'temp')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'cache/numba')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import competitive_stream_baselines as original
import run_main_baselines as frozen
from site_data import make_site
from fixed_schedules import fixed_site_mask
from statistical_analysis import load_run

OUT = ROOT/'results/revision/competitive'
SITES = ('Manchester', 'London', 'York')
POLICIES = ('periodic', 'risk_fixed', 'risk')
GP_GRID = [dict(gp_length_scale=l, gp_noise=n) for l in (.5, 1., 2.) for n in (.1, .5)]
HGB_GRID = [dict(max_leaf_nodes=l, l2_regularization=a) for l in (7, 15) for a in (1., 10.)]
ORIGINAL_PARAMS = dict(gp=dict(gp_length_scale=2., gp_noise=.5),
                       hgb=dict(max_leaf_nodes=7, l2_regularization=1.))
PLAN = dict(stage='Post-review supplemental analysis, after original test results were known',
    budget=48, sites=list(SITES), calendars=list(POLICIES), missing=[['gp','risk_fixed'],['gp','risk'],['hgb','risk_fixed']],
    original_parameters=ORIGINAL_PARAMS, gp_grid=GP_GRID, hgb_grid=HGB_GRID,
    development=dict(site='Manchester', end_exclusive='2021-01-01', phase=0., budget=48,
                     init_hours=336, minimum_initial_pairs=72, minimum_scored_hours=672,
                     shared_setup=True, block_hours=672, starts=4),
    selection='Minimum physical-instrument macro RMSE; tied values retain grid order',
    common=dict(guard_width=8., weather=True, max_pairs=512, window_hours=2160,
                hgb_iterations=80, hgb_learning_rate=.05, hgb_min_samples_leaf=12,
                hgb_max_bins=64, hgb_seed=20260930, prediction_floor=0., threads=2),
    tuned_evaluation='One development-selected configuration per model, all three frozen B48 calendars; reuse existing cells when parameters coincide')


def dump(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    return frozen._hash(path)


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'revision_plan.json'
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload['plan'] != PLAN:
            raise RuntimeError('Existing prospective revision plan differs')
    else:
        payload = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), plan=PLAN,
            original_freeze_sha256=sha(ROOT/'evidence/FREEZE.json'),
            code_sha256={n: sha(ROOT/'scripts'/n) for n in ('competitive_stream_baselines.py','run_main_baselines.py','calibration_engine.py','site_data.py','study_data.py','fixed_schedules.py')},
            data_sha256={n: sha(ROOT/'datasets'/n) for n in ('no2_hourly_oob.parquet','reference_no2_ratified.parquet')})
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    for name, digest in payload['code_sha256'].items():
        assert sha(ROOT/'scripts'/name) == digest, name
    for name, digest in payload['data_sha256'].items():
        assert sha(ROOT/'datasets'/name) == digest, name
    return sha(path)


def hgb_stream(X, y, init_end, query, valid, params):
    """Same event chronology as original HGB, with only leaves/L2 configurable."""
    begin = perf_counter()
    X = np.asarray(X, float).copy(); y = np.asarray(y, float)
    valid = np.asarray(valid, bool) & np.isfinite(X[:, 0])
    query = np.asarray(query, bool)
    X[~np.isfinite(X)] = np.nan
    n = len(y); prediction = np.full(n, np.nan)
    initial = np.flatnonzero(valid[:init_end])
    pairs = initial[np.isfinite(y[initial])]
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    fitted = imputer.fit_transform(X[initial])
    scaler = StandardScaler().fit(fitted)
    Z = scaler.transform(imputer.transform(X))
    prep = perf_counter()-begin
    diagnostic = dict(runtime_seconds=0., shared_preprocess_seconds=prep, fits=0,
                      updates=0, max_training_pairs=0, max_training_age_hours=0)

    def fit(indices, next_hour):
        chosen = np.asarray(indices, int)[-512:]
        started = perf_counter()
        model = HistGradientBoostingRegressor(learning_rate=.05, max_iter=80,
            max_leaf_nodes=params['max_leaf_nodes'], min_samples_leaf=12, max_bins=64,
            l2_regularization=params['l2_regularization'], early_stopping=False,
            random_state=20260930)
        model.fit(Z[chosen], y[chosen])
        diagnostic['runtime_seconds'] += perf_counter()-started
        diagnostic['fits'] += 1
        diagnostic['max_training_pairs'] = max(diagnostic['max_training_pairs'], len(chosen))
        diagnostic['max_training_age_hours'] = max(diagnostic['max_training_age_hours'], int(next_hour-chosen[0]))
        return model

    with threadpool_limits(limits=2):
        model = fit(pairs, init_end)
        observed = pairs.tolist()
        events = np.flatnonzero(query & valid & (np.arange(n)>=init_end))
        next_start = init_end
        for event in np.r_[events, n-1]:
            stop = int(event)+1
            if stop < next_start:
                continue
            current = np.flatnonzero(valid[next_start:stop])+next_start
            if len(current):
                started=perf_counter(); prediction[current]=model.predict(Z[current])
                diagnostic['runtime_seconds'] += perf_counter()-started
            if event >= next_start and query[event] and valid[event] and np.isfinite(y[event]):
                observed.append(int(event))
                observed = [j for j in observed if j >= stop-2160]
                model=fit(observed, stop); diagnostic['updates'] += 1
            next_start=stop
    diagnostic['bundle_seconds'] = perf_counter()-begin
    return prediction, diagnostic


def fit_model(model, X, y, query, valid, params):
    if model=='hgb' and params != ORIGINAL_PARAMS['hgb']:
        return hgb_stream(X,y,336,query,valid,params)
    name = 'gp_stream' if model=='gp' else 'hgb_stream'
    options = params if model=='gp' else {}
    result=original.fit_predict_stream_baselines(X,y,336,query,valid_x=valid,
        model_names=(name,),max_pairs=512,**options)
    d=original.LAST_RUN_DIAGNOSTICS
    record=dict(**d['models'][name], shared_preprocess_seconds=d['shared_preprocess_seconds'],
                bundle_seconds=d['total_seconds'])
    return result[name],record


def source_path(site, policy):
    return ROOT/'results/main'/f'{site}_B48_{policy}.npz'


def existing_path(site, model, policy):
    if model=='ar':
        return source_path(site,policy)
    method=model+'_stream'+('' if policy=='periodic' else '_'+policy)
    return ROOT/'results/main'/f'{site}_B48_{method}.npz'


def provenance(source, model, params, tag):
    return dict(revision_plan_sha256=plan(), source_npz=str(source), source_sha256=sha(source),
                model=model, parameters=params, tag=tag,
                runner_sha256=sha(Path(__file__)), original_freeze_sha256=sha(ROOT/'evidence/FREEZE.json'))


def run_cell(source, fleet, model, params, output, tag):
    prov=provenance(source,model,params,tag)
    if output.exists():
        with np.load(output,allow_pickle=False) as z:
            assert json.loads(z['revision_provenance_json'].item())==prov,output
        load_run(output)
        print('RESUME',output.name,flush=True)
        return
    data,local=frozen._load_source(source,fleet)
    prediction=np.full(data['pred'].shape,np.nan)
    acquired=data['query'].astype(bool)|data['initmask'].astype(bool)
    records=[]
    for j,item in enumerate(local):
        left,right=item['start'],item['stop']
        p,d=fit_model(model,item['X'],data['y'][left:right],acquired[left:right],item['valid'],params)
        prediction[j,left:right]=p
        records.append(dict(deployment=item['row']['deployment'],**d))
        print(output.stem,item['row']['deployment'],'done',flush=True)
    saved=frozen._shared_arrays(data)
    saved['pred']=np.maximum(prediction,0.)
    assert np.isfinite(saved['pred'][saved['score_masks']]).all()
    runtime=sum(d['runtime_seconds']+d['shared_preprocess_seconds'] for d in records)
    saved['runtime_seconds']=np.asarray(runtime)
    saved['runtime_json']=np.asarray(dump(records))
    saved['revision_provenance_json']=np.asarray(dump(prov))
    saved['manifest_sha256']=np.asarray(prov['original_freeze_sha256'])
    finite=np.isfinite(saved['y'])
    saved['access_json']=np.asarray(dump(dict(initialization_attempts=int(saved['initmask'].sum()),
        initialization_successes=int((saved['initmask']&finite).sum()),
        additional_attempts=int(saved['query'].sum()),additional_successes=int((saved['query']&finite).sum()),
        runtime_seconds=runtime)))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,**saved)
    temporary.replace(output)
    load_run(output)
    for key in frozen._shared_arrays(data):
        assert np.array_equal(saved[key],data[key],equal_nan=True) if saved[key].dtype.kind in 'fc' else np.array_equal(saved[key],data[key]),key
    print('SAVED',output.name,'seconds',round(runtime,2),flush=True)


def cross(sites):
    plan()
    for site in sites:
        fleet=make_site(site,stage='test',q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
        for model,policy in PLAN['missing']:
            output=OUT/'frozen'/f'{site}_B48_{model}_stream_{policy}.npz'
            run_cell(source_path(site,policy),fleet,model,ORIGINAL_PARAMS[model],output,'original frozen parameters; missing factorial cell')
    if all((OUT/'frozen'/f'{s}_B48_{m}_stream_{p}.npz').exists() for s in SITES for m,p in PLAN['missing']):
        cells=[]
        for site in SITES:
            for model in ('ar','gp','hgb'):
                for policy in POLICIES:
                    p=existing_path(site,model,policy)
                    if not p.exists():
                        p=OUT/'frozen'/p.name
                    assert p.exists(),p
                    cells.append(dict(site=site,model=model,policy=policy,path=str(p),sha256=sha(p),parameters=ORIGINAL_PARAMS.get(model)))
        (OUT/'matrix_frozen.json').write_text(json.dumps(dict(plan_sha256=plan(),cells=cells),indent=2),encoding='utf-8')
        summarize(OUT/'matrix_frozen.json','frozen')


def instrument_rows(path):
    run=load_run(path)
    rows=[]
    for instrument in sorted({r['instrument'] for r in run.metadata}):
        ix=[i for i,r in enumerate(run.metadata) if r['instrument']==instrument]
        mask=run.score_masks[ix]; counts=mask.sum(axis=0); active=counts>0
        e=np.where(mask,run.pred[ix]-run.y[None,:],0.)
        se=(e*e).sum(axis=0)/np.maximum(counts,1)
        ae=np.abs(e).sum(axis=0)/np.maximum(counts,1)
        rows.append(dict(site=run.site,instrument=instrument,score_hours=int(active.sum()),
            squared_error_sum=float(se[active].sum()),absolute_error_sum=float(ae[active].sum()),
            rmse=float(np.sqrt(se[active].mean())),mae=float(ae[active].mean())))
    return rows


def summarize(matrix,tag):
    cells=json.loads(matrix.read_text(encoding='utf-8'))['cells']
    rows=[]
    for cell in cells:
        rows.extend([dict(model=cell['model'],policy=cell['policy'],path=cell['path'],**r) for r in instrument_rows(Path(cell['path']))])
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/f'{tag}_instrument.csv',index=False)
    summary=frame.groupby(['model','policy'],sort=False).agg(rmse=('rmse','mean'),mae=('mae','mean'),instruments=('instrument','nunique'),instrument_hours=('score_hours','sum')).reset_index()
    summary.to_csv(OUT/f'{tag}_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


def development():
    plan()
    fleet=make_site('Manchester',stage='development',q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
    assert fleet['times'].max()<pd.Timestamp('2021-01-01')
    query,audit=fixed_site_mask(len(fleet['y']),int(min(fleet['ready'])),fleet['initialization_mask'],budget=48,phase=0.)
    source=OUT/'development_calendar.npz'
    if not source.exists():
        np.savez_compressed(source,pred=np.full(fleet['score_masks'].shape,np.nan),y=fleet['y'],
            times=fleet['times'].astype(str).to_numpy(dtype='U19'),score_masks=fleet['score_masks'],query=query,
            initmask=fleet['initialization_mask'],metadata_json=json.dumps(fleet['metadata']),
            calendar_only=np.asarray(True),calendar_audit_json=np.asarray(dump(audit)))
    else:
        with np.load(source,allow_pickle=False) as z:
            assert np.array_equal(z['query'],query) and np.array_equal(z['initmask'],fleet['initialization_mask'])
    configs=[]
    for model,grid in [('gp',GP_GRID),('hgb',HGB_GRID)]:
        for k,params in enumerate(grid):
            output=OUT/'development'/f'Manchester_B48_{model}_grid{k}.npz'
            run_cell(source,fleet,model,params,output,'shared-site pre-2021 development grid')
            rows=instrument_rows(output)
            with np.load(output,allow_pickle=False) as z:
                runtime=float(z['runtime_seconds'])
            configs.append(dict(model=model,grid_index=k,parameters=params,path=str(output),
                macro_rmse=float(np.mean([r['rmse'] for r in rows])),macro_mae=float(np.mean([r['mae'] for r in rows])),
                runtime_seconds=runtime,instruments=rows))
            (OUT/'development_grid.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
            pd.DataFrame([dict(model=c['model'],grid_index=c['grid_index'],parameters_json=dump(c['parameters']),macro_rmse=c['macro_rmse'],macro_mae=c['macro_mae'],runtime_seconds=c['runtime_seconds']) for c in configs]).to_csv(OUT/'development_grid.csv',index=False)
    selected={m:min((c for c in configs if c['model']==m),key=lambda c:(c['macro_rmse'],c['grid_index'])) for m in ('gp','hgb')}
    payload=dict(plan_sha256=plan(),selected_utc=datetime.now(timezone.utc).isoformat(),selection_data_end=str(fleet['times'].max()),selected=selected)
    path=OUT/'shared_development_selection.json'
    if path.exists():
        prior=json.loads(path.read_text(encoding='utf-8'))
        assert {m:c['parameters'] for m,c in prior['selected'].items()}=={m:c['parameters'] for m,c in selected.items()}
    else:
        path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('SELECTED',dump({m:c['parameters'] for m,c in selected.items()}),flush=True)


def selected(sites):
    selection=json.loads((OUT/'shared_development_selection.json').read_text(encoding='utf-8'))
    assert selection['plan_sha256']==plan()
    base=json.loads((OUT/'matrix_frozen.json').read_text(encoding='utf-8'))['cells']
    for site in sites:
        fleet=None
        for model in ('gp','hgb'):
            params=selection['selected'][model]['parameters']
            if params==ORIGINAL_PARAMS[model]:
                print('REUSE ORIGINAL',site,model,flush=True);continue
            if fleet is None:
                fleet=make_site(site,stage='test',q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
            for policy in POLICIES:
                output=OUT/'shared_selected'/f'{site}_B48_{model}_shareddev_{policy}.npz'
                run_cell(source_path(site,policy),fleet,model,params,output,'shared-site development-selected supplemental configuration')
    cells=[]
    for cell in base:
        model=cell['model'];site=cell['site'];policy=cell['policy']
        if model=='ar' or selection['selected'][model]['parameters']==ORIGINAL_PARAMS[model]:
            cells.append(dict(**cell,reused_original=True))
        else:
            p=OUT/'shared_selected'/f'{site}_B48_{model}_shareddev_{policy}.npz'
            if not p.exists():return
            cells.append(dict(site=site,model=model,policy=policy,path=str(p),sha256=sha(p),parameters=selection['selected'][model]['parameters'],reused_original=False))
    matrix=OUT/'matrix_shared_selected.json'
    matrix.write_text(json.dumps(dict(plan_sha256=plan(),selection_sha256=sha(OUT/'shared_development_selection.json'),cells=cells),indent=2),encoding='utf-8')
    summarize(matrix,'shared_selected')


def smoke():
    rng=np.random.default_rng(20261001);n=480;init=336
    X=rng.normal(size=(n,3));X[:,0]+=10.;X[7::41,1]=np.nan;X[9::67,0]=np.nan
    y=1.3*np.nan_to_num(X[:,0])+rng.normal(size=n)
    valid=np.isfinite(X[:,0]);query=np.zeros(n,bool);query[345::37]=True;query[-1]=True
    base,diag=hgb_stream(X,y,init,query,valid,ORIGINAL_PARAMS['hgb'])
    expected=original.fit_predict_stream_baselines(X,y,init,query,valid_x=valid,model_names=('hgb_stream',))['hgb_stream']
    np.testing.assert_allclose(base,expected,rtol=0,atol=0,equal_nan=True)
    params=dict(max_leaf_nodes=15,l2_regularization=10.)
    baseline,d=hgb_stream(X,y,init,query,valid,params)
    hidden=(np.arange(n)>=init)&~query
    changed=y.copy();changed[hidden]=1e8
    np.testing.assert_allclose(baseline,hgb_stream(X,changed,init,query,valid,params)[0],rtol=0,atol=0,equal_nan=True)
    changed=y.copy();changed[345]+=1e5
    np.testing.assert_allclose(baseline[:346],hgb_stream(X,changed,init,query,valid,params)[0][:346],rtol=0,atol=0,equal_nan=True)
    future=X.copy();future[400:]+=100.
    np.testing.assert_allclose(baseline[:400],hgb_stream(future,y,init,query,valid,params)[0][:400],rtol=0,atol=0,equal_nan=True)
    assert d['updates']==int((query&valid&np.isfinite(y)&(np.arange(n)>=init)).sum())
    n2=2600;X2=rng.normal(size=(n2,3));y2=rng.normal(size=n2);q2=np.zeros(n2,bool);q2[350::450]=True;q2[-1]=True
    _,d2=hgb_stream(X2,y2,init,q2,np.ones(n2,bool),params)
    assert d2['max_training_age_hours']<=2160 and d2['max_training_pairs']<=512
    payload=dict(status='PASS',checks=['original HGB exact parity','hidden label invariance','current queried target affects no current prediction','future covariate invariance','final query counted once','90-calendar-day age and 512-pair cap'],runner_sha256=sha(Path(__file__)))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'smoke_checks.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(dump(payload),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('plan','smoke','cross','development','selected'))
    parser.add_argument('--sites',nargs='+',choices=SITES,default=list(SITES))
    args=parser.parse_args();start=perf_counter()
    if args.stage=='plan':print(plan(),flush=True)
    elif args.stage=='smoke':smoke()
    elif args.stage=='cross':cross(args.sites)
    elif args.stage=='development':development()
    elif args.stage=='selected':selected(args.sites)
    print('COMPLETE',args.stage,'wall_seconds',round(perf_counter()-start,3),flush=True)


if __name__=='__main__':
    main()
