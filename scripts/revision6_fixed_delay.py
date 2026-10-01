"""Expanded early-only selection and a complete fixed-delivery service matrix.

The revision is retrospective. Earlier controllers and outputs are read-only.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
for key, value in {'NUMBA_CACHE_DIR':ROOT/'cache/numba', 'MPLCONFIGDIR':ROOT/'temp/mpl',
                   'TEMP':ROOT/'temp', 'TMP':ROOT/'temp'}.items():
    os.environ.setdefault(key, str(value))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from site_data import make_site
from revision_access import CONFIG, macro
import revision_access_controller as prior
import revision4_delay_controller as aware
from statistical_analysis import load_run, build_site_bundle, paired_calendar_bootstrap, metric_tables

OUT = ROOT/'results/revision6/fixed_delay'
EV = ROOT/'evidence/revision6'
SITES = ('Manchester','London','York')
DELAYS = (0,24,72,168)
POLICIES = ('periodic','original','aware_unretuned','aware_retained','aware_expanded')

def tag(h, threshold):
    return f'H{h}_T{str(float(threshold)).replace(".","p")}'

def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    EV.mkdir(parents=True, exist_ok=True)
    spec = dict(stage='retrospective exploratory sixth revision; earlier later-period results already known',
        horizons=[168,192,240,336], thresholds=[.75,1.,1.25,1.5],
        selection='minimum pre-2021 Manchester common-support macro physical-instrument RMSE; ties horizon then threshold',
        selection_uncertainty='1000 paired circular seven-day calendar-block resamples within early Manchester quarters, conditional on candidate streams; not retuned-policy or new-site uncertainty',
        selection_seed=20261006, selection_delay=72, selection_shared=True,
        retained_parameters=dict(horizon=240,threshold=1.25), original_parameters=dict(horizon=168,threshold=1.25),
        delays=list(DELAYS), shared=[True,False], policies=list(POLICIES),
        score_support='fixed per-deployment ready+168h for every delivery and policy; full masks also archived',
        contract=dict(budget=48,block=672,max_starts=4,window_length=12),
        calibrator=CONFIG['calibrator'], no_evaluation_selection=True,
        source_sha256={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
            for name in ('revision_access_controller.py','revision4_delay_controller.py','site_data.py')})
    path=EV/'fixed_delay_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == spec
    else:
        path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    return spec

def common_mask(f):
    return f['score_masks'] & (np.arange(len(f['y']))[None,:] >= f['ready'][:,None]+168)

def execute(f,h,threshold,delay,shared,controller=aware):
    return controller.run_site(f['models'],f['y'],f['ready'],f['initialization_mask'],
        budget=48,block=672,max_starts=4,policy='risk_fixed',horizon=h,
        threshold_scale=threshold,fixed_length=12,delay=delay,shared_setup=shared)

def save(path,f,z,meta):
    common=common_mask(f)
    pred=np.maximum(z['pred'],0.)
    assert np.isfinite(pred[common]).all()
    pack=dict(pred=pred,y=f['y'],times=np.asarray(f['times'].astype(str),dtype='U19'),score_masks=common,
        full_score_masks=f['score_masks'],ready_indices=f['ready'],query=z['query'],
        initmask=f['initialization_mask'],metadata_json=json.dumps(f['metadata']),
        revision_provenance_json=json.dumps(meta,sort_keys=True),
        access_json=json.dumps(dict(additional_attempts=int(z['query'].sum()),runtime_seconds=z['runtime_seconds'])))
    for key in ('var','start_mask','forced_query','chosen_lengths','candidate_scores',
                'block_spent','block_target','budget_shortfall','starts'):
        if key in z: pack[key]=z[key]
    np.savez_compressed(path,**pack)

def development(spec):
    selected_path=OUT/'selected_expanded.json'
    if selected_path.exists():
        return json.loads(selected_path.read_text())
    f=make_site('Manchester',stage='development',**CONFIG['calibrator'])
    rows=[]; runs={}
    for h in spec['horizons']:
        for threshold in spec['thresholds']:
            name=tag(h,threshold)
            path=OUT/f'Manchester_B48_development_{name}.npz'
            if not path.exists():
                z=execute(f,h,threshold,72,True)
                save(path,f,z,dict(stage='development only',horizon=h,threshold=threshold,delay=72,shared=True))
            run=load_run(path)
            with np.load(path,allow_pickle=False) as p:
                access=json.loads(p['access_json'].item())
            row=dict(horizon=h,threshold=threshold,method=name,
                rmse=macro(run.pred,run.y,run.score_masks,run.metadata),**access)
            runs[name]=run; rows.append(row)
            print('DEVELOPMENT',json.dumps(row),flush=True)
    names=tuple(runs)
    b=build_site_bundle(runs,methods=names)
    boot=paired_calendar_bootstrap([b],n_replicates=1000,block_days=7,seed=spec['selection_seed'])
    chosen=np.argmin(boot['primary'],axis=1)
    for i,row in enumerate(rows):
        row['selection_frequency']=float((chosen==i).mean())
        row['rmse_ci_low'],row['rmse_ci_high']=np.quantile(boot['primary'][:,i],[.025,.975]).tolist()
    pd.DataFrame(rows).to_csv(OUT/'expanded_development_candidates.csv',index=False)
    np.savez_compressed(OUT/'development_selection_bootstrap.npz',values=boot['primary'],methods=np.asarray(names))
    (OUT/'development_bootstrap_diagnostics.json').write_text(json.dumps(boot['diagnostics'],indent=2),encoding='utf-8')
    selected=dict(min(rows,key=lambda r:(r['rmse'],r['horizon'],r['threshold'])))
    selected.update(selection_data_end_exclusive='2021-01-01',evaluation_used=False,
        inference_scope=spec['selection_uncertainty'],candidate_count=16)
    selected_path.write_text(json.dumps(selected,indent=2),encoding='utf-8')
    print('SELECTED',json.dumps(selected),flush=True)
    return selected

def run_matrix(spec,selected):
    rows=[]
    for site in SITES:
        f=make_site(site,**CONFIG['calibrator'])
        with np.load(ROOT/'results/main'/f'{site}_B48_periodic.npz',allow_pickle=False) as p:
            q=p['query'].copy()
            np.testing.assert_array_equal(p['y'],f['y'])
            np.testing.assert_array_equal(p['score_masks'],f['score_masks'])
        for shared in (True,False):
            for delay in DELAYS:
                for policy in POLICIES:
                    path=OUT/f'{site}_B48_{policy}_d{delay}_shared{int(shared)}.npz'
                    h=168; threshold=1.25
                    if policy=='aware_retained': h=240
                    if policy=='aware_expanded': h=int(selected['horizon']);threshold=float(selected['threshold'])
                    if not path.exists():
                        if policy=='periodic':
                            z=prior.run_site_fixed(f['models'],f['y'],f['ready'],f['initialization_mask'],q,
                                delay=delay,shared_setup=shared)
                        else:
                            z=execute(f,h,threshold,delay,shared,prior if policy=='original' else aware)
                        assert int(z['query'].sum())==int(q.sum()), (site,delay,shared,policy)
                        if policy!='periodic' and 'starts' in z: assert np.max(z['starts'])<=4
                        save(path,f,z,dict(stage='retrospective evaluation; no parameter selection',
                            site=site,policy=policy,delay=delay,shared=shared,horizon=h,threshold=threshold))
                    run=load_run(path)
                    with np.load(path,allow_pickle=False) as p:
                        access=json.loads(p['access_json'].item())
                    row=dict(site=site,shared=shared,delay=delay,policy=policy,
                        rmse=macro(run.pred,run.y,run.score_masks,run.metadata),**access)
                    rows.append(row);print('EVALUATION',json.dumps(row),flush=True)
                if delay==0:
                    with np.load(OUT/f'{site}_B48_original_d0_shared{int(shared)}.npz') as a, np.load(OUT/f'{site}_B48_aware_unretuned_d0_shared{int(shared)}.npz') as b:
                        for key in ('pred','query','candidate_scores'):
                            np.testing.assert_allclose(a[key],b[key],rtol=0,atol=0,equal_nan=True)
    pd.DataFrame(rows).to_csv(OUT/'matrix_site_results.csv',index=False)

def summarize():
    tables=[]; support=[]
    for shared in (True,False):
        for delay in DELAYS:
            bundles=[]
            for site in SITES:
                runs={name:load_run(OUT/f'{site}_B48_{name}_d{delay}_shared{int(shared)}.npz') for name in POLICIES}
                bundles.append(build_site_bundle(runs,methods=POLICIES))
            metrics=metric_tables(bundles)
            frame=metrics['aggregate_metrics']
            frame['delay']=delay;frame['shared']=shared;tables.append(frame)
            n=sum(b['active'].sum() for b in bundles)
            assert n==59674, n
            support.append(dict(delay=delay,shared=shared,instrument_hours=int(n),instruments=18))
    pd.concat(tables,ignore_index=True).to_csv(OUT/'matrix_aggregate_metrics.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'matrix_support.csv',index=False)
    print('MATRIX COMPLETE: 120 site-policy-delivery-sharing cells; identical 59,674 instrument-hours',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','matrix','summarize','all'],default='all')
    args=ap.parse_args();spec=protocol()
    if args.stage in ('development','all'): selected=development(spec)
    else: selected=json.loads((OUT/'selected_expanded.json').read_text())
    if args.stage in ('matrix','all'):run_matrix(spec,selected)
    if args.stage in ('summarize','all'):summarize()

if __name__=='__main__':main()
