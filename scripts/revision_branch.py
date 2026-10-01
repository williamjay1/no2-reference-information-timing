"""Post-review offline window-value diagnostic; never used to select a policy.

Snapshots come from the original B48 periodic history. Each diagnostic compares
12 request hours now with 12 hours starting 72 hours later. The planner sees only
three complete past context paths. Future reference values are used solely for
counterfactual branch assimilation and offline scoring.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'cache/numba'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json, hashlib, datetime
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr
from site_data import make_site
from calibration_engine import _advance, _observe
from acquisition_controller import _cov_advance, _cov_observe, _quadratic

OUT = ROOT/'results/revision/branch'
FREEZE = json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))
PROTOCOL = dict(stage='post-review offline diagnostic; not confirmatory; no policy selection',
    horizon_hours=168, request_length=12, delayed_start_hours=72,
    candidate_grid='site grid indices 0,168,336,..., fixed before outcomes',
    instruments='all Zep family physical instruments in the frozen future-period fleet',
    eligibility='current input valid, commissioned, three complete historical context paths; no site setup in [t-12,t+168]; at least 134 common scorable future hours',
    historical_lags=[192,216,288],
    actual_branch_updates='only the assigned 12-hour window; missing target or invalid sensor prevents update but attempt still counted',
    score_interval='t+1 through t+168 inclusive, same archived target-paired mask for all branches',
    predicted_value='average forecast variance reduction versus no-access path over t+1..t+168 using three past input scenarios',
    signed_difference='late-window loss minus immediate-window loss; positive means immediate window better',
    dependence='weekly points within four long instruments; descriptive correlations and sign agreement, no IID p-values',
    future_use='eligibility based on input/target availability and setup only; future reference values only branch assimilation and offline scoring',
    label_latency=0,
    selection_boundary='No case is retained or discarded using error magnitudes or the sign of the result')


@njit(cache=True)
def snapshot_history(phi, valid, y, access, m0, P0, F, Q, r, ready, points):
    n=len(y); p=len(m0)
    means=np.empty((len(points),p)); covs=np.empty((len(points),p,p))
    pred=np.full(n,np.nan); m=m0.copy(); P=P0.copy(); h=np.ones(p); k=0
    for t in range(ready,n):
        _advance(m,P,F,Q)
        if valid[t]:
            h[:-1]=phi[t]; pred[t]=max(0.,h@m)
        if k<len(points) and points[k]==t:
            means[k]=m; covs[k]=P; k+=1
        if valid[t] and access[t] and np.isfinite(y[t]):
            _observe(m,P,h,y[t],r)
    return means,covs,pred


@njit(cache=True)
def predicted_benefits(phi, valid, P0, F, Q, r, t, horizon=168):
    # Uses only phi[t] and indices strictly less than t. No y argument.
    benefits=np.zeros(2); h=np.ones(P0.shape[0])
    for lag in (192,216,288):
        origin=t-lag
        base=P0.copy(); alternatives=np.empty((2,len(h),len(h)))
        alternatives[0]=P0; alternatives[1]=P0
        for j in range(horizon+1):
            if j>0:
                _cov_advance(base,F,Q)
                for a in range(2):_cov_advance(alternatives[a],F,Q)
            if j>0 and valid[origin+j]:h[:-1]=phi[origin+j]
            else:h[:-1]=phi[t]
            if j>0:
                vb=_quadratic(base,h)
                for a in range(2):
                    benefits[a]+=max(vb-_quadratic(alternatives[a],h),0.)/(3*horizon)
            for a in range(2):
                start=72*a
                if start<=j<start+12:_cov_observe(alternatives[a],h,r)
    return benefits


@njit(cache=True)
def realized_branches(phi,valid,y,m0,P0,F,Q,r,t,horizon=168):
    pred=np.full((3,horizon),np.nan); delivered=np.zeros(2,np.int64)
    for a in range(3):
        m=m0.copy();P=P0.copy();h=np.ones(len(m));start=72*a
        for j in range(horizon+1):
            if j>0:_advance(m,P,F,Q)
            k=t+j
            if valid[k]:
                h[:-1]=phi[k]
                if j>0:pred[a,j-1]=max(0.,h@m)
                if a<2 and start<=j<start+12 and np.isfinite(y[k]):
                    _observe(m,P,h,y[k],r);delivered[a]+=1
    return pred,delivered


def write_once(path,text):
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:raise RuntimeError('Refusing to overwrite '+str(path))
    else:path.write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(PROTOCOL,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_freeze_sha256=hashlib.sha256((ROOT/'evidence/FREEZE.json').read_bytes()).hexdigest())
    if (OUT/'protocol.json').exists():
        prior=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        assert {k:v for k,v in prior.items() if k!='script_sha256'}=={k:v for k,v in protocol.items() if k!='script_sha256'}
    else:write_once(OUT/'protocol.json',json.dumps(protocol,indent=2))
    write_once(OUT/'run_manifest.json',json.dumps(protocol,indent=2))
    rows=[]; exclusions=[]; audits=[]
    for site in FREEZE['sites']:
        f=make_site(site,**FREEZE['calibrator']);n=len(f['y'])
        source=ROOT/'results/main'/f'{site}_B48_periodic.npz'
        with np.load(source,allow_pickle=False) as z:
            query=z['query'].astype(bool);original_pred=z['pred'].copy()
        access=query|f['initialization_mask']
        for d,(m,meta) in enumerate(zip(f['models'],f['metadata'])):
            if not meta['family'].lower().startswith('zep'):continue
            points=[];counts={k:0 for k in ('grid','not_ready','invalid_current','insufficient_past','setup','short_future','insufficient_scores')}
            for t in range(0,n,168):
                counts['grid']+=1
                if t<int(f['ready'][d]):counts['not_ready']+=1;continue
                if not m['valid'][t]:counts['invalid_current']+=1;continue
                if t<288:counts['insufficient_past']+=1;continue
                if t+168>=n:counts['short_future']+=1;continue
                if f['initialization_mask'][max(0,t-12):t+169].any():counts['setup']+=1;continue
                if f['score_masks'][d,t+1:t+169].sum()<134:counts['insufficient_scores']+=1;continue
                points.append(t)
            points=np.asarray(points,np.int64)
            if not len(points):
                exclusions.append(dict(site=site,instrument=meta['instrument'],eligible=0,**counts));continue
            means,covs,pred=snapshot_history(m['phi'],m['valid'],f['y'],access,m['m'],m['P'],m['F'],m['Q'],m['r'],int(f['ready'][d]),points)
            np.testing.assert_allclose(pred,original_pred[d],equal_nan=True,rtol=0,atol=0)
            saved_preds=[];saved_masks=[];saved_y=[];saved_util=[]
            for k,t in enumerate(points):
                utility=predicted_benefits(m['phi'],m['valid'],covs[k],m['F'],m['Q'],m['r'],int(t))
                branches,delivered=realized_branches(m['phi'],m['valid'],f['y'],means[k],covs[k],m['F'],m['Q'],m['r'],int(t))
                score=f['score_masks'][d,t+1:t+169]; yy=f['y'][t+1:t+169]
                assert np.isfinite(branches[:,score]).all()
                mse=np.mean((branches[:,score]-yy[score])**2,axis=1);rmse=np.sqrt(mse)
                rows.append(dict(site=site,instrument=meta['instrument'],deployment=meta['deployment'],index=int(t),time=str(f['times'][t]),n_scored=int(score.sum()),
                    predicted_benefit_now=utility[0],predicted_benefit_late=utility[1],predicted_difference=utility[0]-utility[1],
                    mse_now=mse[0],mse_late=mse[1],mse_no_access=mse[2],rmse_now=rmse[0],rmse_late=rmse[1],rmse_no_access=rmse[2],
                    realized_mse_difference=mse[1]-mse[0],realized_rmse_difference=rmse[1]-rmse[0],
                    delivered_now=int(delivered[0]),delivered_late=int(delivered[1]),attempts_per_branch=12))
                saved_preds.append(branches);saved_masks.append(score);saved_y.append(yy);saved_util.append(utility)
            key=str(meta['deployment']).replace('/','_')
            path=OUT/(key+'_branches.npz')
            assert not path.exists(),path
            np.savez_compressed(path,points=points,pred=np.asarray(saved_preds),score=np.asarray(saved_masks),y=np.asarray(saved_y),predicted_benefits=np.asarray(saved_util),
                metadata_json=json.dumps(meta),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            exclusions.append(dict(site=site,instrument=meta['instrument'],eligible=len(points),**counts))
            audits.append(dict(site=site,instrument=meta['instrument'],history_exactly_matches_original=True,points=len(points)))
            print(json.dumps(audits[-1]),flush=True)
    frame=pd.DataFrame(rows)
    assert len(frame)>0,'No eligible branches'
    frame.to_csv(OUT/'all_points.csv',index=False)
    pd.DataFrame(exclusions).to_csv(OUT/'eligibility.csv',index=False)
    summaries=[]
    groups=[('pooled',frame)]+[(str(k),g) for k,g in frame.groupby('instrument',sort=True)]
    for name,g in groups:
        px=g.predicted_difference.to_numpy();ry=g.realized_mse_difference.to_numpy()
        non_ties=(abs(px)>1e-12)&(abs(ry)>1e-12)
        chosen=np.where(px>=0,g.mse_now,g.mse_late)
        summaries.append(dict(instrument=name,n=len(g),spearman_mse=float(spearmanr(px,ry).statistic),spearman_rmse=float(spearmanr(px,g.realized_rmse_difference).statistic),
            sign_agreement=float(np.mean(np.sign(px[non_ties])==np.sign(ry[non_ties]))),non_ties=int(non_ties.sum()),
            predicted_now_fraction=float(np.mean(px>=0)),realized_now_fraction=float(np.mean(ry>=0)),
            mean_realized_mse_difference=float(np.mean(ry)),mean_realized_rmse_difference=float(g.realized_rmse_difference.mean()),
            mean_predicted_choice_mse=float(np.mean(chosen)),mean_always_now_mse=float(g.mse_now.mean()),mean_always_late_mse=float(g.mse_late.mean()),
            mean_oracle_mse=float(np.minimum(g.mse_now,g.mse_late).mean()),mean_no_access_mse=float(g.mse_no_access.mean())))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    write_once(OUT/'audit.json',json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),history_checks=audits,
        total_points=len(frame),physical_instruments=int(frame.instrument.nunique()),all_attempt_counts=12,
        utility_has_no_y_argument=True,all_context_indices_strictly_before_origin=True,not_used_for_policy_selection=True),indent=2))
    print('COMPLETE',json.dumps(summaries[0]),flush=True)


if __name__=='__main__':main()
