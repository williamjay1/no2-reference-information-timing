"""Post-review descriptive environmental and instrument diagnostics.

Reads unchanged B48 periodic/risk_fixed archives. No policy/model selection.
All new outputs remain under results/revision/environment.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'cache/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'temp/matplotlib'))
import argparse, hashlib, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from statistical_analysis import load_run, check_comparable
from study_data import read_data
from site_data import make_site
from calibration_engine import _advance, _observe
from acquisition_controller import _prepare

OUT=ROOT/'results/revision/environment'
SITES=('Manchester','London','York')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'DESIGN.json'
    design=dict(stage='Second-review descriptive extension after earlier test results were inspected; not preregistration.',
        scope='B48 unchanged AR periodic versus unchanged AR fixed-risk only; no threshold/model/policy selection using evaluation losses.',
        endpoints='Per-physical-instrument RMSE and fixed-risk minus periodic RMSE on identical hours; average simultaneous-channel squared losses before RMSE.',
        environment='Device T and RH after source Error/physical-range screens; no reference weather substitution and no imputation for environmental membership.',
        threshold_rule='For each deployment/channel, q25 and q75 from finite weather in its 336-hour initialization window, requiring at least 72 finite weather values. Low <= q25; middle >q25 and <=q75; high >q75; missing and insufficient-initialization supported bins separately retained.',
        threshold_interpretation='Relative to that recorded deployment evaluation initialization environment; not a universal hot/humid threshold and not validated environment-triggered scheduling.',
        elapsed_time='Hours after the 336-hour evaluation initialization (replacement segments restart); bins <28, 28-<90, 90-<180, >=180 days. This is not electrode age or known physical commissioning age.',
        support='All supported cells exported, including n<72; macro summaries restricted to instrument-cells with at least 72 matched hours. Fractional instrument-hour weights handle simultaneous channel weather bins.',
        heterogeneity='Group by site, source product family and the four established long-record Zep versus 14 short-record Manchester instruments. No population inference, cross-device classification, or environmental causality.',
        trace='Replay fixed archived calendar using same initialized state and exact scalar Kalman operations; check all clipped forecasts and scored supports against immutable archive. Save pre-update and post-update state and physical-unit affine coefficients.',
        coefficient_interpretation='Calibration coefficients of manufacturer concentration product, not raw electrochemical parameters. Traces describe statistical adaptation; correlations with environment do not demonstrate electrode aging or cross-sensitivity mechanisms.',
        trace_correlations='Hourly within-deployment Spearman associations, descriptive with no independence-based p values; same scoring masks and observed finite environment only. Also export weekly means for visual inspection.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for s in SITES for m in ('periodic','risk_fixed') for p in [ROOT/f'results/main/{s}_B48_{m}.npz']},
        frozen_utc=datetime.now(timezone.utc).isoformat())
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['source_sha256']!=design['source_sha256']:
            raise RuntimeError('Frozen diagnostic sources changed')
    else:
        path.write_text(json.dumps(design,indent=2),encoding='utf-8')
    return design

def site_inputs(site):
    p=load_run(ROOT/f'results/main/{site}_B48_periodic.npz')
    f=load_run(ROOT/f'results/main/{site}_B48_risk_fixed.npz')
    check_comparable({'periodic':p,'risk_fixed':f})
    cfg=json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))
    fleet=make_site(site,**cfg['calibrator'])
    assert p.times.equals(fleet['times'])
    assert np.array_equal(p.score_masks,fleet['score_masks'])
    assert p.metadata==fleet['metadata']
    return p,f,fleet

def _condition_masks(X,metadata,n):
    offset=int(metadata['local_offset']); ready=offset+336
    age=(np.arange(n)-ready)/24.
    conditions={'all':np.ones(n,bool)}
    for low,high,label in [(0,28,'elapsed_0_28d'),(28,90,'elapsed_28_90d'),(90,180,'elapsed_90_180d'),(180,np.inf,'elapsed_ge180d')]:
        conditions[label]=(age>=low)&(age<high)
    thresholds=[]
    for j,variable in [(1,'T'),(2,'RH')]:
        weather=X[:,j]
        a=weather[offset:ready]; a=a[np.isfinite(a)]
        supported=len(a)>=72
        q25,q75=np.quantile(a,[.25,.75]) if supported else (np.nan,np.nan)
        thresholds.append(dict(variable=variable,init_finite=int(len(a)),init_distinct=int(np.unique(a).size),supported=supported,q25=q25,q75=q75,degenerate_equal_quartiles=bool(supported and q25==q75)))
        finite=np.isfinite(weather)
        conditions[f'{variable}_missing']=~finite
        if supported:
            conditions[f'{variable}_low']=finite&(weather<=q25)
            conditions[f'{variable}_middle']=finite&(weather>q25)&(weather<=q75)
            conditions[f'{variable}_high']=finite&(weather>q75)
            conditions[f'{variable}_unsupported_init']=np.zeros(n,bool)
        else:
            for category in ('low','middle','high'):
                conditions[f'{variable}_{category}']=np.zeros(n,bool)
            conditions[f'{variable}_unsupported_init']=finite
    return conditions,thresholds

def analyze():
    freeze(); cells=[]; threshold_rows=[]; coverage_rows=[]; instrument_meta=[]
    for site in SITES:
        p,f,fleet=site_inputs(site)
        errors={'periodic':(p.pred-p.y)**2,'risk_fixed':(f.pred-f.y)**2}
        ids=sorted({r['instrument'] for r in p.metadata})
        row_conditions=[]
        for i,(row,X) in enumerate(zip(p.metadata,fleet['Xs'])):
            conditions,thresholds=_condition_masks(X,row,len(p.y));row_conditions.append(conditions)
            for threshold in thresholds:
                threshold_rows.append(dict(site=site,instrument=row['instrument'],deployment=row['deployment'],family=row['family'],**threshold))
            sc=p.score_masks[i]
            coverage_rows.append(dict(site=site,instrument=row['instrument'],deployment=row['deployment'],family=row['family'],channel_scored_hours=int(sc.sum()),T_observed_hours=int((sc&np.isfinite(X[:,1])).sum()),RH_observed_hours=int((sc&np.isfinite(X[:,2])).sum()),both_observed_hours=int((sc&np.isfinite(X[:,1])&np.isfinite(X[:,2])).sum())))
        for instrument in ids:
            ix=[i for i,r in enumerate(p.metadata) if r['instrument']==instrument]
            family=p.metadata[ix[0]]['family']; category='long_record_Zep' if family=='Zep' else 'short_record_nonZep'
            base=p.score_masks[ix];count=base.sum(axis=0); usable=count>0
            instrument_meta.append(dict(site=site,instrument=instrument,family=family,record_group=category,scored_instrument_hours=int(usable.sum()),segments_channels=len(ix)))
            # Weights divide simultaneous channels before conditioning. Bin
            # totals therefore partition physical-instrument support even if
            # distinct concurrent channels have distinct measured weather.
            weights=np.divide(base,count[None,:],out=np.zeros_like(base,dtype=float),where=count[None,:]>0)
            for condition in row_conditions[ix[0]]:
                cm=np.stack([row_conditions[i][condition] for i in ix])
                w=weights*cm; support=float(w.sum())
                if support==0:
                    continue
                metric={}
                for method,e in errors.items():
                    loss=np.where(base,e[ix],0.)
                    metric[f'{method}_rmse']=float(np.sqrt(np.sum(w*loss)/support))
                    abs_loss=np.where(base,np.sqrt(e[ix]),0.)
                    metric[f'{method}_mae']=float(np.sum(w*abs_loss)/support)
                cells.append(dict(site=site,instrument=instrument,family=family,record_group=category,condition=condition,instrument_hour_equivalents=support,unique_instrument_hours=int(np.any(base&cm,axis=0).sum()),channel_hours=int((base&cm).sum()),minimum72=support>=72,delta_rmse=metric['risk_fixed_rmse']-metric['periodic_rmse'],delta_mae=metric['risk_fixed_mae']-metric['periodic_mae'],**metric))
    frame=pd.DataFrame(cells); frame.to_csv(OUT/'instrument_conditions.csv',index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT/'initialization_environment_thresholds.csv',index=False)
    cov=pd.DataFrame(coverage_rows); cov.to_csv(OUT/'channel_environment_coverage.csv',index=False)
    meta=pd.DataFrame(instrument_meta); meta.to_csv(OUT/'instrument_metadata.csv',index=False)
    groups=[]
    for condition,g in frame[frame.minimum72].groupby('condition',sort=True):
        for variable,values in [('all',{'all':g}),('site',{k:v for k,v in g.groupby('site')}),('family',{k:v for k,v in g.groupby('family')}),('record_group',{k:v for k,v in g.groupby('record_group')})]:
            for value,h in values.items():
                groups.append(dict(condition=condition,group_variable=variable,group_value=value,instruments=len(h),instrument_hour_equivalents=float(h.instrument_hour_equivalents.sum()),periodic_rmse=float(h.periodic_rmse.mean()),risk_fixed_rmse=float(h.risk_fixed_rmse.mean()),delta_rmse=float(h.delta_rmse.mean()),instruments_improved=int((h.delta_rmse<0).sum()),minimum_delta=float(h.delta_rmse.min()),maximum_delta=float(h.delta_rmse.max())))
    aggregate=pd.DataFrame(groups);aggregate.to_csv(OUT/'condition_summary.csv',index=False)
    # Validate full physical support and partitioning of all scored hours.
    assert len(meta)==18 and meta.scored_instrument_hours.sum()==62652
    assert int(cov.channel_scored_hours.sum())==68526
    allrows=frame[frame.condition.eq('all')]
    assert np.isclose(allrows.periodic_rmse.mean(),4.582480535740785,atol=1e-7)
    assert np.isclose(allrows.risk_fixed_rmse.mean(),4.1614157,atol=1e-7)
    for variable in ('T','RH'):
        partition=frame[frame.condition.isin([f'{variable}_{x}' for x in ('missing','low','middle','high','unsupported_init')])].instrument_hour_equivalents.sum()
        assert np.isclose(partition,62652)
    assert np.isclose(frame[frame.condition.str.startswith('elapsed_')].instrument_hour_equivalents.sum(),62652)
    rec=dict(status='PASS',instruments=18,instrument_hours=62652,channel_hours=68526,full_delta=float(allrows.delta_rmse.mean()),T_finite_channel_hours=int(cov.T_observed_hours.sum()),RH_finite_channel_hours=int(cov.RH_observed_hours.sum()),both_finite_channel_hours=int(cov.both_observed_hours.sum()),thresholds_initialized_before_eval=True,all_rows_matched=True)
    (OUT/'environment_validation.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec),flush=True)
    print(aggregate[(aggregate.group_variable.eq('all'))|((aggregate.condition.eq('all')))].to_string(index=False),flush=True)

@njit(cache=True)
def trace_kernel(phi,valid,ready,ms,Ps,Fs,Qs,rs,y,initmask,query):
    nd,n,features=phi.shape;p=features+1
    pred=np.full((nd,n),np.nan)
    prior=np.full((nd,n,p),np.nan);posterior=prior.copy()
    h=np.ones(p)
    for t in range(n):
        for d in range(nd):
            if t<ready[d]:continue
            _advance(ms[d],Ps[d],Fs[d],Qs[d])
            prior[d,t]=ms[d]
            if valid[d,t]:
                h[:-1]=phi[d,t];pred[d,t]=h@ms[d]
        if initmask[t] or query[t]:
            if np.isfinite(y[t]):
                for d in range(nd):
                    if t>=ready[d] and valid[d,t]:
                        h[:-1]=phi[d,t]
                        _observe(ms[d],Ps[d],h,y[t],rs[d])
        for d in range(nd):
            if t>=ready[d]:posterior[d,t]=ms[d]
    return pred,prior,posterior

def physical_coefficients(states,model):
    out=states[:,:-1].copy()
    out[:,1:]=states[:,1:-1]/model['scale']
    out[:,0]=states[:,0]-np.sum(states[:,1:-1]*model['med']/model['scale'],axis=1)
    return out

def correlate(a,b,mask):
    good=mask&np.isfinite(a)&np.isfinite(b)
    if good.sum()<72 or np.unique(a[good]).size<2 or np.unique(b[good]).size<2:
        return np.nan,int(good.sum())
    return float(spearmanr(a[good],b[good]).statistic),int(good.sum())

def traces():
    freeze();summary=[];correlations=[];weekly=[];validation=[]
    names=('offset_ppb','main_gain','temperature_ppb_per_degC','humidity_ppb_per_percent')
    for site in SITES:
        p,f,fleet=site_inputs(site)
        packed=_prepare(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'])
        pred,prior,posterior=trace_kernel(*packed,f.query)
        clipped=np.maximum(pred,0.)
        assert np.array_equal(clipped,f.pred,equal_nan=True), f'{site} forecast parity failed'
        file=OUT/f'{site}_B48_fixed_coefficient_trajectories.npz'
        np.savez_compressed(file,prior_standardized_state=prior,posterior_standardized_state=posterior,pred=clipped,query=f.query,initmask=f.initmask,times=f.times.astype(str).to_numpy(dtype='U19'),metadata_json=json.dumps(f.metadata),feature_medians=np.stack([m['med'] for m in fleet['models']]),feature_scales=np.stack([m['scale'] for m in fleet['models']]))
        validation.append(dict(site=site,exact_forecast_parity=True,n_rows=len(f.metadata),channel_score_hours=int(f.score_masks.sum()),trajectory_sha256=sha(file)))
        for i,(row,model,X) in enumerate(zip(f.metadata,fleet['models'],fleet['Xs'])):
            beta=physical_coefficients(prior[i],model)
            post=physical_coefficients(posterior[i],model)
            sc=f.score_masks[i]
            initial=physical_coefficients(model['m'][None,:],model)[0]
            age=(np.arange(len(f.y))-fleet['ready'][i])/24.
            for j,name in enumerate(names):
                z=beta[sc,j]; q=np.quantile(z,[.05,.25,.5,.75,.95])
                summary.append(dict(site=site,instrument=row['instrument'],deployment=row['deployment'],family=row['family'],coefficient=name,scored_hours=int(sc.sum()),initial=initial[j],first_prior=z[0],last_prior=z[-1],net_change=z[-1]-z[0],median=q[2],p05=q[0],p25=q[1],p75=q[3],p95=q[4],iqr=q[3]-q[1],rms_departure_from_initial=float(np.sqrt(np.mean((z-initial[j])**2))),mean_absolute_same_hour_update=float(np.mean(np.abs(post[sc,j]-z)))))
                for variable,v in [('T',X[:,1]),('RH',X[:,2]),('elapsed_days',age)]:
                    rho,n=correlate(beta[:,j],v,sc)
                    correlations.append(dict(site=site,instrument=row['instrument'],deployment=row['deployment'],family=row['family'],coefficient=name,condition=variable,n=n,spearman=rho,interpretation='descriptive dependent time series; no causal interpretation or hourly independence inference'))
            date=pd.Series(f.times).dt.to_period('W').astype(str).to_numpy()
            for week in sorted(set(date[sc])):
                a=sc&(date==week)
                if a.sum()<24:continue
                rec=dict(site=site,instrument=row['instrument'],deployment=row['deployment'],family=row['family'],week=week,scored_hours=int(a.sum()),elapsed_days=float(np.mean(age[a])))
                for j,name in enumerate(names):rec[name]=float(np.mean(beta[a,j]))
                for j,var in [(1,'T'),(2,'RH')]:
                    observed=a&np.isfinite(X[:,j]); rec[f'{var}_observed_hours']=int(observed.sum());rec[f'{var}_mean']=float(np.mean(X[observed,j])) if observed.any() else np.nan
                weekly.append(rec)
        print('TRACE PARITY PASS',site,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'coefficient_summary.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'coefficient_environment_correlations.csv',index=False)
    pd.DataFrame(weekly).to_csv(OUT/'weekly_coefficient_environment.csv',index=False)
    (OUT/'trajectory_validation.json').write_text(json.dumps(dict(status='PASS',sites=validation),indent=2),encoding='utf-8')

def verify():
    """Second calculation path checks archived sources, support and state mapping."""
    design=freeze()
    for relative,digest in design['source_sha256'].items():
        assert sha(ROOT/relative)==digest
    frame=pd.read_csv(OUT/'instrument_conditions.csv',float_precision='round_trip')
    thresholds=pd.read_csv(OUT/'initialization_environment_thresholds.csv',float_precision='round_trip')
    trajectory_checks=[];checked_cells=0
    for site in SITES:
        p,f,fleet=site_inputs(site)
        with np.load(OUT/f'{site}_B48_fixed_coefficient_trajectories.npz',allow_pickle=False) as archive:
            prior=archive['prior_standardized_state'];post=archive['posterior_standardized_state']
            assert np.array_equal(archive['pred'],f.pred,equal_nan=True)
        for i,(row,X,model) in enumerate(zip(f.metadata,fleet['Xs'],fleet['models'])):
            sc=f.score_masks[i];beta=physical_coefficients(prior[i],model)
            guarded=X.copy();g=row['guard'];guarded[:,0]=np.clip(X[:,0],g['center']-g['span'],g['center']+g['span'])
            observed_or_median=np.where(np.isfinite(guarded),guarded,model['med'])
            rebuilt=beta[:,0]+np.sum(beta[:,1:]*observed_or_median,axis=1)+prior[i,:,-1]
            error=float(np.max(np.abs(np.maximum(rebuilt[sc],0.)-f.pred[i,sc])))
            assert error<1e-10
            no_update=(np.arange(len(p.y))>=fleet['ready'][i])&~((f.initmask|f.query)&np.isfinite(f.y)&model['valid'])
            assert np.array_equal(prior[i,no_update],post[i,no_update])
            previous=np.arange(len(p.y))>fleet['ready'][i]
            assert np.array_equal(prior[i,previous,:-1],post[i,np.where(previous)[0]-1,:-1])
            assert np.allclose(prior[i,previous,-1],post[i,np.where(previous)[0]-1,-1]*model['F'][-1],rtol=0,atol=0)
            trajectory_checks.append(dict(site=site,instrument=row['instrument'],deployment=row['deployment'],physical_affine_max_error=error,no_update_state_unchanged=True,prior_transition_exact=True))
        for instrument in sorted({r['instrument'] for r in p.metadata}):
            ix=[i for i,r in enumerate(p.metadata) if r['instrument']==instrument]
            count=p.score_masks[ix].sum(axis=0)
            h=frame[(frame.site.eq(site))&frame.instrument.eq(instrument)]
            for _,row in h.iterrows():
                summed_p=0.;summed_f=0.;support=0.
                for i in ix:
                    # Independently reconstruct memberships from the saved
                    # initialization-only thresholds instead of calling the
                    # analysis helper or borrowing its condition masks.
                    sc=p.score_masks[i];cond=row['condition'];meta=p.metadata[i]
                    if cond=='all':membership=np.ones(len(p.y),bool)
                    elif cond.startswith('elapsed_'):
                        age=(np.arange(len(p.y))-int(meta['local_offset'])-336)/24.
                        lower,upper={'elapsed_0_28d':(0,28),'elapsed_28_90d':(28,90),'elapsed_90_180d':(90,180),'elapsed_ge180d':(180,np.inf)}[cond]
                        membership=(age>=lower)&(age<upper)
                    else:
                        variable,category=cond.split('_',1);j={'T':1,'RH':2}[variable]
                        weather=fleet['Xs'][i][:,j];finite=np.isfinite(weather)
                        thr=thresholds[(thresholds.deployment.eq(meta['deployment']))&thresholds.variable.eq(variable)].iloc[0]
                        if category=='missing':membership=~finite
                        elif category=='unsupported_init':membership=finite&(not bool(thr['supported']))
                        elif category=='low':membership=finite&(weather<=thr.q25)&bool(thr['supported'])
                        elif category=='middle':membership=finite&(weather>thr.q25)&(weather<=thr.q75)&bool(thr['supported'])
                        elif category=='high':membership=finite&(weather>thr.q75)&bool(thr['supported'])
                        else:raise AssertionError(cond)
                    chosen=sc&membership;w=1./count[chosen]
                    support+=float(w.sum())
                    summed_p+=float(np.dot(w,(p.pred[i,chosen]-p.y[chosen])**2))
                    summed_f+=float(np.dot(w,(f.pred[i,chosen]-f.y[chosen])**2))
                assert np.isclose(support,row.instrument_hour_equivalents,rtol=0,atol=1e-8)
                assert np.isclose(np.sqrt(summed_p/support),row.periodic_rmse,rtol=0,atol=1e-10)
                assert np.isclose(np.sqrt(summed_f/support),row.risk_fixed_rmse,rtol=0,atol=1e-10)
                checked_cells+=1
    result=dict(status='PASS',source_files_unchanged=6,independently_recomputed_instrument_condition_cells=checked_cells,trajectory_rows=trajectory_checks,script_sha256=sha(__file__),limitations='No inferential mechanism claim; original predictions/calendars/models unchanged; environmental classes relative to initialization and incomplete, elapsed time is not electrode age.')
    (OUT/'independent_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('INDEPENDENT VALIDATION PASS',checked_cells,'cells',len(trajectory_checks),'trajectories',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['freeze','analyze','traces','verify','all'],default='all');args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.phase=='freeze':freeze()
        if args.phase in ('analyze','all'):analyze()
        if args.phase in ('traces','all'):traces()
        if args.phase in ('verify','all'):verify()

if __name__=='__main__':main()
