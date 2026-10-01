"""Frozen revision-4 computational component and instrument diagnostics.

This post-review factorial replays one already realized calendar. It identifies
computational contrasts, never physical electrode mechanisms. Only this script's
new results/revision4/components directory is written.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'cache/numba')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT/'temp')
os.environ['MPLCONFIGDIR'] = str(ROOT/'temp/matplotlib')
import argparse, hashlib, itertools, json, shutil, time
from datetime import datetime, timezone
from dataclasses import replace
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from calibration_engine import _advance, _observe
from acquisition_controller import _prepare, _fleet_fixed
from site_data import make_site
from statistical_analysis import load_run, build_site_bundle, _calendar_weights
from statistical_analysis import paired_calendar_bootstrap
from revision_diagnostics import episodes, prior_age

OUT = ROOT/'results/revision4/components'
SITES = ('Manchester', 'London', 'York')
CELLS = ('adaptive_persistent', 'adaptive_white', 'frozen_persistent', 'frozen_white')
COEFS = ('baseline_ppb', 'pollutant_gain', 'temperature_ppb_per_C', 'humidity_ppb_per_percent')
BOOT = ((7, 1000), (14, 300))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    sources = [ROOT/'evidence/FREEZE.json',
               ROOT/'scripts/calibration_engine.py',
               ROOT/'scripts/acquisition_controller.py',
               ROOT/'scripts/site_data.py',
               ROOT/'scripts/statistical_analysis.py']
    sources += [ROOT/f'results/main/{s}_B48_{p}.npz' for s in SITES
                for p in ('periodic', 'risk_fixed')]
    sources += [ROOT/f'results/revision/environment/{s}_B48_fixed_coefficient_trajectories.npz'
                for s in SITES]
    design = dict(
        status='Post-review design frozen before new component results; earlier main, white and memory results have been seen. Not preregistration or a new untouched test.',
        date='2026-10-01', budget=48, sites=list(SITES), cells=list(CELLS),
        calendar='Each site unchanged realized original B48 risk_fixed query and commissioning union. Acquisition is not reoptimized for each component.',
        common_initialization='All cells share original m/P, guard width 8, q_scale .001, ar_share .8, original rho and R. No new development or evaluation tuning.',
        adaptive='Original beta process noise and complete scalar Kalman gain.',
        frozen='Set beta process noise to zero and beta mean gains to zero. Joseph/Schmidt covariance update retains initial beta uncertainty and cross-covariance; beta means remain fixed.',
        persistent='Original rho and innovation Q_res=(1-rho^2)*stationary residual variance.',
        white='Only residual transition rho set to zero and residual Q set to original stationary residual variance. Initial residual P and beta P/R/q unchanged.',
        orthogonality='Same initial states, uncertainties, observation information and calendar in all four cells. Existing white retuning and overwrite-latest sensor-decay are separate auxiliaries, not factorial cells.',
        masks='Original common primary scoring; auxiliary off-request >=168 h after last completed merged episode end, using exactly the existing completed-episode definition and common mask in every cell.',
        endpoints='Average simultaneous-channel squared losses within physical instrument-hour, then RMSE per physical instrument; macro instrument RMSE and site equal RMSE.',
        contrasts='All six oriented pair differences (CELLS order), conditional adaptation and persistence contrasts, interaction AP-AW-FP+FW in RMSE units; no additive attribution of explained physical aging.',
        bootstrap=dict(block_days=[7,14],replicates=[1000,300],seed=20261001,
            definition='Paired circular calendar blocks inside each observed site-quarter; joint rejection of replicates with any instrument zero support; observed cohort conditional uncertainty. No independent-hour or instrument-population claims.'),
        coefficient_design='Trace all 23 original adaptive-persistent rows: physical four-coefficient prior/posterior means and marginal SD, weekly means/SD, total variation and weekly changes. Association limited to acquired usable label innovations and same-hour jumps, descriptive/partly algebraic, no p-values.',
        fragility='Original fixed versus periodic all 18 instruments, four long/fourteen short, each leave-one-physical-instrument-out, site equal; Manchester duplicate and clean flags use frozen predictions and matched masks.',
        context='London/York processed sensor product and archived reference level/range/variation, availability and guard. No traffic, maintenance, raw electrode or causal environment data invented.',
        expectations='Based on old memory results, adaptation may be especially material at long gaps. Persistence can change learning as well as carryover, so interaction may be nonzero. All directions, including worse results, retained.',
        resources=dict(cpu_threads=2,gpu=False,api=False,new_downloads=False,
            storage_target_bytes=250_000_000,D_free_bytes=shutil.disk_usage(ROOT).free,
            raw_data='Only existing D datasets are read; F/E raw unchanged.'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        implementation_sha256=sha(Path(__file__)))
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        assert old['source_sha256']==design['source_sha256'], 'Frozen sources changed'
        recorded=old.get('implementation_amendments',[])
        latest=recorded[-1]['new_sha256'] if recorded else old['implementation_sha256']
        assert latest==design['implementation_sha256'], 'Unrecorded frozen implementation change'
        return old
    design['frozen_utc']=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(design,indent=2),encoding='utf-8')
    return design


@njit(cache=True)
def frozen_observe(m, P, h, target, r):
    """Zero beta gain, full Joseph update for constrained residual-only gain."""
    ph=P@h
    s=max(r+h@ph,1e-12)
    k=ph[-1]/s
    m[-1] += k*(target-h@m)
    old=P.copy()
    for j in range(len(m)-1):
        P[-1,j]=old[-1,j]-k*ph[j]
        P[j,-1]=P[-1,j]
    P[-1,-1]=old[-1,-1]-2*k*ph[-1]+k*k*s


@njit(cache=True)
def sd_physical(P,L):
    ans=np.zeros(4)
    for i in range(4):
        v=0.
        for j in range(5):
            for k in range(5):
                v+=L[i,j]*P[j,k]*L[i,k]
        ans[i]=np.sqrt(max(v,0.))
    return ans


@njit(cache=True)
def kernel(phi,valid,ready,ms,Ps,Fs,Qs,rs,y,initmask,query,frozen,trace,L):
    nd,n,features=phi.shape
    pred=np.full((nd,n),np.nan)
    size=n if trace else 1
    prior=np.full((nd,size,5),np.nan)
    posterior=np.full((nd,size,5),np.nan)
    prior_sd=np.full((nd,size,4),np.nan)
    posterior_sd=np.full((nd,size,4),np.nan)
    innovation=np.full((nd,size),np.nan)
    acquired=np.zeros((nd,size),np.bool_)
    cov_samples=np.full((nd,(n+167)//168,5,5),np.nan)
    h=np.ones(features+1)
    for t in range(n):
        for d in range(nd):
            if t<ready[d]:
                continue
            _advance(ms[d],Ps[d],Fs[d],Qs[d])
            if trace:
                prior[d,t]=ms[d]
                prior_sd[d,t]=sd_physical(Ps[d],L[d])
            if t%168==0:
                cov_samples[d,t//168]=Ps[d]
            if valid[d,t]:
                h[:-1]=phi[d,t]
                pred[d,t]=h@ms[d]
        # Predictions at t are fixed before target at t is accessed.
        if (initmask[t] or query[t]) and np.isfinite(y[t]):
            for d in range(nd):
                if t>=ready[d] and valid[d,t]:
                    h[:-1]=phi[d,t]
                    if trace:
                        innovation[d,t]=y[t]-h@ms[d]
                        acquired[d,t]=True
                    if frozen:
                        frozen_observe(ms[d],Ps[d],h,y[t],rs[d])
                    else:
                        _observe(ms[d],Ps[d],h,y[t],rs[d])
        for d in range(nd):
            if t>=ready[d] and trace:
                posterior[d,t]=ms[d]
                posterior_sd[d,t]=sd_physical(Ps[d],L[d])
    return pred,prior,posterior,prior_sd,posterior_sd,innovation,acquired,cov_samples,ms,Ps


def transform(model):
    L=np.zeros((4,5));L[0,0]=1.
    L[0,1:4]=-model['med']/model['scale']
    L[1,1]=1/model['scale'][0]
    L[2,2]=1/model['scale'][1]
    L[3,3]=1/model['scale'][2]
    return L


def prepare(fleet,white=False,frozen=False):
    packed=list(_prepare(fleet['models'],fleet['y'],fleet['ready'],
                         fleet['initialization_mask']))
    packed[5]=packed[5].copy();packed[6]=packed[6].copy()
    if white:
        stationary=np.array([m['P'][-1,-1] for m in fleet['models']])
        np.testing.assert_allclose(packed[6][:,-1]/(1-packed[5][:,-1]**2),
                                   stationary,rtol=1e-14,atol=1e-12)
        packed[5][:,-1]=0.
        packed[6][:,-1]=stationary
    if frozen:
        packed[6][:,:-1]=0.
    return packed


def technical_test():
    rng=np.random.default_rng(20261001)
    z=rng.normal(size=(5,5));P=z@z.T+np.eye(5)
    m=rng.normal(size=5);h=rng.normal(size=5);r=.7;target=3.
    old_m=m.copy();old_P=P.copy();ph=P@h;s=r+h@ph
    K=np.zeros(5);K[-1]=ph[-1]/s
    A=np.eye(5)-np.outer(K,h)
    expected=A@P@A.T+r*np.outer(K,K)
    frozen_observe(m,P,h,target,r)
    np.testing.assert_allclose(P,expected,rtol=1e-13,atol=1e-13)
    assert np.array_equal(m[:-1],old_m[:-1])
    assert np.array_equal(P[:-1,:-1],old_P[:-1,:-1])
    assert np.linalg.eigvalsh(P).min()>0
    test=dict(status='PASS',joseph_direct_matrix_max_error=float(np.max(np.abs(P-expected))),
              beta_mean_exactly_fixed=True,beta_uncertainty_exactly_preserved=True,
              covariance_min_eigenvalue=float(np.linalg.eigvalsh(P).min()))
    (OUT/'technical_test.json').write_text(json.dumps(test,indent=2),encoding='utf-8')
    print('TECHNICAL TEST PASS',flush=True)


def replay():
    cfg=json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))['calibrator']
    checks=[];coef_rows=[];weekly=[];update_rows=[]
    for site in SITES:
        start=time.perf_counter()
        fleet=make_site(site,**cfg)
        source=ROOT/f'results/main/{site}_B48_risk_fixed.npz'
        original=load_run(source)
        assert original.times.equals(fleet['times'])
        assert original.metadata==fleet['metadata']
        assert np.array_equal(original.score_masks,fleet['score_masks'])
        L=np.stack([transform(m) for m in fleet['models']])
        original_states=None
        for cell in CELLS:
            frozen=cell.startswith('frozen')
            white=cell.endswith('white')
            packed=prepare(fleet,white=white,frozen=frozen)
            initial_ms=packed[3].copy();initial_P=packed[4].copy()
            trace=cell=='adaptive_persistent'
            z=kernel(*packed,original.query,frozen,trace,L)
            pred=np.maximum(z[0],0.)
            cov=z[7];finite=np.isfinite(cov).all(axis=(2,3))
            min_eig=float(np.linalg.eigvalsh(cov[finite]).min())
            assert min_eig>-1e-7
            if frozen:
                assert np.array_equal(z[8][:,:4],initial_ms[:,:4])
                assert np.array_equal(z[9][:,:4,:4],initial_P[:,:4,:4])
            if cell=='frozen_white':
                for d,m in enumerate(fleet['models']):
                    mask=np.arange(len(original.y))>=fleet['ready'][d]
                    mask &= m['valid']
                    expected=np.maximum(m['phi'][mask]@m['m'][:4],0.)
                    np.testing.assert_allclose(pred[d,mask],expected,rtol=1e-14,atol=1e-12)
            with np.load(source,allow_pickle=False) as a:
                fields={k:a[k] for k in a.files}
            fields['pred']=pred
            fields['component_json']=json.dumps(dict(cell=cell,frozen=frozen,white=white,
                same_calendar=True,beta_q_zero=frozen,residual_stationary_variance_unchanged=True))
            path=OUT/f'{site}_B48_{cell}.npz'
            np.savez_compressed(path,**fields)
            check=dict(site=site,cell=cell,prediction_shape=list(pred.shape),
                scored_channel_hours=int(original.score_masks.sum()),
                same_calendar=True,source_sha256=sha(source),output_sha256=sha(path),
                sampled_prior_covariance_min_eigenvalue=min_eig)
            if trace:
                assert np.array_equal(pred,original.pred,equal_nan=True)
                oldpath=ROOT/f'results/revision/environment/{site}_B48_fixed_coefficient_trajectories.npz'
                with np.load(oldpath,allow_pickle=False) as a:
                    assert np.array_equal(z[1],a['prior_standardized_state'],equal_nan=True)
                    assert np.array_equal(z[2],a['posterior_standardized_state'],equal_nan=True)
                check['prediction_and_prior_posterior_state_exact_parity']=True
                np.savez_compressed(OUT/f'{site}_coefficient_uncertainty_trajectories.npz',
                    prior_state=z[1],posterior_state=z[2],prior_physical_sd=z[3],
                    posterior_physical_sd=z[4],acquired_innovation=z[5],acquired=z[6],
                    sampled_prior_covariance=z[7],physical_transform=L,
                    times=original.times.astype(str).to_numpy(dtype='U19'),
                    metadata_json=json.dumps(original.metadata),query=original.query,
                    initmask=original.initmask)
                for d,(meta,m) in enumerate(zip(original.metadata,fleet['models'])):
                    summarize_trace(site,original,fleet,d,meta,m,L[d],z,coef_rows,weekly,update_rows)
                original_states=z[1]
            checks.append(check)
            print('REPLAY PASS',site,cell,round(time.perf_counter()-start,2),flush=True)
        # Perturb hidden targets and one acquired current target using original packed state.
        hidden=fleet['y'].copy()
        hidden[~(original.query|original.initmask)]=1234567.
        p=prepare(fleet);p[8]=hidden
        hiddenpred=kernel(*p,original.query,False,False,L)[0]
        assert np.array_equal(np.maximum(hiddenpred,0),original.pred,equal_nan=True)
        usable=np.flatnonzero(original.query&np.isfinite(original.y))
        t=int(usable[len(usable)//2]);changed=fleet['y'].copy();changed[t]+=100.
        p=prepare(fleet);p[8]=changed
        currentpred=kernel(*p,original.query,False,False,L)[0]
        assert np.array_equal(np.maximum(currentpred[:,:t+1],0),original.pred[:,:t+1],equal_nan=True)
        checks.append(dict(site=site,information_order='PASS',hidden_labels_invariant=True,
            same_hour_label_prediction_invariant=True,perturbed_acquired_hour=t))
    pd.DataFrame(coef_rows).to_csv(OUT/'coefficient_variation_uncertainty.csv',index=False)
    pd.DataFrame(weekly).to_csv(OUT/'weekly_coefficients_uncertainty.csv',index=False)
    pd.DataFrame(update_rows).to_csv(OUT/'coefficient_acquired_update_associations.csv',index=False)
    (OUT/'replay_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


def summarize_trace(site,run,fleet,d,meta,model,L,z,rows,weekly,updates):
    mean=z[1][d]@L.T;post=z[2][d]@L.T
    score=run.score_masks[d]
    ticks=np.arange(len(run.y))
    ready=fleet['ready'][d]
    end=run.times.get_indexer([pd.Timestamp(meta['end'])])[0]
    span=(ticks>=ready)&(ticks<=end)
    access=z[6][d]&span
    weeks=run.times.to_period('W-SUN')
    base=dict(site=site,instrument=meta['instrument'],deployment=meta['deployment'],family=meta['family'],
        trajectory_hours=int(span.sum()),scored_hours=int(score.sum()),acquired_usable_pairs=int(access.sum()))
    for j,name in enumerate(COEFS):
        a=mean[span,j];deltas=np.diff(a)
        initial=float(L[j]@model['m'])
        s=z[3][d,span,j]
        q=np.quantile(a,[.05,.5,.95])
        net=a[-1]-a[0];total=np.abs(deltas).sum()
        row=base|dict(coefficient=name,units=('ppb','ppb/ppb','ppb/degC','ppb/%RH')[j],
            initialization_value=initial,first_prior=a[0],last_prior=a[-1],net_change=net,
            total_variation=total,net_over_total_variation=abs(net)/total if total else 0.,
            hourly_change_abs_p95=np.quantile(np.abs(deltas),.95),
            hourly_change_abs_max=np.max(np.abs(deltas)),p05=q[0],median=q[1],p95=q[2],
            rms_departure_from_initial=float(np.sqrt(np.mean((a-initial)**2))),
            prior_sd_initial=float(np.sqrt(L[j]@model['P']@L[j])),
            prior_sd_first=s[0],prior_sd_last=s[-1],prior_sd_median=np.median(s),
            prior_sd_p95=np.quantile(s,.95),
            mean_acquired_sd_contraction=float(np.mean(z[3][d,access,j]-z[4][d,access,j])) if access.any() else np.nan)
        wkmeans=[]
        for w in weeks[span].unique():
            mask=span&np.asarray(weeks==w)
            if mask.sum()<24:continue
            qv=mean[mask,j]
            weekly.append(base|dict(coefficient=name,week=str(w),included_hours=int(mask.sum()),
                scored_week_hours=int((mask&score).sum()),acquired_week_pairs=int((mask&access).sum()),
                elapsed_days=float(np.mean((ticks[mask]-ready)/24)),mean=float(qv.mean()),
                minimum=float(qv.min()),maximum=float(qv.max()),
                mean_prior_sd=float(z[3][d,mask,j].mean()),
                mean_posterior_sd=float(z[4][d,mask,j].mean())))
            wkmeans.append(qv.mean())
        change=np.abs(np.diff(wkmeans))
        row['weekly_mean_change_abs_max']=float(change.max()) if len(change) else np.nan
        row['weekly_mean_change_abs_median']=float(np.median(change)) if len(change) else np.nan
        rows.append(row)
        jump=post[access,j]-mean[access,j]
        innovation=z[5][d,access]
        good=len(jump)>=24 and np.unique(jump).size>1 and np.unique(innovation).size>1
        corr=float(spearmanr(jump,innovation).statistic) if good else np.nan
        updates.append(base|dict(coefficient=name,signed_jump_innovation_spearman=corr,
            jump_abs_mean=float(np.abs(jump).mean()) if len(jump) else np.nan,
            jump_abs_p95=float(np.quantile(np.abs(jump),.95)) if len(jump) else np.nan,
            innovation_mean=float(np.mean(innovation)) if len(jump) else np.nan,
            innovation_rms=float(np.sqrt(np.mean(innovation**2))) if len(jump) else np.nan,
            interpretation='Only acquired pairs; same-hour Kalman jump algebraically depends on innovation. Dependent-series description, not aging evidence or independent-sample test.'))


def subset_bundle(b,keep):
    keep=np.asarray(keep,bool)
    out=b.copy()
    for name in ('active','channel_count'):out[name]=b[name][keep]
    for name in ('se','ae','signed'):out[name]=b[name][:,keep]
    for name in ('instruments','families'):out[name]=tuple(np.asarray(b[name])[keep])
    return out


def bootstrap_instruments(bundles,reps,days,seed):
    """Same paired quarter/block algorithm, retaining per-instrument RMSE draws."""
    methods=bundles[0]['methods']
    ids=tuple(sorted({i for b in bundles for i in b['instruments']}))
    index={i:k for k,i in enumerate(ids)}
    observed_num=np.zeros((len(methods),len(ids)));observed_n=np.zeros(len(ids))
    observed_site=[]
    for b in bundles:
        n=b['active'].sum(axis=1)
        assert np.all(n>0)
        num=b['se'].sum(axis=2)
        observed_site.append(np.sqrt(num/n).mean(axis=1))
        for j,ins in enumerate(b['instruments']):
            observed_num[:,index[ins]]+=num[:,j];observed_n[index[ins]]+=n[j]
    rng=np.random.default_rng(seed);accepted=attempted=rejected=0;allins=[];allsites=[]
    with threadpool_limits(limits=2):
        while accepted<reps and attempted<20*reps:
            count=min(64,reps-accepted)
            num=np.zeros((count,len(methods),len(ids)));n=np.zeros((count,len(ids)))
            valid=np.ones(count,bool);sitevalues=[]
            for b in bundles:
                w=_calendar_weights(b['times'],count,days*24,rng).astype(float)
                nn=w@b['active'].T.astype(float)
                numer=(w@b['se'].reshape(-1,len(b['times'])).T).reshape(count,len(methods),-1)
                valid&=np.all(nn>0,axis=1)
                sitevalues.append(np.sqrt(numer/np.maximum(nn[:,None,:],1)).mean(axis=2))
                for j,ins in enumerate(b['instruments']):
                    num[:,:,index[ins]]+=numer[:,:,j];n[:,index[ins]]+=nn[:,j]
            valid&=np.all(n>0,axis=1)
            allins.append(np.sqrt(num[valid]/n[valid,None,:]))
            allsites.append(np.stack(sitevalues).mean(axis=0)[valid])
            attempted+=count;rejected+=int((~valid).sum());accepted+=int(valid.sum())
    assert accepted==reps
    return dict(methods=methods,instruments=ids,
        instrument=np.concatenate(allins),site=np.concatenate(allsites),
        observed_instrument=np.sqrt(observed_num/observed_n),
        observed_site=np.stack(observed_site).mean(axis=0),
        diagnostics=dict(block_days=days,accepted_replicates=accepted,attempted_replicates=attempted,
            rejected_replicates=rejected,seed=seed,
            uncertainty_scope='Conditional observed cohort paired within-site-quarter calendar blocks'))


def bundle_tables(bundles,scope,insrows,aggrows):
    nums={};counts={};families={};sites=[]
    for b in bundles:
        n=b['active'].sum(axis=1);mse=b['se'].sum(axis=2)
        maes=b['ae'].sum(axis=2);bias=b['signed'].sum(axis=2)
        valid=n>0
        sites.append(np.sqrt(mse[:,valid]/n[valid]).mean(axis=1))
        for j,ins in enumerate(b['instruments']):
            if not n[j]:continue
            assert ins not in nums
            nums[ins]=mse[:,j];counts[ins]=n[j];families[ins]=b['families'][j]
            for k,method in enumerate(b['methods']):
                insrows.append(dict(scope=scope,site=b['site'],instrument=ins,family=b['families'][j],
                    record_group='long_Zep' if b['families'][j]=='Zep' else 'short_multi_product',
                    method=method,instrument_hours=int(n[j]),rmse=float(np.sqrt(mse[k,j]/n[j])),
                    mae=float(maes[k,j]/n[j]),bias=float(bias[k,j]/n[j])))
    methods=bundles[0]['methods']
    for k,method in enumerate(methods):
        aggrows.append(dict(scope=scope,method=method,n_instruments=len(nums),n_sites=len(bundles),
            instrument_hours=sum(int(n) for n in counts.values()),
            macro_instrument_rmse=float(np.mean([np.sqrt(v[k]/counts[i]) for i,v in nums.items()])),
            site_equal_rmse=float(np.mean([a[k] for a in sites]))))


def ci_row(scope,name,observed,samples,boot,**extra):
    ci=np.quantile(samples,[.025,.975])
    return dict(scope=scope,contrast=name,estimate=float(observed),ci_low=float(ci[0]),ci_high=float(ci[1]),
                units='ppb',**boot['diagnostics'],**extra)


def component_analysis():
    insrows=[];aggrows=[];cis=[];bootdiag=[];support=[]
    allbundles=[];gapbundles=[]
    for site in SITES:
        runs={c:load_run(OUT/f'{site}_B48_{c}.npz') for c in CELLS}
        base=runs[CELLS[0]]
        assert all(np.array_equal(r.query,base.query) for r in runs.values())
        allbundles.append(build_site_bundle(runs,methods=CELLS))
        access=base.query|base.initmask
        _,ends=episodes(access);last=np.zeros(len(access),bool);last[ends[ends<len(access)]-1]=True
        age=prior_age(last);gap=(age>=168)&np.isfinite(age)&~access
        restricted={c:replace(r,score_masks=r.score_masks&gap) for c,r in runs.items()}
        b=build_site_bundle(restricted,methods=CELLS)
        gapbundles.append(subset_bundle(b,b['active'].any(axis=1)))
    for basis,bundles in [('all_scored',allbundles),('gap_ge168h',gapbundles)]:
        for group in ('all','long_Zep','short_multi_product'):
            bs=[]
            for b in bundles:
                keep=np.array([group=='all' or ((f=='Zep')==(group=='long_Zep')) for f in b['families']])
                if keep.any():bs.append(subset_bundle(b,keep))
            scope=basis+'__'+group
            support.append(dict(scope=scope,n_instruments=sum(len(b['instruments']) for b in bs),
                n_sites=len(bs),instrument_hours=sum(int(b['active'].sum()) for b in bs)))
            bundle_tables(bs,scope,insrows,aggrows)
            for days,reps in BOOT:
                boot=bootstrap_instruments(bs,reps,days,20261001)
                bootdiag.append(dict(scope=scope,**boot['diagnostics']))
                draws=boot['instrument'].mean(axis=2);obs=boot['observed_instrument'].mean(axis=1)
                for metric,v,z in [('macro_instrument_rmse',draws,obs),
                                   ('site_equal_rmse',boot['site'],boot['observed_site'])]:
                    for a,c in itertools.combinations(range(4),2):
                        cis.append(ci_row(scope,CELLS[a]+' minus '+CELLS[c],z[a]-z[c],v[:,a]-v[:,c],
                                          boot,metric=metric))
                    interaction=z[0]-z[1]-z[2]+z[3]
                    samples=v[:,0]-v[:,1]-v[:,2]+v[:,3]
                    cis.append(ci_row(scope,'AP-AW-FP+FW interaction',interaction,samples,boot,metric=metric))
                np.savez_compressed(OUT/f'bootstrap_components_{scope}_{days}d.npz',
                    instrument_rmse=boot['instrument'],site_equal_rmse=boot['site'],
                    observed_instrument_rmse=boot['observed_instrument'],instruments=np.array(boot['instruments']),
                    methods=np.array(boot['methods']),diagnostics_json=json.dumps(boot['diagnostics']))
            print('COMPONENT METRICS/CI',scope,flush=True)
    pd.DataFrame(insrows).to_csv(OUT/'component_instrument_metrics.csv',index=False)
    pd.DataFrame(aggrows).to_csv(OUT/'component_aggregate_metrics.csv',index=False)
    pd.DataFrame(cis).to_csv(OUT/'component_paired_intervals.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'component_support.csv',index=False)
    gap=pd.DataFrame(support).set_index('scope')
    assert gap.loc['gap_ge168h__long_Zep','instrument_hours']==9745
    assert gap.loc['gap_ge168h__short_multi_product','instrument_hours']==8495
    (OUT/'component_bootstrap_diagnostics.json').write_text(json.dumps(bootdiag,indent=2),encoding='utf-8')
    # Previously verified overwrite/decay and reference-only comparisons remain auxiliary.
    old=pd.read_csv(ROOT/'results/revision/diagnostics/supplement_aggregate_metrics.csv')
    old=old[old.analysis_group.astype(str).str.contains('memory')].copy()
    old['provenance']='Existing verified overwrite-latest residual and reference-only replay; not factorial cells.'
    old.to_csv(OUT/'existing_memory_auxiliary_metrics.csv',index=False)


def fragility():
    insrows=[];aggrows=[];ci=[];loio=[];bymask={}
    methods=('periodic','risk_fixed')
    for reference in ('primary','clean_flags','duplicate'):
        bundles=[]
        for site in SITES:
            runs={m:load_run(ROOT/f'results/main/{site}_B48_{m}.npz') for m in methods}
            if reference=='duplicate' and site!='Manchester':continue
            b=build_site_bundle(runs,methods=methods,reference=reference)
            if b is None:continue
            b=subset_bundle(b,b['active'].any(axis=1))
            bundles.append(b)
        if not bundles:continue
        bymask[reference]=bundles
        bundle_tables(bundles,reference,insrows,aggrows)
        for days,reps in BOOT:
            boot=bootstrap_instruments(bundles,reps,days,20261001)
            families={i:f for b in bundles for i,f in zip(b['instruments'],b['families'])}
            obs=boot['observed_instrument'];z=boot['instrument']
            groups=['all','long_Zep','short_multi_product'] if reference=='primary' else ['all']
            for group in groups:
                keep=np.array([group=='all' or ((families[i]=='Zep')==(group=='long_Zep'))
                               for i in boot['instruments']])
                estimate=obs[:,keep].mean(axis=1);samples=z[:,:,keep].mean(axis=2)
                ci.append(ci_row(reference+'__'+group,'risk_fixed minus periodic',
                    estimate[1]-estimate[0],samples[:,1]-samples[:,0],boot,metric='macro_instrument_rmse',
                    n_instruments=int(keep.sum())))
            ci.append(ci_row(reference+'__all','risk_fixed minus periodic',
                boot['observed_site'][1]-boot['observed_site'][0],boot['site'][:,1]-boot['site'][:,0],
                boot,metric='site_equal_rmse',n_instruments=len(boot['instruments'])))
            if reference=='primary':
                for j,ins in enumerate(boot['instruments']):
                    keep=np.arange(len(boot['instruments']))!=j
                    estimate=obs[:,keep].mean(axis=1);samples=z[:,:,keep].mean(axis=2)
                    loio.append(ci_row('primary_leave_one_out','risk_fixed minus periodic',
                        estimate[1]-estimate[0],samples[:,1]-samples[:,0],boot,
                        omitted_instrument=ins,omitted_family=families[ins],
                        periodic_rmse=estimate[0],risk_fixed_rmse=estimate[1],n_instruments=int(keep.sum()),
                        note='Observed-cohort leave-one-instrument mean, no model refitting or independent cohort validation.'))
            np.savez_compressed(OUT/f'bootstrap_fragility_{reference}_{days}d.npz',
                instrument_rmse=boot['instrument'],site_equal_rmse=boot['site'],
                instruments=np.array(boot['instruments']),methods=np.array(methods),
                observed_instrument_rmse=boot['observed_instrument'],diagnostics_json=json.dumps(boot['diagnostics']))
        print('FRAGILITY',reference,flush=True)
    pd.DataFrame(insrows).to_csv(OUT/'fixed_periodic_instrument_metrics.csv',index=False)
    pd.DataFrame(aggrows).to_csv(OUT/'fixed_periodic_aggregate_metrics.csv',index=False)
    pd.DataFrame(ci).to_csv(OUT/'fragility_paired_intervals.csv',index=False)
    pd.DataFrame(loio).to_csv(OUT/'leave_one_instrument_out.csv',index=False)
    # Duplicate versus primary endpoint comparison on exactly QC-valid duplicate support.
    duppairs=[]
    runs={m:load_run(ROOT/f'results/main/Manchester_B48_{m}.npz') for m in methods}
    dup=next(r for r in runs.values() if r.duplicate_valid is not None)
    restricted={m:replace(r,score_masks=r.score_masks&dup.duplicate_valid[None,:])
                for m,r in runs.items()}
    for ref in ('primary','duplicate'):
        b=build_site_bundle(restricted,methods=methods,reference=ref)
        bundle_tables([b],'Manchester_duplicate_matched_'+ref,duppairs,[])
    pd.DataFrame(duppairs).to_csv(OUT/'primary_duplicate_identical_support.csv',index=False)
    available=pd.read_csv(ROOT/'results/revision/diagnostics/availability.csv')
    available=available[available.domain.eq('post_initialization')].copy()
    available['provenance']='Existing verified archived D-data availability analysis; not maintenance records.'
    available.to_csv(OUT/'instrument_available_time.csv',index=False)


def context():
    cfg=json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))['calibrator']
    rows=[]
    for site in SITES:
        fleet=make_site(site,**cfg);run=load_run(ROOT/f'results/main/{site}_B48_risk_fixed.npz')
        ticks=np.arange(len(run.y))
        for d,(meta,m,X) in enumerate(zip(run.metadata,fleet['models'],fleet['Xs'])):
            ready=fleet['ready'][d];end=run.times.get_indexer([pd.Timestamp(meta['end'])])[0]
            span=(ticks>=ready)&(ticks<=end)
            scored=run.score_masks[d]
            row=dict(site=site,instrument=meta['instrument'],deployment=meta['deployment'],family=meta['family'],
                     post_init_active_hours=int(span.sum()),paired_scored_hours=int(scored.sum()),
                     reference_missing_hours=int((span&~np.isfinite(run.y)).sum()),
                     sensor_product_missing_hours=int((span&~np.isfinite(X[:,0])).sum()),
                     weather_T_missing_hours=int((span&~np.isfinite(X[:,1])).sum()),
                     weather_RH_missing_hours=int((span&~np.isfinite(X[:,2])).sum()),
                     guarded_scored_hours=int(meta['guarded_eval_hours']),rho=float(meta['rho']))
            for name,a in [('reference',run.y),('processed_sensor_product',X[:,0])]:
                use=scored&np.isfinite(a);z=a[use]
                q=np.quantile(z,[.05,.25,.5,.75,.95])
                adj=use[1:]&use[:-1];change=np.abs(np.diff(a)[adj])
                row.update({name+'_mean':float(np.mean(z)),name+'_sd':float(np.std(z)),
                    name+'_minimum':float(z.min()),name+'_maximum':float(z.max()),
                    name+'_p05':q[0],name+'_p25':q[1],name+'_median':q[2],name+'_p75':q[3],name+'_p95':q[4],
                    name+'_consecutive_pairs':int(adj.sum()),
                    name+'_abs_hourly_change_median':float(np.median(change)) if len(change) else np.nan,
                    name+'_abs_hourly_change_p95':float(np.quantile(change,.95)) if len(change) else np.nan})
            diff=X[scored,0]-run.y[scored]
            row.update(raw_product_minus_reference_bias=float(diff.mean()),
                raw_product_minus_reference_rmse=float(np.sqrt(np.mean(diff**2))),
                processed_product_reference_pearson=float(np.corrcoef(X[scored,0],run.y[scored])[0,1]))
            init=transform(m)@m['m']
            row.update({f'initial_{c}':float(init[k]) for k,c in enumerate(COEFS)})
            # Disclose whether physical coefficients had observed input variation.
            begin=int(meta['local_offset'])
            for k,name in enumerate(('pollutant','temperature','humidity')):
                initial=X[begin:ready,k]
                initial=initial[np.isfinite(initial)]
                evaluation=X[span,k]
                evaluation=evaluation[np.isfinite(evaluation)]
                for label,values in [('initialization',initial),('post_initialization',evaluation)]:
                    row[f'{name}_{label}_observed_hours']=int(len(values))
                    row[f'{name}_{label}_distinct_values']=int(np.unique(values).size)
                    row[f'{name}_{label}_sd']=float(np.std(values)) if len(values) else np.nan
                    row[f'{name}_{label}_range']=float(np.ptp(values)) if len(values) else np.nan
                row[f'{name}_no_initialization_variation']=np.unique(initial).size<2
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'reference_and_processed_product_context.csv',index=False)


def family_descriptive():
    """Prespecified physical family descriptions; no small-group inference."""
    frame=pd.read_csv(OUT/'fixed_periodic_instrument_metrics.csv')
    frame=frame[frame.scope.eq('primary')]
    rows=[]
    for (family,method),g in frame.groupby(['family','method']):
        rows.append(dict(family=family,method=method,n_instruments=g.instrument.nunique(),
            instrument_hours=int(g.instrument_hours.sum()),macro_instrument_rmse=float(g.rmse.mean()),
            scope='Observed product and record-duration/site are confounded; descriptive, no population inference.'))
    pd.DataFrame(rows).to_csv(OUT/'family_descriptive_metrics.csv',index=False)


def verify():
    design=freeze()
    for path,digest in design['source_sha256'].items():assert sha(ROOT/path)==digest
    checks=json.loads((OUT/'replay_checks.json').read_text(encoding='utf-8'))
    assert len([r for r in checks if 'cell' in r])==12
    coef=pd.read_csv(OUT/'coefficient_variation_uncertainty.csv')
    assert len(coef)==92
    assert coef.deployment.nunique()==23
    assert np.isfinite(coef[['prior_sd_median','total_variation']]).all().all()
    metrics=pd.read_csv(OUT/'component_instrument_metrics.csv')
    aggregate=pd.read_csv(OUT/'component_aggregate_metrics.csv')
    validated=0
    for _,row in metrics.iterrows():
        run=load_run(OUT/f'{row.site}_B48_{row.method}.npz')
        ids=[i for i,m in enumerate(run.metadata) if m['instrument']==row.instrument]
        mask=run.score_masks[ids].copy()
        if row.scope.startswith('gap'):
            q=run.query|run.initmask;_,ends=episodes(q)
            last=np.zeros(len(q),bool);last[ends[ends<len(q)]-1]=True
            age=prior_age(last)
            mask&=((age>=168)&np.isfinite(age)&~q)[None,:]
        count=mask.sum(axis=0);active=count>0
        err=np.where(mask,run.pred[ids]-run.y,0.)
        se=(err**2).sum(axis=0)/np.maximum(count,1)
        rmse=np.sqrt(se.sum()/active.sum())
        assert active.sum()==row.instrument_hours
        np.testing.assert_allclose(rmse,row.rmse,rtol=1e-13,atol=1e-13)
        validated+=1
    for _,row in aggregate.iterrows():
        g=metrics[metrics.scope.eq(row.scope)&metrics.method.eq(row.method)]
        np.testing.assert_allclose(g.rmse.mean(),row.macro_instrument_rmse,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(g.groupby('site').rmse.mean().mean(),row.site_equal_rmse,rtol=1e-13,atol=1e-13)
    # Independent physical transform, full-covariance marginal SD and same-hour checks.
    coefficient_cells=0
    cfg=json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))['calibrator']
    for site in SITES:
        fleet=make_site(site,**cfg)
        with np.load(OUT/f'{site}_coefficient_uncertainty_trajectories.npz',allow_pickle=False) as a:
            for d,m in enumerate(fleet['models']):
                L=transform(m)
                covariance=a['sampled_prior_covariance'][d]
                idx=np.arange(len(covariance))*168
                valid=np.isfinite(covariance).all(axis=(1,2))
                expected=np.sqrt(np.maximum(np.diagonal(L@covariance[valid]@L.T,axis1=1,axis2=2),0))
                np.testing.assert_allclose(expected,a['prior_physical_sd'][d,idx[valid]],rtol=1e-10,atol=1e-10)
                no_update=np.isfinite(a['prior_state'][d,:,0])&~a['acquired'][d]
                assert np.array_equal(a['prior_state'][d,no_update],a['posterior_state'][d,no_update])
                before=np.array(a['prior_state'][d]@L.T)
                sel=np.isfinite(before[:,0]);physical=before[sel]
                source=coef[coef.site.eq(site)&coef.deployment.eq(fleet['metadata'][d]['deployment'])]
                for j,c in enumerate(COEFS):
                    row=source[source.coefficient.eq(c)].iloc[0]
                    # Trajectory span ends at metadata end, so avoid post-end predictions.
                    end=fleet['times'].get_indexer([pd.Timestamp(fleet['metadata'][d]['end'])])[0]
                    use=sel&(np.arange(len(sel))<=end)
                    vals=before[use,j]
                    np.testing.assert_allclose(np.abs(np.diff(vals)).sum(),row.total_variation,rtol=1e-12,atol=1e-12)
                    coefficient_cells+=1
    primary=pd.read_csv(OUT/'fixed_periodic_instrument_metrics.csv')
    g=primary[primary.scope.eq('primary')]
    assert g.instrument.nunique()==18 and len(g)==36
    pivot=g.pivot(index='instrument',columns='method',values='rmse')
    assert ((pivot.risk_fixed-pivot.periodic)<0).sum()==12
    loio=pd.read_csv(OUT/'leave_one_instrument_out.csv')
    for _,r in loio.iterrows():
        d=(pivot.risk_fixed-pivot.periodic).drop(r.omitted_instrument)
        np.testing.assert_allclose(d.mean(),r.estimate,rtol=1e-13,atol=1e-13)
    ci=pd.read_csv(OUT/'component_paired_intervals.csv')
    for _,r in ci.iterrows():
        with np.load(OUT/f'bootstrap_components_{r.scope}_{int(r.block_days)}d.npz',allow_pickle=False) as a:
            if r.metric=='macro_instrument_rmse':
                draws=a['instrument_rmse'].mean(axis=2)
                point=a['observed_instrument_rmse'].mean(axis=1)
            else:
                draws=a['site_equal_rmse']
                # Point estimate recomputed from the verified saved aggregate table.
                point=np.array([aggregate[(aggregate.scope.eq(r.scope))&
                    aggregate.method.eq(c)].site_equal_rmse.iloc[0] for c in CELLS])
        if r.contrast=='AP-AW-FP+FW interaction':
            weights=np.array([1,-1,-1,1])
            delta=draws@weights;estimate=point@weights
        else:
            left,right=r.contrast.split(' minus ')
            i,j=CELLS.index(left),CELLS.index(right)
            delta=draws[:,i]-draws[:,j];estimate=point[i]-point[j]
        np.testing.assert_allclose(estimate,r.estimate,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(np.quantile(delta,[.025,.975]),[r.ci_low,r.ci_high],rtol=1e-10,atol=1e-12)
    # Existing independently reviewed bootstrap implementation must agree exactly.
    bs=[build_site_bundle({c:load_run(OUT/f'{s}_B48_{c}.npz') for c in CELLS},methods=CELLS)
        for s in SITES]
    for days,reps in BOOT:
        independent=paired_calendar_bootstrap(bs,n_replicates=reps,block_days=days,seed=20261001)
        with np.load(OUT/f'bootstrap_components_all_scored__all_{days}d.npz',allow_pickle=False) as a:
            assert np.array_equal(independent['primary'],a['instrument_rmse'].mean(axis=2))
            assert np.array_equal(independent['secondary'],a['site_equal_rmse'])
    record=dict(status='PASS',component_instrument_metric_rows_verified=validated,
        component_aggregate_rows_verified=len(aggregate),coefficient_variation_sd_cells_verified=coefficient_cells,
        original_sources_unchanged=True,all_twelve_replays_validated=True,
        exact_original_prediction_prior_posterior_parity=True,
        constrained_covariance_Joseph_test=True,PSD_weekly_samples_checked=True,
        component_paired_interval_rows_verified=len(ci),
        bootstrap_matches_existing_independently_reviewed_implementation_exactly=True,
        coefficient_uncertainty='Kalman model marginal SD, not empirically calibrated coverage or physical aging uncertainty.',
        manuscript_claim_boundary='Observed cohort computational ablation; fixed realized schedule; no physical causal component attribution.',
        verified_utc=datetime.now(timezone.utc).isoformat())
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='verification.json']
    record['output_sha256']={p.name:sha(p) for p in files}
    (OUT/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('VERIFY PASS',json.dumps({k:v for k,v in record.items() if k!='output_sha256'}),flush=True)


def main(stage):
    freeze()
    with threadpool_limits(limits=2):
        if stage in ('test','all'):technical_test()
        if stage in ('replay','all'):replay()
        if stage in ('analysis','all'):
            component_analysis();fragility();context();family_descriptive()
        if stage in ('verify','all'):verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['freeze','test','replay','analysis','verify','all'],default='all')
    args=parser.parse_args()
    main(args.stage)
