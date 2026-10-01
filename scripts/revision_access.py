"""Post-review causal arrival/setup, residual and memory experiments.

Original experiments remain immutable. New analyses are supplemental.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'cache/numba'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,copy,json,hashlib,time
import numpy as np
import pandas as pd
from calibration_engine import initialize,guard_features,_advance,_observe
from site_data import make_site
from fixed_schedules import fixed_site_mask
import acquisition_controller as original
import revision_access_controller as revised

OUT=ROOT/'results/revision/access_v2'
CONFIG=json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))
POLICIES=('periodic','risk_fixed','risk')


def macro(pred,y,mask,meta):
    values=[]
    for ident in sorted({r['instrument'] for r in meta}):
        rows=[i for i,r in enumerate(meta) if r['instrument']==ident]
        count=mask[rows].sum(axis=0);ok=count>0
        error=np.where(mask[rows],pred[rows]-y,0.)
        loss=(error*error).sum(axis=0)/np.maximum(count,1)
        values.append(np.sqrt(loss[ok].mean()))
    return float(np.mean(values))


def load_source(site,policy):
    p=ROOT/'results/main'/f'{site}_B48_{policy}.npz'
    with np.load(p,allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    return source,p


def save(fleet,out,source,source_path,method,score_mask=None,extra=None):
    score=fleet['score_masks'] if score_mask is None else score_mask
    pred=np.maximum(out['pred'],0.)
    assert np.isfinite(pred[score]).all(),method
    keys=('y','times','metadata_json','warning_mask','quality_flag_mask','y_duplicate','duplicate_valid')
    pack={k:source[k] for k in keys if k in source}
    pack.update(pred=pred,score_masks=score,query=np.asarray(out['query'],bool),initmask=fleet['initialization_mask'])
    for k in ('var','start_mask','forced_query','chosen_lengths','candidate_scores','block_spent','block_target','budget_shortfall'):
        if k in out:pack[k]=out[k]
    provenance=dict(stage='post-review supplemental',method=method,source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),controller_sha256=hashlib.sha256((ROOT/'scripts/revision_access_controller.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'evidence/REVISION_ACCESS_PROTOCOL.json').read_bytes()).hexdigest(),extra=extra or {})
    finite_requested=int(((pack['query']|pack['initmask'])&np.isfinite(pack['y'])).sum())
    access=dict(initialization_attempts=int(fleet['initialization_mask'].sum()),additional_attempts=int(pack['query'].sum()),delivered_references=int(np.asarray(out['success']).sum()) if 'success' in out else finite_requested,finite_requested_references=finite_requested,runtime_seconds=float(out.get('runtime_seconds',0.)),runtime_measured='success' in out)
    pack['access_json']=json.dumps(access);pack['revision_provenance_json']=json.dumps(provenance,sort_keys=True)
    p=OUT/f"{fleet['site']}_B48_{method}.npz"
    if p.exists():
        with np.load(p,allow_pickle=False) as prior:
            assert np.array_equal(prior['pred'],pred,equal_nan=True),p
            assert json.loads(prior['revision_provenance_json'].item())==provenance,p
    else:np.savez_compressed(p,**pack)
    result=dict(site=fleet['site'],method=method,rmse=macro(pred,fleet['y'],score,fleet['metadata']),score_hours=int(score.any(axis=0).sum()),score_channel_hours=int(score.sum()),**access)
    print(json.dumps(result),flush=True)
    return result


def run_policy(fleet,policy,source,delay=0,shared_setup=True):
    if policy=='periodic':
        return revised.run_site_fixed(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'],source['query'],delay=delay,shared_setup=shared_setup)
    return revised.run_site(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'],budget=48,policy=policy,delay=delay,shared_setup=shared_setup,**CONFIG['policies'][policy])


def self_test():
    rng=np.random.default_rng(20261001);n=430;ready=96
    X=rng.normal(size=(n,3));X[:,0]+=20
    y=1.5*X[:,0]+rng.normal(size=n)
    m=initialize(X,y,ready,q_scale=.001)
    init=np.arange(n)<ready
    q=np.zeros(n,bool);q[100:112]=True;q[220:232]=True;q[310:322]=True
    for delay in (0,24,72):
        base=revised.run_site_fixed([m],y,[ready],init,q,delay=delay)
        if delay==0:
            old=original.run_site_fixed([m],y,[ready],init,q)
            np.testing.assert_allclose(base['pred'],old['pred'],equal_nan=True,rtol=0,atol=0)
        at=100;changed_y=y.copy();changed_y[at]+=100
        changed=revised.run_site_fixed([m],changed_y,[ready],init,q,delay=delay)
        np.testing.assert_allclose(changed['pred'][:,:at+delay+1],base['pred'][:,:at+delay+1],equal_nan=True,rtol=0,atol=0)
        assert np.nanmax(abs(changed['pred'][:,at+delay+1:]-base['pred'][:,at+delay+1:]))>.01
        # Independent brute-force replay conditional on labels received before t.
        for t in (ready+delay,ready+delay+17,250+delay):
            if t>=n:continue
            mean,cov=m['m'].copy(),m['P'].copy()
            for u in range(ready,t+1):
                _advance(mean,cov,m['F'],m['Q'])
                h=np.r_[m['phi'][u],1.]
                if u==t:expected=h@mean
                if u<t-delay and (q[u] or init[u]) and np.isfinite(y[u]):
                    _observe(mean,cov,h,y[u],m['r'])
            np.testing.assert_allclose(base['pred'][0,t],expected,rtol=1e-10,atol=1e-10)
    settings=dict(budget=12,block=168,horizon=48,policy='risk_fixed',fixed_length=12)
    old=original.run_site([m],y,[ready],init,**settings)
    new=revised.run_site([m],y,[ready],init,**settings)
    np.testing.assert_array_equal(old['query'],new['query'])
    np.testing.assert_allclose(old['pred'],new['pred'],equal_nan=True,rtol=0,atol=0)
    lagged=revised.run_site([m],y,[ready],init,delay=24,**settings)
    changed_y=y.copy();changed_y[~(init|lagged['query'])]=1e8
    changed=revised.run_site([m],changed_y,[ready],init,delay=24,**settings)
    np.testing.assert_array_equal(lagged['query'],changed['query'])
    np.testing.assert_allclose(lagged['pred'],changed['pred'],equal_nan=True,rtol=0,atol=0)
    second=copy.deepcopy(m);second['init_end']=140
    staggered=revised.run_site([m,second],y,[ready,140],init,delay=72,**settings)
    hidden_model=copy.deepcopy(second);hidden_model['P']*=1e6;hidden_model['m']+=1e6
    perturb=revised.run_site([m,hidden_model],y,[ready,140],init,delay=72,**settings)
    np.testing.assert_allclose(staggered['candidate_scores'][:212],perturb['candidate_scores'][:212],equal_nan=True,rtol=0,atol=0)
    np.testing.assert_array_equal(staggered['query'][:212],perturb['query'][:212])
    free=init.copy();free[260:270]=True
    own=revised.run_site_fixed([m],y,[ready],free,q,shared_setup=False)
    yy=y.copy();yy[260:270]+=1e6
    altered=revised.run_site_fixed([m],yy,[ready],free,q,shared_setup=False)
    np.testing.assert_allclose(own['pred'],altered['pred'],equal_nan=True,rtol=0,atol=0)
    result=dict(zero_delay_fixed_exact=True,zero_delay_adaptive_exact=True,arrival_boundary=True,independent_bruteforce=True,hidden_targets_do_not_change_policy=True,uncommissioned_covariance_cannot_influence_actions=True,unshared_later_setup_cannot_update=True)
    (OUT/'causal_self_test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('CAUSAL SELF TEST PASS',flush=True)


def matched_white(fleet,q_scale=None):
    f=copy.deepcopy(fleet)
    for m in f['models']:
        m['F'][-1]=0.
        m['Q'][-1]=m['variance']*.8
        if q_scale is not None:
            m['Q'][:-1]=m['variance']*q_scale*np.r_[1.,np.full(len(m['m'])-2,.05)]
    return f


def choose_white_drift():
    f=make_site('Manchester',stage='development',**CONFIG['calibrator'])
    q,_=fixed_site_mask(len(f['y']),int(f['ready'].min()),f['initialization_mask'],budget=48,phase=0.)
    rows=[]
    for drift in (.0001,.001,.01):
        z=matched_white(f,drift);o=revised.run_site_fixed(z['models'],z['y'],z['ready'],z['initialization_mask'],q)
        rows.append(dict(q_scale=drift,rmse=macro(np.maximum(o['pred'],0.),f['y'],f['score_masks'],f['metadata'])))
    selected=min(rows,key=lambda r:(r['rmse'],r['q_scale']))['q_scale']
    payload=dict(stage='post-review pre2021 development choice',calendar='site-shared periodic B48 phase0',grid=rows,selected=selected)
    (OUT/'white_development.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return selected


def memory(fleet,query,kind):
    nd,n=fleet['score_masks'].shape;pred=np.full((nd,n),np.nan)
    times=fleet['times'];hours=((times-times[0]).total_seconds()/3600).to_numpy()
    daily=2*np.pi*hours/24.;weekly=2*np.pi*hours/168.
    profile=np.column_stack([np.ones(n),np.sin(daily),np.cos(daily),np.sin(weekly),np.cos(weekly)])
    access=query|fleet['initialization_mask'];y=fleet['y']
    for d,(model,row) in enumerate(zip(fleet['models'],fleet['metadata'])):
        a=row['local_offset'];r=int(fleet['ready'][d])
        if kind=='sensor_decay':
            mu=model['phi']@model['m'][:-1];rho=model['rho'];residual=0.
        else:
            ok=np.arange(a,r)[np.isfinite(y[a:r])]
            penalty=np.eye(5);penalty[0,0]=.01
            coef=np.linalg.solve(profile[ok].T@profile[ok]+penalty,profile[ok].T@y[ok])
            mu=profile@coef;res=y[a:r]-mu[a:r]
            pairs=np.isfinite(res[:-1])&np.isfinite(res[1:])
            rho=float(np.clip(res[:-1][pairs]@res[1:][pairs]/max(res[:-1][pairs]@res[:-1][pairs],1e-8),0,.95))
            last=int(ok[-1]);residual=(y[last]-mu[last])*rho**(r-1-last)
        for t in range(r,n):
            residual*=rho
            if model['valid'][t]:pred[d,t]=mu[t]+residual
            if access[t] and np.isfinite(y[t]) and (kind=='reference_only' or model['valid'][t]):
                residual=y[t]-mu[t]
    return dict(pred=pred,query=query,runtime_seconds=0.)


def filtered_fleet(fleet,source):
    f=copy.deepcopy(fleet);flags=source['quality_flag_mask'].astype(bool)
    f['score_masks']=f['score_masks']&~flags
    diagnostics=[]
    for d,row in enumerate(f['metadata']):
        a=row['local_offset'];b=int(f['times'].searchsorted(pd.Timestamp(row['end']),side='right'))
        X=f['Xs'][d][a:b].copy();X[flags[d,a:b],0]=np.nan
        guarded,_=guard_features(X,init_end=336,width=8.)
        m=initialize(guarded,f['y'][a:b],336,q_scale=.001,ar_share=.8)
        phi=np.zeros_like(f['models'][d]['phi']);phi[a:b]=m['phi']
        valid=np.zeros(len(f['y']),bool);valid[a:b]=m['valid']
        m['phi']=phi;m['valid']=valid;m['weight']=f['models'][d]['weight'];f['models'][d]=m
        diagnostics.append(dict(deployment=row['deployment'],original_init_pairs=fleet['models'][d]['init_pairs'],filtered_init_pairs=m['init_pairs'],filtered_scored=int(flags[d,fleet['score_masks'][d]].sum())))
    return f,diagnostics


def main(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    if stage=='test':self_test();return
    summaries=[]
    selected=choose_white_drift() if stage=='components' else None
    for site in CONFIG['sites']:
        f=make_site(site,**CONFIG['calibrator'])
        for policy in POLICIES:
            source,path=load_source(site,policy)
            if stage=='arrival':
                common=f['score_masks']&(np.arange(len(f['y']))[None,:]>=f['ready'][:,None]+168)
                for delay in (0,24,72,168):
                    z=run_policy(f,policy,source,delay=delay)
                    assert np.array_equal(z['query'],source['query']) if delay==0 else z['additional_attempts']==source['query'].sum()
                    if delay==0:np.testing.assert_allclose(np.maximum(z['pred'],0.),source['pred'],equal_nan=True,rtol=0,atol=0)
                    summaries.append(save(f,z,source,path,f'{policy}_delay{delay}',score_mask=common,extra=dict(delay=delay,common_post_ready=168)))
            elif stage=='setup':
                z=run_policy(f,policy,source,shared_setup=False)
                assert z['additional_attempts']==source['query'].sum()
                summaries.append(save(f,z,source,path,f'{policy}_own_setup',extra=dict(shared_setup=False)))
            elif stage=='components' and policy=='periodic':
                for name,drift in [('white_matched',None),('white_retuned',selected)]:
                    ff=matched_white(f,drift)
                    z=run_policy(ff,policy,source)
                    summaries.append(save(ff,z,source,path,name,extra=dict(q_scale=drift)))
            elif stage=='memory' and policy!='risk':
                for name in ('sensor_decay','reference_only'):
                    z=memory(f,source['query'].astype(bool),name)
                    summaries.append(save(f,z,source,path,policy+'_'+name))
            elif stage=='quality' and policy!='risk':
                ff,diagnostics=filtered_fleet(f,source)
                z=run_policy(ff,policy,source)
                summaries.append(save(ff,z,source,path,policy+'_filtered_replay',extra=dict(flag_init_diagnostics=diagnostics)))
                original_output=dict(pred=source['pred'],query=source['query'])
                summaries.append(save(ff,original_output,source,path,policy+'_clean_scoring_only',extra=dict(replay=False)))
        pd.DataFrame(summaries).to_csv(OUT/f'{stage}_summary.csv',index=False)
    print('COMPLETE',stage,len(summaries),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['test','arrival','setup','components','memory','quality'],required=True)
    main(ap.parse_args().stage)
