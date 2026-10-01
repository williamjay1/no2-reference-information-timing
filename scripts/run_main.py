"""Execute the frozen future-period calibration/acquisition comparisons."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'cache/numba'))
import json,hashlib,time
import numpy as np
import pandas as pd
from site_data import make_site
from study_data import read_data
from acquisition_controller import run_site,run_site_fixed
from fixed_schedules import fixed_site_mask


def extra_arrays(fleet):
    data=read_data(); data=data[data.site.eq(fleet['site'])]
    warn=[]; qual=[]
    for r in fleet['metadata']:
        g=data[data.deployment.eq(r['deployment'])].set_index('timestamp').reindex(fleet['times'])
        w=g.sensor_flag.eq('Warning').fillna(False).to_numpy(dtype=bool)
        q=w|g.manufacturer_rebase_flag.fillna(False).to_numpy(bool)|g.retrospective_collection_flag.fillna(False).to_numpy(bool)
        warn.append(w);qual.append(q)
    out=dict(warning_mask=np.asarray(warn,dtype=bool),quality_flag_mask=np.asarray(qual,dtype=bool))
    dup=pd.read_parquet(ROOT/'datasets/duplicate_reference_no2.parquet')
    dup=dup[dup.site.eq(fleet['site'])]
    if len(dup):
        assert not dup.timestamp.duplicated().any()
        dup=dup.set_index('timestamp').reindex(fleet['times'])
        out['y_duplicate']=dup.duplicate_y.to_numpy(float)
        out['duplicate_valid']=dup.qc_flag.eq(0).to_numpy()&np.isfinite(out['y_duplicate'])&np.isfinite(fleet['y'])
    return out


def save_run(fleet,out,budget,method,extras,manifest_hash):
    pred=np.maximum(out['pred'],0.)
    assert np.isfinite(pred[fleet['score_masks']]).all(),method
    common=dict(pred=pred,y=fleet['y'],times=fleet['times'].astype(str).to_numpy(dtype='U19'),
        score_masks=fleet['score_masks'],query=out['query'],
        initmask=fleet['initialization_mask'],metadata_json=json.dumps(fleet['metadata']),
        runtime_seconds=float(out['runtime_seconds']),manifest_sha256=manifest_hash,
        **extras)
    for k in ['var','start_mask','forced_query','chosen_lengths','candidate_scores',
              'block_spent','block_target','budget_shortfall']:
        if k in out:common[k]=out[k]
    keys=['initialization_attempts','initialization_successes','additional_attempts','additional_successes','runtime_seconds']
    summary={k:out[k] for k in keys}
    summary.update(site=fleet['site'],budget=budget,method=method,
        starts=int(np.sum(out['starts'])),forced_hours=int(np.sum(out.get('forced_query',0))))
    common['access_json']=json.dumps(summary)
    p=ROOT/'results/main'/f"{fleet['site']}_B{budget}_{method}.npz"
    np.savez_compressed(p,**common)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    config_path=ROOT/'evidence/FREEZE.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    manifest_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
    (ROOT/'results/main').mkdir(exist_ok=True)
    access=[]
    for site in config['sites']:
        f=make_site(site,**config['calibrator']); extra=extra_arrays(f)
        noar=make_site(site,**(config['calibrator']|dict(use_ar=False)))
        assert np.array_equal(f['score_masks'],noar['score_masks'])
        print('SITE',site,len(f['models']),len(f['y']),flush=True)
        for b in config['budgets']:
            origin=int(min(f['ready']))
            q,qa=fixed_site_mask(len(f['y']),origin,f['initialization_mask'],budget=b,phase=config['periodic_phase'])
            out=run_site_fixed(f['models'],f['y'],f['ready'],f['initialization_mask'],q)
            access.append(save_run(f,out,b,'periodic',extra,manifest_hash))
            for seed in config['random_seeds']:
                rq,_=fixed_site_mask(len(f['y']),origin,f['initialization_mask'],budget=b,seed=seed)
                z=run_site_fixed(f['models'],f['y'],f['ready'],f['initialization_mask'],rq)
                assert z['additional_attempts']==out['additional_attempts']
                access.append(save_run(f,z,b,f'random{seed}',extra,manifest_hash))
            for policy,settings in config['policies'].items():
                z=run_site(f['models'],f['y'],f['ready'],f['initialization_mask'],budget=b,policy=policy,**settings)
                assert z['additional_attempts']==out['additional_attempts'],(site,b,policy)
                assert np.max(z['starts'])<=4
                access.append(save_run(f,z,b,policy,extra,manifest_hash))
            z=run_site_fixed(noar['models'],f['y'],f['ready'],f['initialization_mask'],q)
            # Common row identities, not AR-specific initialization diagnostics.
            access.append(save_run(f,z,b,'noAR',extra,manifest_hash))
            for method,mask in [('init_only',np.zeros(len(q),bool)),('reference_all',~f['initialization_mask'])]:
                z=run_site_fixed(f['models'],f['y'],f['ready'],f['initialization_mask'],mask)
                access.append(save_run(f,z,b,method,extra,manifest_hash))
            pd.DataFrame(access).to_csv(ROOT/'results/main_controller_access.csv',index=False)
    print('COMPLETE CONTROLLERS',len(access),flush=True)


if __name__=='__main__':main()
