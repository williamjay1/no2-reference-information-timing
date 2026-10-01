"""Replay permission-only ablations with independent allocation and information masks."""
from pathlib import Path
import hashlib,json,shutil
import revision6_fixed_delay as matrix
import revision6_contract_controller as controller
from statistical_analysis import load_run
from revision_access import macro,CONFIG
from site_data import make_site
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/revision6/fixed_delay'
OUT=ROOT/'results/revision6/fixed_delay_corrected'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for path in SOURCE.iterdir():
        if path.is_file() and (path.suffix!='.npz' or '_shared1.npz' in path.name or 'development' in path.name):
            shutil.copy2(path,OUT/path.name)
    selected=json.loads((SOURCE/'selected_expanded.json').read_text())
    rows=[]
    for site in matrix.SITES:
        f=make_site(site,**CONFIG['calibrator'])
        with np.load(ROOT/'results/main'/f'{site}_B48_periodic.npz') as p:q=p['query'].copy()
        for shared in (True,False):
            for delay in matrix.DELAYS:
                for policy in matrix.POLICIES:
                    path=OUT/f'{site}_B48_{policy}_d{delay}_shared{int(shared)}.npz'
                    h=int(selected['horizon']) if policy=='aware_expanded' else 240 if policy=='aware_retained' else 168
                    threshold=float(selected['threshold']) if policy=='aware_expanded' else 1.25
                    if not path.exists():
                        if policy=='periodic':
                            z=controller.run_site_fixed(f['models'],f['y'],f['ready'],f['initialization_mask'],q,delay=delay,shared_setup=shared)
                        else:
                            z=controller.run_site(f['models'],f['y'],f['ready'],f['initialization_mask'],budget=48,block=672,
                                max_starts=4,policy='risk_fixed',horizon=h,threshold_scale=threshold,fixed_length=12,
                                delay=delay,shared_setup=shared,delay_aware=policy!='original')
                        assert int(z['query'].sum())==int(q.sum())
                        allocation=f['initialization_mask']
                        if allocation.ndim>1:allocation=allocation.any(axis=0)
                        assert not np.any(z['query'] & allocation)
                        if policy!='periodic':assert np.max(z['starts'])<=4
                        matrix.save(path,f,z,dict(stage='corrected permissions-only retrospective ablation',site=site,
                            policy=policy,shared=shared,delay=delay,horizon=h,threshold=threshold,
                            allocation_mask='true initialization regardless of permission',
                            controller_sha256=hashlib.sha256(Path(controller.__file__).read_bytes()).hexdigest()))
                    run=load_run(path)
                    with np.load(path) as p:access=json.loads(p['access_json'].item())
                    row=dict(site=site,shared=shared,delay=delay,policy=policy,
                        rmse=macro(run.pred,run.y,run.score_masks,run.metadata),**access)
                    rows.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'matrix_site_results.csv',index=False)
    matrix.OUT=OUT
    matrix.summarize()
    (OUT/'correction.json').write_text(json.dumps(dict(
        source_directory=str(SOURCE.relative_to(ROOT)),shared_streams='copied exactly unchanged',
        nonshared_streams='all 60 replayed with independent masks; fixed periodic query unchanged',
        controller_sha256=hashlib.sha256(Path(controller.__file__).read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
