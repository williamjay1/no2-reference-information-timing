"""Audit-driven component checks, no tuning or changes to primary methods."""
from pathlib import Path
import json,hashlib
import numpy as np
from site_data import make_site
from run_main import save_run,extra_arrays,ROOT
from acquisition_controller import run_site,run_site_fixed


def main():
    manifest=ROOT/'evidence/FREEZE.json'
    c=json.loads(manifest.read_text());mh=hashlib.sha256(manifest.read_bytes()).hexdigest()
    for s in c['sites']:
        f=make_site(s,**c['calibrator']);extras=extra_arrays(f)
        with np.load(ROOT/'results/main'/f'{s}_B48_periodic.npz') as z:q=z['query']
        for name,kw in [('guard_removed',dict(guard_width=float('inf'))),('weather_removed',dict(weather=False))]:
            ff=make_site(s,**(c['calibrator']|kw))
            assert np.array_equal(ff['score_masks'],f['score_masks'])
            out=run_site_fixed(ff['models'],f['y'],f['ready'],f['initialization_mask'],q)
            save_run(f,out,48,name,extras,mh)
        for policy,name in [('risk_independent','risk_independent_matched'),('risk_fixed','risk_fixed_matched')]:
            out=run_site(f['models'],f['y'],f['ready'],f['initialization_mask'],budget=48,policy=policy,**c['policies']['risk'])
            assert out['additional_attempts']==int(q.sum())
            save_run(f,out,48,name,extras,mh)


if __name__=='__main__':main()
