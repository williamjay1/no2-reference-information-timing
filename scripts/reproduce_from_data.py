"""Recompute periodic AR predictions from supplied public-data derivatives.

Nine site/budget runs are reproduced without reading the saved reference masks
or predictions to construct the models or calendars. Saved outputs are used
only afterward for exact verification. Outputs go to a fresh directory.
"""
from pathlib import Path
import os,argparse,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'cache/numba'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from site_data import make_site
from fixed_schedules import fixed_site_mask
from acquisition_controller import run_site_fixed
from statistical_analysis import load_run,build_site_bundle,metric_tables


def main(output):
    output=Path(output).resolve()
    if output.drive.upper()!='D:':raise ValueError('Computation outputs must remain on D:')
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))
    checks=[];bundles={b:[] for b in config['budgets']}
    for site in config['sites']:
        fleet=make_site(site,**config['calibrator'])
        for budget in config['budgets']:
            query,_=fixed_site_mask(len(fleet['y']),int(min(fleet['ready'])),fleet['initialization_mask'],
                budget=budget,phase=config['periodic_phase'])
            result=run_site_fixed(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'],query)
            pred=np.maximum(result['pred'],0.)
            original=load_run(ROOT/'results/main'/f'{site}_B{budget}_periodic.npz')
            np.testing.assert_array_equal(query,original.query)
            np.testing.assert_array_equal(fleet['score_masks'],original.score_masks)
            np.testing.assert_array_equal(pred,original.pred)
            original.pred=pred
            bundles[budget].append(build_site_bundle({'periodic':original}))
            checks.append(dict(site=site,budget=budget,prediction_parity='EXACT',calendar_parity='EXACT',channel_hours=int(fleet['score_masks'].sum())))
            print('PASS',site,budget,flush=True)
    errors=pd.concat([metric_tables(b)['aggregate_metrics'] for b in bundles.values()],ignore_index=True)
    errors.to_csv(output/'from_data_aggregate_errors.csv',index=False)
    (output/'from_data_validation.json').write_text(json.dumps(dict(status='PASS',checks=checks,data_sha256={name:hashlib.sha256((ROOT/'datasets'/name).read_bytes()).hexdigest() for name in ('no2_hourly_oob.parquet','reference_no2_ratified.parquet')}),indent=2),encoding='utf-8')
    print('Recomputed all nine periodic streams from data with exact saved-output agreement.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/reproduction_from_data'))
    main(p.parse_args().output)
