"""Evaluate frozen Gaussian beliefs; diagnostics do not tune any model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from site_data import make_site
from acquisition_controller import run_site_fixed
from study_data import ROOT


def main():
    c=json.loads((ROOT/'evidence/FREEZE.json').read_text());rows=[];ref=[]
    for site in c['sites']:
        f=make_site(site,**c['calibrator'])
        for method in ['periodic','risk','risk_fixed']:
            with np.load(ROOT/'results/main'/f'{site}_B48_{method}.npz') as z:
                q=z['query']; saved=z['pred']; masks=z['score_masks']
            out=run_site_fixed(f['models'],f['y'],f['ready'],f['initialization_mask'],q)
            np.testing.assert_allclose(np.maximum(out['pred'],0),saved,equal_nan=True,atol=0,rtol=0)
            for instrument in sorted({r['instrument'] for r in f['metadata']}):
                ids=[i for i,r in enumerate(f['metadata']) if r['instrument']==instrument]
                m=masks[ids]; count=m.sum(axis=0); active=count>0
                errors=np.where(m,out['pred'][ids]-f['y'][None,:],0)
                v=out['var'][ids];den=np.maximum(count,1)
                cover=(m&(np.abs(errors)<=1.95996398454*np.sqrt(v))).sum(axis=0)/den
                se=(errors**2).sum(axis=0)/den
                pv=np.where(m,v,0).sum(axis=0)/den
                validcor=active&np.isfinite(pv)&np.isfinite(se)
                rho=float(spearmanr(pv[validcor],se[validcor]).statistic) if validcor.sum()>2 else None
                rows.append(dict(site=site,method=method,instrument=instrument,hours=int(active.sum()),
                    nominal95_coverage=float(cover[active].mean()),
                    mean_gaussian_variance=float(pv[active].mean()),observed_mse=float(se[active].mean()),
                    observed_to_predicted_variance=float(se[active].mean()/pv[active].mean()),
                    variance_squared_error_spearman=rho,
                    negative_raw_prediction_channel_hours=int((m&(out['pred'][ids]<0)).sum())))
        del f
    pd.DataFrame(rows).to_csv(ROOT/'results/uncertainty_diagnostics.csv',index=False)
    d=pd.read_parquet(ROOT/'datasets/duplicate_reference_no2.parquet')
    p=pd.read_parquet(ROOT/'datasets/reference_no2_ratified.parquet')
    d=d[(d.timestamp>=pd.Timestamp('2021-01-01'))&d.qc_flag.eq(0)]
    d=d[['timestamp','site','duplicate_y','qc_flag']].merge(p,on=['timestamp','site'],how='inner');e=d.duplicate_y-d.y
    summary=dict(purpose='Post-result diagnostic only; no tuning or model selection',
        coverage='Unclipped pre-update Gaussian observation intervals; nominal95%; scored support unchanged',
        duplicate_reference_difference=dict(n_hours=len(e),bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.abs(e).mean())),
        aggregate=json.loads(pd.DataFrame(rows).groupby('method')[['nominal95_coverage','observed_to_predicted_variance','variance_squared_error_spearman']].mean().reset_index().to_json(orient='records')))
    (ROOT/'results/uncertainty_diagnostics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
