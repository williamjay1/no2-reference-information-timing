"""Conditional paired intervals for non-selected GP planning-budget sensitivity.

Read-only prediction analysis; no new calibration fitting, model selection or
confirmatory p-values. Uses the existing synchronized site-quarter circular
seven-day block bootstrap, 1000 replicates, with physical-instrument weighting.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import revision6_gp_active as gp
from statistical_analysis import load_run, build_site_bundle, paired_calendar_bootstrap

def main():
    methods=('gp_enhanced','gp_original','gp_ar_fixed','gp_ar_variable','gp_periodic')
    bundles=[];sources=[]
    for site in gp.SITES:
        runs={}
        for method in methods:
            path=gp.path_for(site,method);run=load_run(path);run.method=method;runs[method]=run
            sources.append(dict(site=site,method=method,path=str(path),sha256=gp.base.sha(path)))
        bundles.append(build_site_bundle(runs,methods=methods))
    result=paired_calendar_bootstrap(bundles,n_replicates=1000,block_days=7,seed=20261008)
    contrasts=[];intervals=[]
    for metric,samples,observed in (('macro_instrument_rmse',result['primary'],result['observed_primary']),
                                    ('site_balanced_rmse',result['secondary'],result['observed_secondary'])):
        for index,method in enumerate(methods):
            low,high=np.quantile(samples[:,index],[.025,.975])
            intervals.append(dict(method=method,metric=metric,estimate=float(observed[index]),
                ci_low=float(low),ci_high=float(high),block_days=7,n_replicates=1000))
        for comparator in ('gp_periodic','gp_original','gp_ar_fixed'):
            left=methods.index('gp_enhanced');right=methods.index(comparator)
            low,high=np.quantile(samples[:,left]-samples[:,right],[.025,.975])
            contrasts.append(dict(method='gp_enhanced',comparator=comparator,metric=metric,
                difference=float(observed[left]-observed[right]),ci_low=float(low),ci_high=float(high),
                block_days=7,n_replicates=1000,units='ppb',negative_favors='gp_enhanced',
                status='exploratory conditional sensitivity; no confirmatory p-value'))
    pd.DataFrame(contrasts).to_csv(gp.OUT/'paired_contrasts.csv',index=False)
    pd.DataFrame(intervals).to_csv(gp.OUT/'metric_intervals.csv',index=False)
    gp.write_json(gp.OUT/'bootstrap_diagnostics.json',dict(diagnostics=result['diagnostics'],
        interpretation='Conditional uncertainty under observed streams and sites, no retraining or selection uncertainty; exploratory sensitivity, not independent confirmation.',
        analysis_runner_sha256=gp.base.sha(__file__),sources=sources))
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)

if __name__=='__main__': main()
