"""Selected-GP correction on the existing 72-hour delivered AR calendars.

This is a fixed-calendar nonlinear delivery comparator, not delayed GP-active
policy optimization. All 72-hour and zero-delay comparisons use the original
59,674-instrument-hour delivery support. Acquired event u is delivered after
the prediction at u+72. GP refits retain ninety event-time calendar days and
at most 512 delivered pairs, ending at the newest delivered event.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for key in ("TEMP","TMP"):os.environ[key]=str(ROOT/"temp")
os.environ["NUMBA_CACHE_DIR"]=str(ROOT/"cache/numba")
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="2"
import argparse,json
from datetime import datetime,timezone
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import revision4_gp_active as active
from site_data import make_site
from statistical_analysis import load_run,build_site_bundle,metric_tables,paired_calendar_bootstrap

OUT=ROOT/"results/revision4/gp_delay"
SITES=active.SITES;PARAMS=active.PARAMS;DELAY=72
METHODS=("gp_periodic_d0","gp_fixed_d0","gp_periodic_d72","gp_fixed_d72")
PLAN=dict(stage="Fourth-review retrospective fixed-calendar nonlinear extension",
    parameters=PARAMS,delay_hours=72,
    source_calendars="Existing access_v2 periodic and regenerated AR fixed-risk delay72 calendars; not GP-active calendars",
    prediction_order="Prediction at delivery t precedes inspection and update from acquired event u=t-72",
    ready="Own commissioning fit available at original ready+72",
    memory="Latest ninety calendar event-time days ending at the newest delivered event+1; at most 512 delivered pairs. Transformations/target scale fitted only on own commissioning.",
    shared_setup=True,budget=48,block=672,max_starts=4,
    support="Original common delay support starts168h after each original ready; 59674 instrument-hours/64886 channel observations",
    analysis="Seven/fourteen-day synchronized site-quarter circular blocks,1000 draws; within-delay fixed-minus-periodic and four-stream change relative to zero-delay; no retuning or independent geographic inference")


def plan():
    identity=dict(plan=PLAN,code_sha256={n:active.sha(ROOT/"scripts"/n) for n in ("revision4_gp_active.py","competitive_stream_baselines.py","calibration_engine.py","revision_access_controller.py","site_data.py","study_data.py","statistical_analysis.py")},
        original_freeze_sha256=active.sha(ROOT/"evidence/FREEZE.json"),
        selection_sha256=active.sha(ROOT/"results/revision/competitive/shared_development_selection.json"))
    p=OUT/"plan.json"
    if p.exists():assert json.loads(p.read_text(encoding="utf-8"))["identity"]==identity
    else:active.write_json(p,dict(recorded_utc=datetime.now(timezone.utc).isoformat(),identity=identity))
    return active.sha(p)


def replay_delayed(models,y,access,delay):
    pred=np.full((len(models),len(y)),np.nan);delivered=np.zeros(len(y),bool)
    original_ready=[m.ready for m in models]
    for m in models:m.ready+=delay
    for t in range(len(y)):
        for i,m in enumerate(models):
            if m.applicable(t):pred[i,t]=m.predict(t)
        u=t-delay
        if u>=0 and access[u]:
            delivered[u]=np.isfinite(y[u])
            if delivered[u]:
                for i,m in enumerate(models):
                    if original_ready[i]<=u<m.stop and m.valid[u-m.start]:m.observe(u,float(y[u]))
    return pred,delivered


def smoke():
    plan();rng=np.random.default_rng(20261004);n=820;initial=96
    X=rng.normal(size=(n,3));X[:,0]+=12.;y=1.2*X[:,0]+rng.normal(size=n)
    access=np.arange(n)<initial;access[initial+12::49]=True
    def run(target=y,features=X,delay=72):
        with threadpool_limits(limits=2):
            model=active.GPStream(features,target[:initial],np.isfinite(features[:,0]),initial=initial)
            p,d=replay_delayed([model],target,access,delay)
        return p[0],d,model
    base,delivered,model=run()
    altered=y.copy();hidden=~access;altered[hidden]=1e8;altered[np.flatnonzero(hidden)[::2]]=np.nan
    np.testing.assert_allclose(base,run(altered)[0],rtol=0,atol=0,equal_nan=True)
    event=initial+12;altered=y.copy();altered[event]+=100
    changed=run(altered)[0]
    np.testing.assert_allclose(base[:event+72+1],changed[:event+72+1],rtol=0,atol=0,equal_nan=True)
    assert np.nanmax(np.abs(base[event+73:]-changed[event+73:]))>.01
    cut=400;future=X.copy();future[cut:]+=100
    np.testing.assert_allclose(base[:cut],run(features=future)[0][:cut],rtol=0,atol=0,equal_nan=True)
    assert np.isnan(base[:initial+72]).all() and np.isfinite(base[initial+72:]).all()
    zero=run(delay=0)[0]
    expected=active.original.fit_predict_stream_baselines(X,y,initial,access,model_names=("gp_stream",),**PARAMS)["gp_stream"]
    np.testing.assert_allclose(zero,np.maximum(expected,0.),rtol=1e-12,atol=1e-12,equal_nan=True)
    # Independent prediction reconstruction uses only delivered event-time pairs.
    largest=0.
    for t in (initial+72, initial+72+40, 430, 650, n-1):
        m=active.GPStream(X,y[:initial],np.ones(n,bool),initial=initial)
        for u in range(initial,t-72):
            if access[u] and np.isfinite(y[u]):m.observe(u,float(y[u]))
        p=m.predict(t);largest=max(largest,abs(p-base[t]))
        np.testing.assert_allclose(p,base[t],rtol=1e-12,atol=1e-12)
    receipt=dict(status="PASS",checks=["hidden target values/missingness", "delivery-hour target cannot change its prediction", "next-hour delayed update effect", "future feature invariance", "postponed commissioning readiness", "zero-delay original GP parity", "independent delivered-history reconstruction"],
        reconstruction_max_difference=largest,runner_sha256=active.sha(__file__))
    active.write_json(OUT/"smoke_checks.json",receipt);print(json.dumps(receipt,indent=2),flush=True)


def run():
    identity=plan()
    for site in SITES:
        fleet=make_site(site,stage="test",q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
        main=ROOT/"results/main"/f"{site}_B48_periodic.npz"
        source,local=active.frozen._load_source(main,fleet)
        aligned_ready=np.asarray([item["start"]+336 for item in local])
        common=source["score_masks"]&(np.arange(len(source["y"]))[None,:]>=aligned_ready[:,None]+168)
        for policy in ("periodic","risk_fixed"):
            calendar=ROOT/"results/revision/access_v2"/f"{site}_B48_{policy}_delay72.npz"
            with np.load(calendar,allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
            for key in ("y","times","initmask","metadata_json"):
                assert np.array_equal(old[key],source[key],equal_nan=True) if old[key].dtype.kind in "fc" else np.array_equal(old[key],source[key]),key
            np.testing.assert_array_equal(old["score_masks"],common)
            p=OUT/"streams"/f"{site}_B48_gp_{policy}_delay72.npz"
            provenance=dict(plan_sha256=identity,runner_sha256=active.sha(__file__),calendar_sha256=active.sha(calendar),calendar=str(calendar),parameters=PARAMS)
            if p.exists():
                with np.load(p,allow_pickle=False) as z:assert json.loads(z["revision4_provenance_json"].item())==provenance
                load_run(p);print("RESUME",p.name,flush=True);continue
            begin=perf_counter()
            with threadpool_limits(limits=2):
                models=active.models_from_local(local,source,fleet)
                prediction,delivered=replay_delayed(models,source["y"],old["query"]|old["initmask"],72)
            saved=active.frozen._shared_arrays(old)
            saved.update(pred=prediction,delivered=delivered,runtime_seconds=np.asarray(perf_counter()-begin),
                revision4_provenance_json=np.asarray(json.dumps(provenance,sort_keys=True)),
                runtime_json=np.asarray(json.dumps([dict(deployment=m.name,**m.diag) for m in models])))
            assert np.isfinite(prediction[common]).all()
            assert all(m.diag["max_training_age_hours"]<=2160 and m.diag["max_training_pairs"]<=512 for m in models)
            p.parent.mkdir(parents=True,exist_ok=True)
            tmp=p.with_suffix(".npz.tmp")
            with tmp.open("wb") as handle:np.savez_compressed(handle,**saved)
            tmp.replace(p);load_run(p)
            print("SAVED",p.name,"wall_seconds",float(saved["runtime_seconds"]),flush=True)


def summarize():
    plan();bundles=[];cells=[]
    for site in SITES:
        runs={}
        for method in METHODS:
            fixed="fixed" in method;policy="risk_fixed" if fixed else "periodic"
            if method.endswith("d72"):p=OUT/"streams"/f"{site}_B48_gp_{policy}_delay72.npz"
            else:p=ROOT/"results/revision/competitive/shared_selected"/f"{site}_B48_gp_shareddev_{policy}.npz"
            r=load_run(p);r.method=method
            if method.endswith("d0"):
                with np.load(ROOT/"results/revision/access_v2"/f"{site}_B48_periodic_delay72.npz",allow_pickle=False) as z:r.score_masks=z["score_masks"].copy()
            runs[method]=r;cells.append(dict(site=site,method=method,path=str(p),sha256=active.sha(p),score_support="original168h-common-delay mask"))
        bundles.append(build_site_bundle(runs,methods=METHODS))
    tables=metric_tables(bundles)
    for name,frame in tables.items():frame.to_csv(OUT/f"{name}.csv",index=False)
    rows=[];diagnostics=[]
    for days in (7,14):
        result=paired_calendar_bootstrap(bundles,n_replicates=1000,block_days=days,seed=20261004+days)
        diagnostics.append(result["diagnostics"])
        for metric,samples,observed in (("macro_instrument_rmse",result["primary"],result["observed_primary"]),("site_balanced_rmse",result["secondary"],result["observed_secondary"])):
            for label,weights in (("fixed-minus-periodic d0",[-1,1,0,0]),("fixed-minus-periodic d72",[0,0,-1,1]),("change in timing advantage d72-minus-d0",[1,-1,-1,1])):
                weights=np.asarray(weights);draws=samples@weights;low,high=np.quantile(draws,[.025,.975])
                rows.append(dict(contrast=label,metric=metric,difference=float(observed@weights),ci_low=float(low),ci_high=float(high),block_days=days))
    pd.DataFrame(rows).to_csv(OUT/"paired_contrasts.csv",index=False)
    active.write_json(OUT/"bootstrap_diagnostics.json",diagnostics)
    active.write_json(OUT/"source_index.json",dict(plan_sha256=plan(),runner_sha256=active.sha(__file__),cells=cells))
    print(tables["aggregate_metrics"].to_string(index=False),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def verify():
    """Direct GP fits from independently formed delivered event-time index sets."""
    plan();records=[];total=0;largest=0.
    for site in SITES:
        fleet=make_site(site,stage="test",q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
        source,local=active.frozen._load_source(ROOT/"results/main"/f"{site}_B48_periodic.npz",fleet)
        for policy in ("periodic","risk_fixed"):
            path=OUT/"streams"/f"{site}_B48_gp_{policy}_delay72.npz"
            with np.load(path,allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
            source_calendar=ROOT/"results/revision/access_v2"/f"{site}_B48_{policy}_delay72.npz"
            with np.load(source_calendar,allow_pickle=False) as z:
                for key in ("y","times","score_masks","query","initmask","metadata_json"):
                    assert np.array_equal(saved[key],z[key],equal_nan=True) if saved[key].dtype.kind in "fc" else np.array_equal(saved[key],z[key]),key
            allaccess=saved["query"]|saved["initmask"];row_count=0;maxdiff=0.
            with threadpool_limits(limits=2):
                for j,item in enumerate(local):
                    left,right=item["start"],item["stop"];event_y=source["y"][left:right]
                    initialization=np.flatnonzero(item["valid"][:336]&np.isfinite(event_y[:336]))
                    usable=np.flatnonzero(item["valid"]&np.isfinite(event_y)&allaccess[left:right])
                    usable=usable[usable>=336]
                    score=np.flatnonzero(saved["score_masks"][j])
                    probes=np.unique(score[[0,len(score)//2,-1]])
                    for t in probes:
                        fresh=active.GPStream(item["X"],event_y[:336],item["valid"])
                        permitted=usable[usable<t-left-72]
                        if len(permitted):
                            chosen=np.r_[initialization,permitted]
                            chosen=chosen[chosen>=permitted[-1]+1-2160][-512:]
                            fresh.model.fit(fresh.Z[chosen],(event_y[chosen]-fresh.mean)/fresh.scale)
                        predicted=max(float(fresh.model.predict(fresh.Z[t-left:t-left+1])[0]*fresh.scale+fresh.mean),0.)
                        error=abs(predicted-float(saved["pred"][j,t]));maxdiff=max(maxdiff,error)
                        np.testing.assert_allclose(predicted,saved["pred"][j,t],rtol=1e-12,atol=1e-12)
                        row_count+=1
            records.append(dict(site=site,policy=policy,checked_direct_predictions=row_count,max_abs_difference=maxdiff,
                instrument_channel_observations=int(saved["score_masks"].sum()),output_sha256=active.sha(path),status="PASS"))
            total+=row_count;largest=max(largest,maxdiff)
            print("VERIFIED DIRECT",site,policy,row_count,"predictions",maxdiff,flush=True)
    active.write_json(OUT/"verification.json",dict(status="PASS",checks=records,total_direct_predictions=total,
        max_abs_difference=largest,runner_sha256=active.sha(__file__),
        scope="Direct GP fits at first/middle/last common scored hour of every segment, using independently constructed available event indices. All source support/calendars/references are exact; synthetic checks verify arrival order."))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage",required=True,choices=("plan","smoke","run","summarize","verify"));args=parser.parse_args()
    if args.stage=="plan":print(plan(),flush=True)
    elif args.stage=="smoke":smoke()
    elif args.stage=="run":run()
    elif args.stage=="summarize":summarize()
    elif args.stage=="verify":verify()


if __name__=="__main__":main()
