"""Independent GP posterior-variance reference windows; fourth-review extension.

Frozen correction parameters come from the existing pre-2021 Manchester
selection. A GP, not an AR trajectory, chooses nominal twelve-hour site windows.
All targets enter after the current prediction and access decision. Planning
uses a fully historical context path, sparse integration points and known
external commissioning access. This is an explicit variance-reduction
comparator, not an exact reproduction of a published active-learning pipeline.
Only new revision4 outputs are written; existing sources are immutable.
"""
from __future__ import annotations

import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for key in ("TEMP", "TMP"):
    os.environ[key] = str(ROOT/"temp")
os.environ["NUMBA_CACHE_DIR"] = str(ROOT/"cache/numba")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import argparse
import hashlib
import json
from datetime import datetime, timezone
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import competitive_stream_baselines as original
import run_main_baselines as frozen
from site_data import make_site
from statistical_analysis import load_run, build_site_bundle, metric_tables, paired_calendar_bootstrap

OUT = ROOT/"results/revision4/gp_active"
SITES = ("Manchester", "London", "York")
PARAMS = dict(gp_length_scale=2., gp_noise=.1)
OFFSETS = np.asarray([6, 12, 24, 48, 72, 96, 120, 144, 168], dtype=int)
QUADRATURE = np.diff(np.r_[0, OFFSETS])/168.
SETTINGS = dict(budget=48, block=672, max_starts=4, fixed_length=12,
                horizon=168, candidate_stride=12, threshold_scale=1.25,
                lookback=336, historical_lag=192)
PLAN = dict(stage="Fourth-review retrospective extension; original evaluation results were already known",
    parameters=PARAMS, settings=SETTINGS, evaluation_offsets=OFFSETS.tolist(),
    quadrature_weights=QUADRATURE.tolist(), historical_scenarios=1,
    context="One full historical 168-hour path from t-192; its endpoint precedes t. Current context replaces missing historical inputs. Before a complete historical path exists, use frozen current context.",
    future_setup="Known exogenous setup observations condition both no-paid and paid paths after their predictions; paid prefixes pause during setup.",
    objective="Mean GP latent posterior variance decrease at right-endpoint quadrature points; convert each segment to ppb squared and aggregate using physical-instrument channel weights.",
    covariance="Fixed current GP kernel/posterior; hypothetical labels are integrated out. Future age pruning and pair-cap removal are not simulated by this planning approximation.",
    resources="B48 attempted hours per 672-hour block, four starts, nominal twelve-hour windows; shared final-start/catch-up enforcement may extend a window. Hourly enforced attempts are independent of label availability.",
    tuning="No new parameter search. GP length/noise reuse the existing early-Manchester shared-task selection. Planner cadence/quadrature/one scenario are specified before this new run for bounded computation. Threshold scale and resource enforcement match fixed AR.",
    comparison=["same-parameter GP periodic", "same-parameter GP on AR fixed-risk calendar", "same-parameter GP on AR variable-risk calendar"],
    scope="Immediate delivery and shared commissioning only; all three original evaluation sites and eighteen physical instruments. GP generates its own calendar. This does not isolate covariance family alone because context quadrature and candidate cadence differ from the hourly AR planner.",
    analysis="Instrument-equal RMSE/MAE and per-device values on identical support; synchronized paired circular site-quarter blocks, 1000 replicates at seven and fourteen days; conditional observed streams without retraining or selection uncertainty.")


def sha(path):
    return frozen._hash(Path(path))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path=ROOT/"results/revision/competitive/shared_development_selection.json"
    selected=json.loads(selection_path.read_text(encoding="utf-8"))
    assert selected["selection_data_end"] < "2021-01-01"
    assert selected["selected"]["gp"]["parameters"] == PARAMS
    identity=dict(plan=PLAN, original_freeze_sha256=sha(ROOT/"evidence/FREEZE.json"),
        selected_parameters_sha256=sha(selection_path),
        fixed_source_sha256={n:sha(ROOT/"scripts"/n) for n in (
            "calibration_engine.py", "acquisition_controller.py", "competitive_stream_baselines.py",
            "revision_competitive.py", "run_main_baselines.py", "site_data.py", "study_data.py", "statistical_analysis.py")},
        data_sha256={n:sha(ROOT/"datasets"/n) for n in ("no2_hourly_oob.parquet", "reference_no2_ratified.parquet")})
    path=OUT/"plan.json"
    if path.exists():
        previous=json.loads(path.read_text(encoding="utf-8"))
        assert previous["identity"] == identity, "Existing prospective settings or inputs differ"
    else:
        write_json(path,dict(recorded_utc=datetime.now(timezone.utc).isoformat(), identity=identity))
    return sha(path)


class GPStream:
    """Same preprocessing, mean prediction and pair-memory rules as frozen GP."""
    def __init__(self, X, y_initial, valid, initial=336, weight=1., start=0, name="synthetic"):
        self.start=int(start); self.stop=self.start+len(X); self.ready=self.start+initial
        self.weight=float(weight); self.name=name
        self.X=np.asarray(X,float).copy(); self.valid=np.asarray(valid,bool).copy()
        self.valid &= np.isfinite(self.X[:,0]); self.X[~np.isfinite(self.X)]=np.nan
        rows=np.flatnonzero(self.valid[:initial]); pairs=rows[np.isfinite(y_initial[rows])]
        assert len(pairs)>=2
        begin=perf_counter()
        imputer=SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        initial_X=imputer.fit_transform(self.X[rows])
        scaler=StandardScaler().fit(initial_X)
        self.Z=scaler.transform(imputer.transform(self.X))
        self.mean=float(np.mean(y_initial[pairs])); self.scale=max(float(np.std(y_initial[pairs])),1.)
        self.diag=dict(preprocessing_seconds=perf_counter()-begin, fit_seconds=0., predict_seconds=0.,
            score_seconds=0., fits=0, updates=0, scores=0, max_training_pairs=0,
            max_training_age_hours=0, max_score_contexts=0)
        self.observed=pairs.tolist(); self.observed_y=np.asarray(y_initial[pairs],float).tolist()
        self.fit(initial)

    def fit(self, next_local):
        keep=np.asarray(self.observed,int)>=next_local-2160
        self.observed=np.asarray(self.observed,int)[keep].tolist()
        self.observed_y=np.asarray(self.observed_y,float)[keep].tolist()
        indices=np.asarray(self.observed,int)[-512:]
        target=np.asarray(self.observed_y,float)[-512:]
        begin=perf_counter()
        self.model=GaussianProcessRegressor(
            kernel=ConstantKernel(1.,constant_value_bounds="fixed")*
                   RBF(np.full(self.Z.shape[1],PARAMS["gp_length_scale"]),length_scale_bounds="fixed")+
                   WhiteKernel(noise_level=PARAMS["gp_noise"],noise_level_bounds="fixed"),
            alpha=1e-6,optimizer=None,normalize_y=False)
        self.model.fit(self.Z[indices],(target-self.mean)/self.scale)
        self.diag["fit_seconds"]+=perf_counter()-begin; self.diag["fits"]+=1
        self.diag["max_training_pairs"]=max(self.diag["max_training_pairs"],len(indices))
        self.diag["max_training_age_hours"]=max(self.diag["max_training_age_hours"],int(next_local-indices[0]))

    def applicable(self, t):
        return self.ready<=t<self.stop and self.valid[t-self.start]

    def predict(self, t):
        begin=perf_counter()
        pred=float(self.model.predict(self.Z[t-self.start:t-self.start+1])[0]*self.scale+self.mean)
        self.diag["predict_seconds"]+=perf_counter()-begin
        return max(pred,0.)

    def observe(self, t, target):
        self.observed.append(t-self.start); self.observed_y.append(float(target))
        self.fit(t-self.start+1); self.diag["updates"]+=1

    def covariance(self, U):
        kernel=self.model.kernel_.k1
        cross=kernel(self.model.X_train_,U)
        v=solve_triangular(self.model.L_,cross,lower=True,check_finite=False)
        C=kernel(U)-v.T@v
        return .5*(C+C.T)

    def utility(self, t, length, initmask, settings):
        begin=perf_counter(); H=settings["horizon"]
        local_t=t-self.start; historical=local_t-settings["historical_lag"]
        complete=historical>=0 and historical+H<local_t
        free=np.zeros(H+1,bool)
        last=min(len(initmask),t+H+1)
        free[:last-t]=initmask[t:last]
        paid=[]
        for j in range(H+1):
            if not free[j] and len(paid)<length: paid.append(j)
        offsets=OFFSETS[OFFSETS<=H]
        weights=QUADRATURE[:len(offsets)]; weights=weights/weights.sum()
        context_hours=np.asarray(sorted(set(paid+np.flatnonzero(free).tolist()+offsets.tolist())),int)
        U=np.repeat(self.Z[local_t][None,:],len(context_hours),axis=0)
        if complete:
            for k,j in enumerate(context_hours):
                ix=historical+j
                if j>0 and self.valid[ix]: U[k]=self.Z[ix]
        C=self.covariance(U)
        noise=PARAMS["gp_noise"]+1e-6
        value=sequential_variance_reduction(C,context_hours,offsets,weights,free,set(paid),noise)
        self.diag["score_seconds"]+=perf_counter()-begin; self.diag["scores"]+=1
        self.diag["max_score_contexts"]=max(self.diag["max_score_contexts"],len(U))
        return value*self.scale**2


def sequential_variance_reduction(C, hours, offsets, weights, free, paid, noise):
    """Exact sequential covariance updates on the supplied finite context set."""
    base=C.copy(); alt=C.copy(); value=0.
    eval_weights={int(j):float(w) for j,w in zip(offsets,weights)}
    for ix,j in enumerate(hours):
        if int(j) in eval_weights:
            value+=eval_weights[int(j)]*max(float(base[ix,ix]-alt[ix,ix]),0.)
        if free[j]:
            for matrix in (base,alt):
                column=matrix[:,ix].copy(); denominator=max(float(matrix[ix,ix])+noise,1e-12)
                matrix-=np.outer(column,column)/denominator
        elif int(j) in paid:
            column=alt[:,ix].copy(); denominator=max(float(alt[ix,ix])+noise,1e-12)
            alt-=np.outer(column,column)/denominator
    return max(value,0.)


def budget_arrays(n, ready, initmask, settings):
    origin=int(min(ready)); block=settings["block"]; budget=settings["budget"]
    nb=(n-origin+block-1)//block
    target=np.zeros(nb,int); feasible=np.zeros(nb,int); remaining=np.zeros(n,int)
    for b in range(nb):
        left=origin+b*block; right=min(n,left+block)
        target[b]=int(np.floor(budget*(right-left)/block))
        slots=(~initmask[left:right]).astype(int)
        remaining[left:right]=np.cumsum(slots[::-1])[::-1]
        feasible[b]=min(target[b],int(slots.sum()))
    return origin,target,feasible,remaining


def replay(models, y, initmask, *, settings=SETTINGS, external_query=None):
    n=len(y); ready=[m.ready for m in models]
    origin,target,feasible,slots=budget_arrays(n,ready,initmask,settings)
    pred=np.full((len(models),n),np.nan); query=np.zeros(n,bool); success=np.zeros(n,bool)
    starts=np.zeros(len(target),int); spent=np.zeros(len(target),int)
    start_mask=np.zeros(n,bool); forced=np.zeros(n,bool); lengths=np.zeros(n,int); scores=np.full(n,np.nan)
    left_window=0; forced_window=False; cooldown=False
    for t in range(n):
        active=[i for i,m in enumerate(models) if m.applicable(t)]
        for i in active: pred[i,t]=models[i].predict(t)
        if external_query is not None:
            query[t]=bool(external_query[t] and not initmask[t] and t>=origin)
        elif t>=origin:
            b=(t-origin)//settings["block"]
            if (t-origin)%settings["block"]==0: left_window=0;forced_window=False;cooldown=False
            budget=feasible[b]-spent[b]; available=slots[t]
            if not initmask[t] and budget>0:
                execute=False
                if left_window>0: execute=True
                elif budget>=available:
                    left_window=budget;forced_window=True;lengths[t]=left_window
                    if starts[b]<settings["max_starts"]: starts[b]+=1;start_mask[t]=True
                    execute=True
                elif cooldown: cooldown=False
                elif starts[b]<settings["max_starts"] and active and (t-origin)%settings["candidate_stride"]==0:
                    nominal=min(settings["fixed_length"],budget)
                    total_weight=sum(models[i].weight for i in active)
                    utility=sum(models[i].weight*models[i].utility(t,nominal,initmask,settings) for i in active)/total_weight
                    best=utility/(nominal+feasible[b]/settings["max_starts"])
                    scores[t]=best
                    previous=scores[max(0,t-settings["lookback"]):t]
                    previous=previous[np.isfinite(previous)]
                    rate=min(.95,max(.001,budget/max(available,1)))
                    threshold=float(np.quantile(previous,1-rate)) if len(previous) else best
                    if best>0 and best>=settings["threshold_scale"]*threshold:
                        ell=budget if starts[b]==settings["max_starts"]-1 else nominal
                        forced_window=ell!=nominal
                        left_window=min(ell,available);lengths[t]=left_window
                        starts[b]+=1;start_mask[t]=True;execute=True
                if execute:
                    query[t]=True;forced[t]=forced_window;spent[b]+=1;left_window-=1
                    if left_window==0: cooldown=True;forced_window=False
        # Neither value nor finiteness is inspected before predictions/actions.
        if initmask[t] or query[t]:
            success[t]=np.isfinite(y[t])
            if success[t]:
                for i in active: models[i].observe(t,float(y[t]))
    if external_query is None:
        assert np.array_equal(spent,feasible), (spent,feasible)
        assert np.all(starts<=settings["max_starts"])
        assert not np.any(query&initmask)
    return dict(pred=pred,query=query,success=success,start_mask=start_mask,
                forced_query=forced,chosen_lengths=lengths,candidate_scores=scores,
                block_target=target,block_feasible=feasible,block_spent=spent,
                starts=starts,block_origin=origin,budget_shortfall=target-spent)


def models_from_local(local, source, fleet):
    weights={r["deployment"]:m["weight"] for r,m in zip(fleet["metadata"],fleet["models"])}
    models=[]
    for item in local:
        start=item["start"]
        models.append(GPStream(item["X"],source["y"][start:start+336],item["valid"],
            start=start,weight=weights[item["row"]["deployment"]],name=item["row"]["deployment"]))
    return models


def run_site(site, stage="test", smoke_hours=None):
    plan_hash=plan(); begin=perf_counter()
    fleet=make_site(site,stage=stage,q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
    if stage=="test":
        source_path=ROOT/"results/main"/f"{site}_B48_periodic.npz"
    else:
        assert fleet["times"].max()<pd.Timestamp("2021-01-01")
        source_path=ROOT/"results/revision/competitive/development_calendar.npz"
    source,local=frozen._load_source(source_path,fleet)
    output=OUT/("streams" if stage=="test" else "development")/f"{site}_B48_gp_active.npz"
    provenance=dict(plan_sha256=plan_hash,runner_sha256=sha(__file__),source_npz=str(source_path),
        source_sha256=sha(source_path),stage=stage,parameters=PARAMS)
    if output.exists() and smoke_hours is None:
        with np.load(output,allow_pickle=False) as z: assert json.loads(z["revision4_provenance_json"].item())==provenance
        load_run(output);print("RESUME",output,flush=True);return
    with threadpool_limits(limits=2):
        models=models_from_local(local,source,fleet)
        n=len(source["y"]) if smoke_hours is None else min(len(source["y"]),smoke_hours)
        result=replay(models,source["y"][:n],source["initmask"][:n].astype(bool))
    elapsed=perf_counter()-begin
    records=[dict(deployment=m.name,**m.diag) for m in models]
    if smoke_hours is not None:
        payload=dict(site=site,hours=n,total_seconds=elapsed,models=records,
            attempts=int(result["query"].sum()),starts=result["starts"].tolist())
        write_json(OUT/f"real_smoke_{site}.json",payload); print(json.dumps(payload,indent=2),flush=True);return
    saved=frozen._shared_arrays(source)
    saved.update(result)
    saved["pred"]=np.maximum(saved["pred"],0.)
    saved["runtime_seconds"]=np.asarray(elapsed)
    saved["runtime_json"]=np.asarray(json.dumps(records))
    saved["revision4_provenance_json"]=np.asarray(json.dumps(provenance,sort_keys=True))
    finite=np.isfinite(saved["y"])
    saved["access_json"]=np.asarray(json.dumps(dict(initialization_attempts=int(saved["initmask"].sum()),
        additional_attempts=int(saved["query"].sum()),additional_successes=int((saved["query"]&finite).sum()),
        total_attempts=int((saved["query"]|saved["initmask"]).sum()),
        merged_episodes=int(((saved["query"]|saved["initmask"])&~np.r_[False,(saved["query"]|saved["initmask"])[:-1]]).sum()),
        forced_paid_hours=int(saved["forced_query"].sum()),runtime_seconds=elapsed)))
    for key in ("y","times","score_masks","initmask","metadata_json"):
        assert np.array_equal(saved[key],source[key],equal_nan=True) if saved[key].dtype.kind in "fc" else np.array_equal(saved[key],source[key]),key
    assert np.isfinite(saved["pred"][saved["score_masks"]]).all()
    assert int(saved["query"].sum())==int(source["query"].sum()),"Matched attempted budget changed"
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle: np.savez_compressed(handle,**saved)
    tmp.replace(output);load_run(output)
    write_json(output.with_suffix(".json"),dict(provenance=provenance,records=records,elapsed_seconds=elapsed))
    print("SAVED",output.name,"seconds",round(elapsed,2),"attempts",int(saved["query"].sum()),flush=True)


def smoke():
    plan(); rng=np.random.default_rng(20261003);n=1050;initial=96
    X=rng.normal(size=(n,3));X[:,0]+=10;X[12::49,1]=np.nan;X[29::61,0]=np.nan
    y=1.3*np.nan_to_num(X[:,0])+rng.normal(size=n);valid=np.isfinite(X[:,0])
    initmask=np.arange(n)<initial;initmask[500:520]=True
    def execute(features=X,targets=y,external=None,weight_duplicate=False):
        args=dict(initial=initial,start=0,weight=.5 if weight_duplicate else 1.)
        models=[GPStream(features,targets[:initial],np.isfinite(features[:,0]),**args)]
        if weight_duplicate: models.append(GPStream(features,targets[:initial],np.isfinite(features[:,0]),**args))
        with threadpool_limits(limits=2): result=replay(models,targets,initmask,external_query=external)
        return result,models
    begin=perf_counter();base,models=execute()
    hidden=~(base["query"]|initmask)
    altered=y.copy();altered[hidden]=1e8;altered[np.flatnonzero(hidden)[::2]]=np.nan
    changed,_=execute(targets=altered)
    np.testing.assert_array_equal(base["query"],changed["query"])
    np.testing.assert_allclose(base["pred"],changed["pred"],equal_nan=True,rtol=0,atol=0)
    event=int(np.flatnonzero(base["query"])[0]); altered=y.copy();altered[event]+=100
    changed,_=execute(targets=altered)
    np.testing.assert_allclose(base["pred"][:,:event+1],changed["pred"][:,:event+1],equal_nan=True,rtol=0,atol=0)
    np.testing.assert_array_equal(base["query"][:event+1],changed["query"][:event+1])
    assert np.nanmax(np.abs(changed["pred"][:,event+1:]-base["pred"][:,event+1:]))>.01
    cut=initial+300;altered_X=X.copy();altered_X[cut:]+=100
    changed,_=execute(features=altered_X)
    np.testing.assert_array_equal(base["query"][:cut],changed["query"][:cut])
    np.testing.assert_allclose(base["pred"][:,:cut],changed["pred"][:,:cut],equal_nan=True,rtol=0,atol=0)
    doubled,_=execute(weight_duplicate=True)
    np.testing.assert_array_equal(base["query"],doubled["query"])
    # Frozen external-mask chronology reproduces the existing GP exactly.
    replayed,_=execute(external=base["query"])
    expected=original.fit_predict_stream_baselines(X,y,initial,base["query"]|initmask,
        valid_x=valid,model_names=("gp_stream",),**PARAMS)["gp_stream"]
    np.testing.assert_allclose(replayed["pred"][0],np.maximum(expected,0),equal_nan=True,rtol=1e-12,atol=1e-12)
    # Latent covariance plus independent noise matches sklearn predictive covariance.
    model=models[0];U=model.Z[40:48]
    _,expected_cov=model.model.predict(U,return_cov=True)
    np.testing.assert_allclose(model.covariance(U)+PARAMS["gp_noise"]*np.eye(len(U)),expected_cov,rtol=1e-10,atol=1e-10)
    # Analytical constant latent process: exact finite-set reduction including free access.
    hours=np.arange(13);C=np.ones((13,13));free=np.zeros(13,bool);free[[2,3,5]]=True
    offsets=np.asarray([6,12]);weights=np.asarray([.5,.5]);paid={0,1,4}
    actual=sequential_variance_reduction(C,hours,offsets,weights,free,paid,1.)
    nb=na=0;expected_value=0.
    for j in hours:
        if j in offsets: expected_value+=.5*(1/(1+nb)-1/(1+na))
        if free[j]: nb+=1;na+=1
        elif j in paid: na+=1
    np.testing.assert_allclose(actual,expected_value,rtol=1e-12,atol=1e-12)
    assert np.array_equal(base["block_spent"],base["block_feasible"])
    assert np.all(base["starts"]<=4) and not np.any(base["query"]&initmask)
    checks=["hidden label value/missingness invariance", "current target cannot alter current prediction/decision",
        "immediate next-hour acquired-target effect", "future-feature invariance", "instrument-channel weight duplication",
        "same-GP external-mask prediction parity", "latent/noise covariance versus sklearn",
        "analytical sequential variance reduction with shared free access", "exact attempted budgets/start caps/setup exclusion"]
    payload=dict(status="PASS",checks=checks,elapsed_seconds=perf_counter()-begin,runner_sha256=sha(__file__),
                 tolerance=dict(prediction_parity=1e-12,covariance=1e-10),settings=SETTINGS)
    write_json(OUT/"smoke_checks.json",payload);print(json.dumps(payload,indent=2),flush=True)


def summarize():
    plan();cells=[];bundles=[];access=[]
    methods=("gp_active","gp_periodic","gp_ar_fixed","gp_ar_variable")
    for site in SITES:
        runs={}
        for method in methods:
            if method=="gp_active":p=OUT/"streams"/f"{site}_B48_gp_active.npz"
            else:
                policy={"gp_periodic":"periodic","gp_ar_fixed":"risk_fixed","gp_ar_variable":"risk"}[method]
                p=ROOT/"results/revision/competitive/shared_selected"/f"{site}_B48_gp_shareddev_{policy}.npz"
            run=load_run(p);run.method=method;runs[method]=run
            assert int(run.query.sum())==int(runs["gp_active"].query.sum())
            cells.append(dict(site=site,method=method,path=str(p),sha256=sha(p),parameters=PARAMS))
            all_access=run.initmask|run.query
            access.append(dict(site=site,method=method,commissioning_attempts=int(run.initmask.sum()),
                additional_attempts=int(run.query.sum()),total_attempts=int(all_access.sum()),
                merged_episodes=int((all_access&~np.r_[False,all_access[:-1]]).sum())))
        bundles.append(build_site_bundle(runs,methods=methods))
    tables=metric_tables(bundles)
    for name,frame in tables.items():frame.to_csv(OUT/f"{name}.csv",index=False)
    pd.DataFrame(access).to_csv(OUT/"access.csv",index=False)
    write_json(OUT/"source_index.json",dict(plan_sha256=plan(),runner_sha256=sha(__file__),cells=cells))
    contrasts=[];intervals=[];diagnostics=[]
    for days in (7,14):
        result=paired_calendar_bootstrap(bundles,n_replicates=1000,block_days=days,seed=20261003+days)
        diagnostics.append(result["diagnostics"])
        for metric,samples,observed in (("macro_instrument_rmse",result["primary"],result["observed_primary"]),
                                        ("site_balanced_rmse",result["secondary"],result["observed_secondary"])):
            for i,method in enumerate(methods):
                low,high=np.quantile(samples[:,i],[.025,.975])
                intervals.append(dict(method=method,metric=metric,estimate=float(observed[i]),ci_low=float(low),ci_high=float(high),block_days=days))
            for right in methods[1:]:
                a=methods.index("gp_active");b=methods.index(right)
                low,high=np.quantile(samples[:,a]-samples[:,b],[.025,.975])
                contrasts.append(dict(method="gp_active",comparator=right,metric=metric,
                    difference=float(observed[a]-observed[b]),ci_low=float(low),ci_high=float(high),block_days=days,
                    units="ppb",negative_favors="gp_active"))
    pd.DataFrame(contrasts).to_csv(OUT/"paired_contrasts.csv",index=False)
    pd.DataFrame(intervals).to_csv(OUT/"metric_intervals.csv",index=False)
    write_json(OUT/"bootstrap_diagnostics.json",diagnostics)
    print(tables["aggregate_metrics"].to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


def verify():
    """Separate chronological fixed-mask replay checks generated active streams."""
    records=[]
    for site in SITES:
        fleet=make_site(site,stage="test",q_scale=.001,use_ar=True,weather=True,ar_share=.8,guard_width=8.)
        source_path=ROOT/"results/main"/f"{site}_B48_periodic.npz"
        source,local=frozen._load_source(source_path,fleet)
        output=OUT/"streams"/f"{site}_B48_gp_active.npz"
        with np.load(output,allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
        largest=0.;access=saved["query"]|saved["initmask"]
        for j,item in enumerate(local):
            left,right=item["start"],item["stop"]
            with threadpool_limits(limits=2):
                actual=original.fit_predict_stream_baselines(item["X"],source["y"][left:right],336,
                    access[left:right],valid_x=item["valid"],model_names=("gp_stream",),**PARAMS)["gp_stream"]
            expected=np.maximum(actual,0.);observed=saved["pred"][j,left:right]
            np.testing.assert_allclose(observed,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
            difference=np.nanmax(np.abs(observed-expected))
            largest=max(largest,float(difference))
        for key in ("y","times","score_masks","initmask","metadata_json"):
            assert np.array_equal(saved[key],source[key],equal_nan=True) if source[key].dtype.kind in "fc" else np.array_equal(saved[key],source[key]),key
        assert int(saved["query"].sum())==int(source["query"].sum())
        assert np.array_equal(saved["block_spent"],saved["block_feasible"])
        assert np.all(saved["starts"]<=4) and not np.any(saved["query"]&saved["initmask"])
        records.append(dict(site=site,prediction_max_abs_difference=largest,deployments=len(local),
            attempted_hours=int(saved["query"].sum()),status="PASS",output_sha256=sha(output)))
        print("VERIFIED",site,"max difference",largest,flush=True)
    write_json(OUT/"verification.json",dict(status="PASS",checks=records,runner_sha256=sha(__file__),
        scope="Independent original-GP prediction replay on generated calendars, exact support/access identity, budget/start/setup checks. Synthetic smoke separately tests acquisition information order and the analytic objective."))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("plan","smoke","real-smoke","development","run","summarize","verify"))
    parser.add_argument("--sites",nargs="+",choices=SITES,default=list(SITES))
    parser.add_argument("--smoke-hours",type=int,default=1680)
    args=parser.parse_args()
    if args.stage=="plan":print(plan(),flush=True)
    elif args.stage=="smoke":smoke()
    elif args.stage=="real-smoke":
        for site in args.sites:run_site(site,stage="development",smoke_hours=args.smoke_hours)
    elif args.stage=="development":run_site("Manchester",stage="development")
    elif args.stage=="run":
        for site in args.sites:run_site(site)
    elif args.stage=="summarize":summarize()
    elif args.stage=="verify":verify()


if __name__=="__main__":main()
