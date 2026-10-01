"""Event-level out-of-order delivery on frozen historical access calendars.

This is retrospective calendar-transfer sensitivity, not an online random-delay
planner. Random delivery contracts are simulated on real QUANT observations.
The original computational files and result streams remain read-only.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
for key, value in {'NUMBA_CACHE_DIR': ROOT/'cache/numba',
                   'MPLCONFIGDIR': ROOT/'temp/mpl', 'TEMP': ROOT/'temp',
                   'TMP': ROOT/'temp'}.items():
    os.environ.setdefault(key, str(value))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import hashlib
import json
from time import perf_counter
import numpy as np
import pandas as pd
from numba import njit
from calibration_engine import _advance, _observe, initialize
from site_data import make_site
from revision_access import CONFIG
import revision_access_controller as original
from statistical_analysis import load_run, build_site_bundle, metric_tables

OUT = ROOT/'results/revision6/random_delivery'
EV = ROOT/'evidence/revision6'
FIXED = ROOT/'results/revision6/fixed_delay'
SITES = ('Manchester', 'London', 'York')
POLICIES = ('periodic', 'original', 'aware_retained')
SEEDS = (610001, 610002, 610003)
DELAYS = np.array([0, 24, 72, 168], dtype=np.int64)
PROBABILITIES = np.array([.2, .2, .35, .25])  # Mean delay exactly 72 hours.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def schedule(query, initmask, delays):
    """CSR release-time index; simultaneous releases have deterministic order."""
    n = len(query)
    authorized = query | initmask
    events = np.flatnonzero(authorized)
    arrivals = events + delays[events]
    in_record = arrivals < n
    events, arrivals = events[in_record], arrivals[in_record]
    order = np.lexsort((events, arrivals))
    events, arrivals = events[order], arrivals[order]
    offsets = np.r_[0, np.cumsum(np.bincount(arrivals, minlength=n))].astype(np.int64)
    return events.astype(np.int64), offsets


@njit(cache=True)
def _replay_oosm(phi, valid, ready, m0, P0, Fs, Qs, rs, y,
                 initmask, query, event_order, release_offsets, max_delay,
                 shared_setup):
    """Insert released labels in event order and replay a bounded history.

    At every t, first predict using labels released strictly before t. Then
    insert all labels released at t. Cached post-event states are conditioned
    on the labels that have arrived so far. An older observation invalidates
    the suffix from that event to t; the suffix is recomputed in event order.
    A max_delay+2 ring preserves the state just before every insertable event.
    No smoothing output is retrospectively substituted into saved predictions.
    """
    nd, n, features = phi.shape
    p = features + 1
    width = max_delay + 2
    means = np.zeros((nd, width, p))
    covs = np.zeros((nd, width, p, p))
    current_m, current_P = m0.copy(), P0.copy()
    received = np.zeros(n, np.bool_)
    success = np.zeros(n, np.bool_)
    pred = np.full((nd, n), np.nan)
    variance = np.full((nd, n), np.nan)
    h = np.ones(p)
    replay_steps = 0
    for t in range(n):
        # Output always precedes every release assigned to this clock hour.
        for d in range(nd):
            if t < ready[d]:
                continue
            _advance(current_m[d], current_P[d], Fs[d], Qs[d])
            if valid[d, t]:
                h[:-1] = phi[d, t]
                pred[d, t] = h @ current_m[d]
                variance[d, t] = h @ current_P[d] @ h + rs[d]
            means[d, t % width] = current_m[d]
            covs[d, t % width] = current_P[d]
        earliest = n
        for k in range(release_offsets[t], release_offsets[t+1]):
            u = event_order[k]
            received[u] = True
            # Unavailable values are only inspected when this event releases.
            success[u] = np.isfinite(y[u])
            if success[u] and (query[u] or (shared_setup and initmask[u])):
                earliest = min(earliest, u)
        if earliest == n:
            continue
        for d in range(nd):
            start = max(earliest, ready[d])
            if start > t:
                continue
            if start == ready[d]:
                m, P = m0[d].copy(), P0[d].copy()
            else:
                m = means[d, (start-1) % width].copy()
                P = covs[d, (start-1) % width].copy()
            for u in range(start, t+1):
                _advance(m, P, Fs[d], Qs[d])
                if received[u] and valid[d, u] and np.isfinite(y[u]) and (
                        query[u] or (shared_setup and initmask[u])):
                    h[:-1] = phi[d, u]
                    _observe(m, P, h, y[u], rs[d])
                means[d, u % width] = m
                covs[d, u % width] = P
                replay_steps += 1
            current_m[d], current_P[d] = m, P
    return pred, variance, success, replay_steps


def replay(models, y, ready, initmask, query, event_delays, shared=True):
    started = perf_counter()
    packed = original._prepare(models, y, ready, initmask)
    delays = np.asarray(event_delays, dtype=np.int64)
    query = np.asarray(query, dtype=bool).copy()
    assert delays.shape == query.shape == packed[-1].shape
    assert np.min(delays) >= 0 and not np.any(query & packed[-1])
    events, offsets = schedule(query, packed[-1], delays)
    values = _replay_oosm(*packed, query, events, offsets, int(delays.max()), bool(shared))
    return dict(pred=values[0], var=values[1], success=values[2], replay_steps=int(values[3]),
                query=query, event_delays=delays, runtime_seconds=perf_counter()-started,
                release_event_indices=events, release_offsets=offsets)


def brute_gaussian(model, y, ready, initmask, query, delays):
    """Independent full-prior Gaussian conditioning, only for small self-tests."""
    n = len(y)
    F, Q = model['F'], model['Q']
    means, covs = [], []
    m, P = model['m'].copy(), model['P'].copy()
    for t in range(ready, n):
        m = F * m
        P = F[:, None] * P * F[None, :] + np.diag(Q)
        means.append(m.copy()); covs.append(P.copy())
    def cross(a, b):
        if a >= b:
            return (F**(a-b))[:, None] * covs[b-ready]
        return cross(b, a).T
    pred = np.full(n, np.nan); var = pred.copy()
    authorized = initmask | query
    for t in range(ready, n):
        if not model['valid'][t]:
            continue
        h = np.r_[model['phi'][t], 1.]
        observed = np.flatnonzero(authorized & model['valid'] & np.isfinite(y) &
                                  (np.arange(n) >= ready) &
                                  (np.arange(n)+delays < t))
        mean = h @ means[t-ready]
        variance = h @ covs[t-ready] @ h + model['r']
        if len(observed):
            hs = {u: np.r_[model['phi'][u], 1.] for u in observed}
            K = np.array([[hs[u] @ cross(u, v) @ hs[v] for v in observed] for u in observed])
            K += np.eye(len(observed)) * model['r']
            c = np.array([h @ cross(t, u) @ hs[u] for u in observed])
            innovations = np.array([y[u] - hs[u] @ means[u-ready] for u in observed])
            mean += c @ np.linalg.solve(K, innovations)
            variance -= c @ np.linalg.solve(K, c)
        pred[t], var[t] = mean, variance
    return pred, var


def self_test():
    rng = np.random.default_rng(20261016)
    n, ready = 620, 96
    X = rng.normal(size=(n, 3)); X[:, 0] += 20
    y = 1.5 * X[:, 0] + rng.normal(size=n)
    model = initialize(X, y, ready, q_scale=.001)
    initmask = np.arange(n) < ready
    initmask[260:272] = True
    query = np.zeros(n, bool)
    query[ready:n:17] = True
    query[~initmask & ((np.arange(n) % 41) < 5)] = True
    query[:ready] = False; query &= ~initmask
    max_parity = 0.
    for delay in (0, 24, 72, 168):
        got = replay([model], y, [ready], initmask, query, np.full(n, delay))
        fixed = original.run_site_fixed([model], y, [ready], initmask, query, delay=delay)
        for key in ('pred', 'var'):
            a, b = got[key][:, ready+delay:], fixed[key][:, ready+delay:]
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-11, equal_nan=True)
            max_parity = max(max_parity, float(np.nanmax(np.abs(a-b))))
    delays = rng.choice(DELAYS, n, p=PROBABILITIES)
    got = replay([model], y, [ready], initmask, query, delays)
    hidden = y.copy(); hidden[~(query | initmask)] += 1000
    changed = replay([model], hidden, [ready], initmask, query, delays)
    for key in ('pred', 'var'):
        np.testing.assert_allclose(got[key], changed[key], rtol=0, atol=0, equal_nan=True)
    cut = 330
    future = y.copy(); future[cut+1:] += 900
    changed = replay([model], future, [ready], initmask, query, delays)
    for key in ('pred', 'var'):
        np.testing.assert_allclose(got[key][:, :cut+1], changed[key][:, :cut+1],
                                   rtol=0, atol=0, equal_nan=True)
    # A requested event can first affect the hour after its own release.
    u = int(np.flatnonzero(query & (np.arange(n)+delays < n-1))[0])
    altered = y.copy(); altered[u] += 50
    changed = replay([model], altered, [ready], initmask, query, delays)
    release = u + int(delays[u])
    np.testing.assert_allclose(got['pred'][:, :release+1], changed['pred'][:, :release+1],
                               rtol=0, atol=0, equal_nan=True)
    assert np.nanmax(np.abs(got['pred'][:, release+1:] - changed['pred'][:, release+1:])) > .01
    # Independent dense Gaussian solution checks collisions, reverse arrival,
    # missing feature/label values and a nonzero initial cross-covariance.
    nsmall, rsmall = 52, 3
    plain = dict(phi=rng.normal(size=(nsmall, 1)), valid=np.ones(nsmall, bool),
                 m=np.array([.8, -.2]), P=np.array([[.8, .1], [.1, .5]]),
                 F=np.array([.97, .6]), Q=np.array([.02, .3]), r=.4)
    plain['valid'][18] = False
    small_y = rng.normal(size=nsmall); small_y[24] = np.nan
    small_init = np.arange(nsmall) < rsmall; small_init[25:29] = True
    small_query = (np.arange(nsmall) % 3 == 0) & ~small_init
    small_query[:rsmall] = False
    small_delays = rng.choice(np.array([0, 2, 7]), nsmall)
    small_delays[3] = 7; small_delays[6] = 0  # Explicit reverse-order pair.
    actual = replay([plain], small_y, [rsmall], small_init, small_query, small_delays)
    expected = brute_gaussian(plain, small_y, rsmall, small_init, small_query, small_delays)
    for key, b in zip(('pred', 'var'), expected):
        np.testing.assert_allclose(actual[key][0], b, rtol=2e-11, atol=2e-11, equal_nan=True)
    receipt = dict(status='PASS',constant_delay_parity_delays=[0,24,72,168],
        maximum_constant_delay_absolute_error=max_parity,
        hidden_nonrequested_target_invariance=True,future_label_prefix_invariance=True,
        output_before_current_release=True,out_of_order_dense_gaussian_parity=True,
        synthetic_self_tests_only=True,
        statement='Synthetic cases validate the implementation and are not study evidence.')
    write_json(OUT/'self_test.json', receipt)
    print('SELF TEST', json.dumps(receipt), flush=True)
    return receipt


def protocol():
    OUT.mkdir(parents=True, exist_ok=True); EV.mkdir(parents=True, exist_ok=True)
    value = dict(stage='retrospective sixth-revision sensitivity; no new untouched data',
        experiment_type='fixed historical calendar transfer, with adaptive replanning disabled',
        random_contract=dict(delays_hours=DELAYS.tolist(),probabilities=PROBABILITIES.tolist(),
                             expected_delay_hours=72.,seeds=list(SEEDS),
                             paired_design='One potential delivery lag per site/event-hour, shared across calendars; independent site random streams'),
        calendars=dict(random='d72 shared1 periodic/original/aware_retained',
                       mismatch='aware_retained d0/24/72/168 shared1 calendars, actual delivery72',
                       retained=dict(horizon=240,threshold=1.25)),
        ordering='Output t precedes all releases at t; a release at u+d first affects output u+d+1',
        engine='Insert OOSM at its event time, then refilter the max168h suffix using only already released labels; bounded170-state ring',
        own_commissioning='Original initialized models supplied at own ready indices; post-ready shared commissioning events follow the simulated release contract',
        support='Each original score mask intersected with t>=ready+168, identical 18 instruments and59674 instrument-hours',
        limitations=['Delivery contracts are simulated and are not observed field logs.',
            'Transferred adaptive calendars were generated under their original fixed-delay replay; they do not establish causal online decisions under a new delivery law.',
            'No distribution-aware online random-delay planner is claimed.',
            'Seed ranges describe conditional simulation variability, not a confidence interval over deployments.'],
        frozen_code_sha256={name:sha(ROOT/'scripts'/name) for name in (
            'calibration_engine.py','revision_access_controller.py','site_data.py','study_data.py')})
    path=OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text(encoding='utf-8')) == value
    else: write_json(path,value)
    return value


def load_calendar(site, policy, assumed):
    path = FIXED/f'{site}_B48_{policy}_d{assumed}_shared1.npz'
    with np.load(path, allow_pickle=False) as z:
        return path, {key:z[key].copy() for key in (
            'query','initmask','y','times','score_masks','full_score_masks','ready_indices','metadata_json')}


def save_stream(path, fleet, calendar_path, source, z, meta):
    mask = fleet['score_masks'] & (np.arange(len(fleet['y']))[None,:] >= fleet['ready'][:,None]+168)
    np.testing.assert_array_equal(source['score_masks'], mask)
    np.testing.assert_array_equal(source['y'], fleet['y'])
    np.testing.assert_array_equal(source['initmask'], fleet['initialization_mask'])
    np.testing.assert_array_equal(source['ready_indices'], fleet['ready'])
    pred = np.maximum(z['pred'],0.)
    assert np.isfinite(pred[mask]).all()
    authorized = source['query'] | source['initmask']
    event_times = np.arange(len(fleet['y']))
    release_times = np.where(authorized, event_times+z['event_delays'], -1)
    diagnostic = dict(runtime_seconds=z['runtime_seconds'],replayed_state_steps=z['replay_steps'],
        requested_events=int(source['query'].sum()),authorized_events=int(authorized.sum()),
        releases_beyond_record=int((release_times>=len(fleet['y'])).sum()),
        adjacent_release_order_inversions=int((np.diff(z['release_event_indices'])<0).sum()),
        actual_authorized_mean_delay_hours=float(z['event_delays'][authorized].mean()))
    meta.update(calendar_path=calendar_path.relative_to(ROOT).as_posix(),calendar_sha256=sha(calendar_path),
                scientific_type='fixed-calendar transfer; real observations with simulated delivery',
                runner_sha256=sha(__file__))
    np.savez_compressed(path,pred=pred,var=z['var'],query=source['query'],
        initmask=source['initmask'],y=source['y'],times=source['times'],score_masks=mask,
        full_score_masks=source['full_score_masks'],ready_indices=source['ready_indices'],
        metadata_json=source['metadata_json'],event_delay=z['event_delays'],release_time=release_times,
        release_event_indices=z['release_event_indices'],release_offsets=z['release_offsets'],
        revision_provenance_json=json.dumps(meta,sort_keys=True),
        delivery_diagnostics_json=json.dumps(diagnostic,sort_keys=True))
    return diagnostic


def run():
    rows=[]; parity=[]
    for site_index, site in enumerate(SITES):
        fleet=make_site(site,**CONFIG['calibrator'])
        n=len(fleet['y'])
        # Real-data constant-delay replay independently checks the cache engine.
        for policy in POLICIES:
            calendar_path, source=load_calendar(site,policy,72)
            replayed=replay(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'],
                            source['query'],np.full(n,72,dtype=np.int64))
            with np.load(calendar_path,allow_pickle=False) as z:
                for key in ('pred','var'):
                    actual=np.maximum(replayed[key],0.) if key=='pred' else replayed[key]
                    np.testing.assert_allclose(actual[source['score_masks']],z[key][source['score_masks']],
                                               rtol=2e-11,atol=2e-10)
                error=float(np.max(np.abs(np.maximum(replayed['pred'],0.)[source['score_masks']]-z['pred'][source['score_masks']])))
            parity.append(dict(site=site,policy=policy,delay=72,maximum_prediction_error=error,status='PASS'))
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([seed,site_index]))
            delays=rng.choice(DELAYS,n,p=PROBABILITIES)
            for policy in POLICIES:
                method=f'{policy}_random{seed}'
                path=OUT/f'{site}_B48_{method}.npz'
                calendar_path, source=load_calendar(site,policy,72)
                if not path.exists():
                    z=replay(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'],source['query'],delays)
                    diagnostic=save_stream(path,fleet,calendar_path,source,z,
                        dict(site=site,policy=policy,condition='random_delivery',seed=seed,assumed_delay=72))
                else:
                    with np.load(path,allow_pickle=False) as saved:
                        assert json.loads(saved['revision_provenance_json'].item())['runner_sha256']==sha(__file__)
                        np.testing.assert_array_equal(saved['event_delay'],delays)
                        diagnostic=json.loads(saved['delivery_diagnostics_json'].item())
                rows.append(dict(site=site,policy=policy,condition='random_delivery',seed=seed,assumed_delay=72,**diagnostic))
                print('RANDOM',json.dumps(rows[-1]),flush=True)
        for assumed in (0,24,72,168):
            method=f'aware_retained_assumed{assumed}_actual72'
            path=OUT/f'{site}_B48_{method}.npz'
            calendar_path,source=load_calendar(site,'aware_retained',assumed)
            if not path.exists():
                z=replay(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'],source['query'],np.full(n,72,dtype=np.int64))
                diagnostic=save_stream(path,fleet,calendar_path,source,z,
                    dict(site=site,policy='aware_retained',condition='delay_mismatch',assumed_delay=assumed,actual_delay=72))
            else:
                with np.load(path,allow_pickle=False) as saved:
                    assert json.loads(saved['revision_provenance_json'].item())['runner_sha256']==sha(__file__)
                    diagnostic=json.loads(saved['delivery_diagnostics_json'].item())
            rows.append(dict(site=site,policy='aware_retained',condition='delay_mismatch',seed=0,assumed_delay=assumed,**diagnostic))
            print('MISMATCH',json.dumps(rows[-1]),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'delivery_diagnostics.csv',index=False)
    write_json(OUT/'constant_delay_real_parity.json',dict(status='PASS',checks=parity))


def summarize():
    aggregate=[]; instruments=[]
    for seed in SEEDS:
        methods=tuple(f'{policy}_random{seed}' for policy in POLICIES)
        bundles=[]
        for site in SITES:
            runs={method:load_run(OUT/f'{site}_B48_{method}.npz') for method in methods}
            bundles.append(build_site_bundle(runs,methods=methods))
        assert sum(b['active'].sum() for b in bundles)==59674
        tables=metric_tables(bundles)
        for key,destination in (('aggregate_metrics',aggregate),('instrument_metrics',instruments)):
            frame=tables[key].copy();frame['seed']=seed;frame['condition']='random_delivery';frame['assumed_delay']=72
            frame['policy']=frame.method.str.replace(f'_random{seed}','',regex=False)
            destination.append(frame)
    methods=tuple(f'aware_retained_assumed{d}_actual72' for d in (0,24,72,168))+('periodic_actual72',)
    bundles=[]
    for site in SITES:
        runs={m:load_run(OUT/f'{site}_B48_{m}.npz') for m in methods[:-1]}
        runs[methods[-1]]=load_run(FIXED/f'{site}_B48_periodic_d72_shared1.npz')
        bundles.append(build_site_bundle(runs,methods=methods))
    assert sum(b['active'].sum() for b in bundles)==59674
    tables=metric_tables(bundles)
    for key,destination in (('aggregate_metrics',aggregate),('instrument_metrics',instruments)):
        frame=tables[key].copy();frame['seed']=0;frame['condition']='delay_mismatch'
        frame['assumed_delay']=frame.method.str.extract(r'assumed(\d+)',expand=False).fillna('72').astype(int)
        frame['policy']=np.where(frame.method.eq('periodic_actual72'),'periodic','aware_retained')
        destination.append(frame)
    all_aggregate=pd.concat(aggregate,ignore_index=True)
    all_aggregate.to_csv(OUT/'aggregate_metrics.csv',index=False)
    pd.concat(instruments,ignore_index=True).to_csv(OUT/'instrument_metrics.csv',index=False)
    random=all_aggregate[all_aggregate.condition.eq('random_delivery')]
    random.groupby('policy').macro_instrument_rmse.agg(['mean','min','max']).reset_index().to_csv(
        OUT/'random_seed_ranges.csv',index=False)
    comparisons=[]
    for seed,frame in random.groupby('seed'):
        values=frame.set_index('policy').macro_instrument_rmse
        for policy in ('original','aware_retained'):
            comparisons.append(dict(seed=int(seed),policy=policy,comparator='periodic',
                difference=float(values[policy]-values['periodic']),
                description='Conditional point difference under one paired simulated delivery draw; not a CI'))
        comparisons.append(dict(seed=int(seed),policy='aware_retained',comparator='original',
            difference=float(values['aware_retained']-values['original']),description='Conditional point difference; not a CI'))
    pd.DataFrame(comparisons).to_csv(OUT/'random_paired_point_contrasts.csv',index=False)
    summary=dict(status='PASS',random_site_streams=27,mismatch_site_streams=12,
                 instruments=18,instrument_hours=59674,random_seeds=list(SEEDS),
                 all_fixed_calendar_conditioning=True,online_random_planner=False)
    write_json(OUT/'summary.json',summary)
    text=("# Random reference delivery and delay mismatch\n\n"
        "These are retrospective fixed-calendar transfer tests on real QUANT sensor and reference observations. "
        "They do not implement or validate an online distribution-aware planner. The transferred adaptive calendars "
        "were generated under their original fixed-delay contracts and remain fully committed during these replays.\n\n"
        "Per-event delays are 0, 24, 72, or 168 hours with probabilities 0.20, 0.20, 0.35, and 0.25 "
        "(expected 72 hours), using three fixed seeds. A potential delay is drawn for every site/event hour before "
        "applying any calendar, making delivery draws paired across the three calendars. Simulated delivery is not "
        "an observed release log. All outputs precede same-hour releases. The engine inserts out-of-sequence "
        "observations at event time and refilters a maximum 168-hour suffix using only labels already delivered. "
        "A 170-state ring preserves the pre-insertion anchor. Historical outputs are never revised.\n\n"
        "Nine constant-72-hour real-data parity checks match the frozen fixed-delay streams. Synthetic implementation "
        "tests separately establish constant-delay parity at 0/24/72/168 hours, hidden-target and future-label "
        "invariance, output-before-release ordering, and agreement with an independent dense joint-Gaussian "
        "conditioning solution for reverse arrivals and simultaneous releases. Synthetic tests are not empirical results.\n\n"
        "Random-delivery results contain 27 site streams; delay mismatch contains 12. Every comparison uses 18 "
        "physical instruments and 59,674 common instrument-hours after ready+168. Random seed ranges are conditional "
        "simulation variability and must not be described as deployment confidence intervals. Delays assumed by "
        "the retained-aware calendar (0/24/72/168 hours) are separately transferred to actual 72-hour delivery.\n\n"
        "## Aggregate random-delivery RMSE (ppb)\n\n"+random[['seed','policy','macro_instrument_rmse']].to_csv(index=False)+
        "\n## Assumed-delay calendars under actual 72-hour delivery\n\n"+
        all_aggregate[all_aggregate.condition.eq('delay_mismatch')][['policy','assumed_delay','macro_instrument_rmse']].to_csv(index=False))
    (EV/'random_delivery.md').write_text(text,encoding='utf-8')
    print('COMPLETE',json.dumps(summary),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['test','run','summarize','all'],default='all')
    args=ap.parse_args();protocol()
    if args.stage in ('test','all'):self_test()
    if args.stage in ('run','all'):run()
    if args.stage in ('summarize','all'):summarize()


if __name__=='__main__':main()
