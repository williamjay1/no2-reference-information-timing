"""Read-only statistical augmentation of the retrospective sixth revision.

Writes only results/revision6/statistics and evidence/revision6/statistics.md.
No estimator, schedule, old stream, or manuscript is modified. The principal
contrast family was specified in REVISION6_STATE.md before these computations;
the evaluation data had already been inspected in earlier revisions.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key, value in {'NUMBA_CACHE_DIR': ROOT/'cache/numba',
                   'MPLCONFIGDIR': ROOT/'temp/matplotlib',
                   'TEMP': ROOT/'temp', 'TMP': ROOT/'temp'}.items():
    os.environ.setdefault(key, str(value))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')

import argparse
import hashlib
import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from statistical_analysis import load_run, build_site_bundle, metric_tables
from revision4_components import bootstrap_instruments

OUT = ROOT/'results/revision6/statistics'
DEFAULT_MATRIX = ROOT/'results/revision6/fixed_delay'
MATRIX = DEFAULT_MATRIX
SITES = ('Manchester', 'London', 'York')
DELAYS = (0, 24, 72, 168)
POLICIES = ('periodic', 'original', 'aware_unretuned', 'aware_retained', 'aware_expanded')
SOURCES: set[Path] = set()
SHARED_REUSED = False
SCOPE = ('Conditional paired calendar-block uncertainty for the observed sites, '
         'physical instruments, deployments, support, and fixed policies; '
         'retrospective, not independent confirmation or new-instrument uncertainty.')


def read_csv(path):
    path = Path(path)
    SOURCES.add(path)
    return pd.read_csv(path)


def run_file(path):
    path = Path(path)
    SOURCES.add(path)
    return load_run(path)


def write_csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT/(name+'.csv'), index=False)
    return frame


def centered_p(samples, estimate):
    """Two-sided centered bootstrap tail probability, with plus-one correction.

    This is not a randomization test: it uses the empirical block-bootstrap
    approximation to the null distribution of the centered estimator.
    """
    samples = np.asarray(samples, float)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError('A finite univariate paired bootstrap sample is required')
    return float((1 + np.count_nonzero(np.abs(samples-estimate) >= abs(estimate))) /
                 (len(samples)+1))


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError('Invalid probabilities')
    order = np.argsort(pvalues, kind='stable')
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        pvalues[order]*(len(pvalues)-np.arange(len(pvalues)))))
    return adjusted


def interval_row(estimate, draws):
    low, high = np.quantile(draws, [.025, .975])
    return dict(estimate=float(estimate), ci_low=float(low), ci_high=float(high))


def bootstrap_details(boot):
    return {k: v for k, v in boot['diagnostics'].items() if k != 'uncertainty_scope'}


def primary_bundles():
    # All eleven methods share the same score mask, reference, commissioning,
    # row order and bootstrap calendar weights, including across delays.
    methods = ('d0_periodic', 'd0_original') + tuple(
        f'd{d}_{p}' for d in DELAYS[1:] for p in ('periodic', 'original', 'aware_retained'))
    bundles = []
    for site in SITES:
        runs = {}
        for name in methods:
            prefix, policy = name.split('_', 1)
            path = MATRIX/f'{site}_B48_{policy}_{prefix}_shared1.npz'
            runs[name] = run_file(path)
        bundle = build_site_bundle(runs, methods=methods)
        bundles.append(bundle)
    assert sum(int(b['active'].sum()) for b in bundles) == 59674
    assert len({i for b in bundles for i in b['instruments']}) == 18
    return bundles, methods


def assert_shared_principal_unchanged():
    """A matrix-only permission correction may reuse existing shared inference.

    Compare actual arrays, not formatted result tables. If any principal
    shared stream changes, stop rather than reuse its old inference silently.
    """
    global SHARED_REUSED
    if MATRIX == DEFAULT_MATRIX:
        return
    names = [('periodic', 0), ('original', 0)] + [
        (policy, delay) for delay in DELAYS[1:]
        for policy in ('periodic', 'original', 'aware_retained')]
    fields = ('pred', 'var', 'y', 'times', 'score_masks', 'query', 'initmask', 'metadata_json')
    for site in SITES:
        for policy, delay in names:
            name = f'{site}_B48_{policy}_d{delay}_shared1.npz'
            old, current = DEFAULT_MATRIX/name, MATRIX/name
            SOURCES.update((old, current))
            with np.load(old, allow_pickle=False) as a, np.load(current, allow_pickle=False) as b:
                for key in fields:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f'{name}: shared {key} changed')
    print('SHARED PRINCIPAL REUSE: all 33 streams exactly unchanged', flush=True)
    SHARED_REUSED = True


def principal_family(replicates=5000, sensitivity_replicates=1500):
    bundles, names = primary_bundles()
    family = [(0, 'original', 'periodic')] + [
        (d, 'aware_retained', comparator) for d in DELAYS[1:]
        for comparator in ('original', 'periodic')]
    all_rows = []
    for days, reps in ((7, replicates), (14, sensitivity_replicates)):
        boot = bootstrap_instruments(bundles, reps, days, 20261009)
        assert boot['diagnostics']['accepted_replicates'] == reps
        values = boot['instrument'].mean(axis=2)
        observed = boot['observed_instrument'].mean(axis=1)
        rows = []
        for delay, policy, comparator in family:
            a, b = names.index(f'd{delay}_{policy}'), names.index(f'd{delay}_{comparator}')
            delta = observed[a]-observed[b]
            draws = values[:, a]-values[:, b]
            rows.append(dict(family='principal seven retrospective contrasts',
                delay=delay, shared=True, policy=policy, comparator=comparator,
                **interval_row(delta, draws), policy_rmse=float(observed[a]),
                comparator_rmse=float(observed[b]),
                relative_rmse_difference_percent=float(100*delta/observed[b]),
                p_centered_bootstrap=centered_p(draws, delta),
                interval='Marginal percentile 95%; not simultaneous family coverage',
                **bootstrap_details(boot), uncertainty_scope=SCOPE))
        for row, adjusted in zip(rows, holm([r['p_centered_bootstrap'] for r in rows])):
            row['p_holm_7'] = float(adjusted)
            row['holm_below_0p05'] = bool(adjusted < .05)
            row['analysis_role'] = 'principal' if days == 7 else 'block-length sensitivity'
            row.update(method=row['policy'], difference=row['estimate'],
                       p_centered=row['p_centered_bootstrap'], p_holm=row['p_holm_7'])
        all_rows.extend(rows)
        np.savez_compressed(OUT/f'principal_joint_bootstrap_{days}d.npz',
            methods=np.asarray(names), instruments=np.asarray(boot['instruments']),
            instrument_rmse=boot['instrument'], observed_instrument_rmse=boot['observed_instrument'],
            site_equal_rmse=boot['site'], observed_site_equal_rmse=boot['observed_site'],
            diagnostics_json=json.dumps(boot['diagnostics']))
        if days == 7:
            device_intervals_from_boot(boot, bundles, 'common_ready_plus168',
                'd0_original', 'd0_periodic', 'device_contrasts_common_support')
        print(f'PRINCIPAL {days}d {reps} paired draws complete', flush=True)
    frame = write_csv('principal_contrasts_holm', all_rows)
    write_csv('principal_contrasts_7d', frame[frame.block_days.eq(7)])
    write_csv('principal_contrasts_14d_sensitivity', frame[frame.block_days.eq(14)])
    return frame


def device_intervals_from_boot(boot, bundles, support, policy, comparator, name):
    methods = tuple(boot['methods'])
    a, b = methods.index(policy), methods.index(comparator)
    lookup = {}
    for bundle in bundles:
        for j, identity in enumerate(bundle['instruments']):
            item = lookup.setdefault(identity, dict(sites=set(), family=bundle['families'][j], hours=0))
            item['sites'].add(bundle['site'])
            item['hours'] += int(bundle['active'][j].sum())
    rows = []
    for j, identity in enumerate(boot['instruments']):
        info = lookup[identity]
        delta = boot['observed_instrument'][a, j]-boot['observed_instrument'][b, j]
        draws = boot['instrument'][:, a, j]-boot['instrument'][:, b, j]
        rows.append(dict(instrument=identity, sites='|'.join(sorted(info['sites'])),
            site='|'.join(sorted(info['sites'])), difference=float(delta),
            family=info['family'], instrument_hours=info['hours'], support=support,
            policy=policy, comparator=comparator,
            policy_rmse=float(boot['observed_instrument'][a, j]),
            comparator_rmse=float(boot['observed_instrument'][b, j]),
            **interval_row(delta, draws), **bootstrap_details(boot),
            analysis_role='descriptive device contrast, no independent device-family test',
            interval='Conditional paired percentile confidence interval; NOT a new-device prediction interval',
            uncertainty_scope=SCOPE))
    return write_csv(name, rows)


def existing_device_and_factorial():
    component = ROOT/'results/revision4/components'
    bundles = [build_site_bundle({m: run_file(ROOT/f'results/main/{site}_B48_{m}.npz')
                                for m in ('periodic', 'risk_fixed')},
                                methods=('periodic', 'risk_fixed')) for site in SITES]
    assert sum(int(b['active'].sum()) for b in bundles) == 62652
    all_device = []
    full_contrasts = []
    for days in (7, 14):
        path = component/f'bootstrap_fragility_primary_{days}d.npz'
        SOURCES.add(path)
        with np.load(path, allow_pickle=False) as z:
            boot = dict(methods=tuple(z['methods'].astype(str)),
                instruments=tuple(z['instruments'].astype(str)),
                instrument=z['instrument_rmse'], observed_instrument=z['observed_instrument_rmse'],
                diagnostics=json.loads(z['diagnostics_json'].item()))
        frame = device_intervals_from_boot(boot, bundles, 'original_full_support_62652',
            'risk_fixed', 'periodic', f'device_contrasts_full_support_{days}d')
        all_device.append(frame)
        names = boot['methods']; a, b = names.index('risk_fixed'), names.index('periodic')
        obs = boot['observed_instrument'].mean(axis=1)
        draws = boot['instrument'][:, a].mean(axis=1)-boot['instrument'][:, b].mean(axis=1)
        full_contrasts.append(dict(support='original_full_support', instrument_hours=62652,
            policy='risk_fixed', comparator='periodic', **interval_row(obs[a]-obs[b], draws),
            policy_rmse=float(obs[a]), comparator_rmse=float(obs[b]),
            **bootstrap_details(boot), analysis_role='earlier full-support descriptive contrast',
            uncertainty_scope=SCOPE))
    write_csv('device_contrasts_full_support', pd.concat(all_device, ignore_index=True))
    write_csv('immediate_full_support_contrast', full_contrasts)
    # Reuse existing four-cell experiments and already generated interaction
    # draws. No new factorial replay is needed or performed.
    agg = read_csv(component/'component_aggregate_metrics.csv')
    ci = read_csv(component/'component_paired_intervals.csv')
    write_csv('factorial_existing_cell_metrics', agg)
    interaction = ci[ci.contrast.eq('AP-AW-FP+FW interaction')].copy()
    interaction['analysis_role'] = 'existing retrospective computational interaction'
    interaction['interpretation'] = ('Interaction of RMSE contrasts, not additive attribution '
        'of physical drift or electrode aging; same fixed calendar in all four cells')
    write_csv('factorial_existing_interactions', interaction)
    for scope in agg.scope.unique():
        cells = agg[agg.scope.eq(scope)].set_index('method').macro_instrument_rmse
        estimate = (cells['adaptive_persistent']-cells['adaptive_white']
                    -cells['frozen_persistent']+cells['frozen_white'])
        target = interaction[interaction.scope.eq(scope) & interaction.metric.eq('macro_instrument_rmse')]
        np.testing.assert_allclose(target.estimate, estimate, rtol=1e-12, atol=1e-12)
    (OUT/'factorial_cell_equations.txt').write_text(
        'Shared calibrated observation equation: y_t = phi_t^T beta_t + r_t + epsilon_t; '
        'epsilon_t ~ N(0,R). Identical initial beta/residual means and covariance, R, '
        'scoring masks and realized B48 risk_fixed reference calendar in all cells.\n'
        'AP: beta_t = beta_(t-1)+eta_t, eta_t~N(0,Q_beta); '
        'r_t=rho r_(t-1)+xi_t, xi_t~N(0,(1-rho^2)sigma_r^2); full Kalman mean gain.\n'
        'AW: same adaptive beta; rho=0 and Q_r=sigma_r^2. '
        'Initial residual marginal variance is unchanged.\n'
        'FP: Q_beta=0 and K_beta=0, fixing beta means while retaining their initial '
        'uncertainty; residual remains persistent. '
        'Constrained K=(0,0,0,0,K_r)^T uses P+=(I-Kh^T)P-(I-Kh^T)^T+KRK^T, '
        'preserving positive semidefiniteness and beta marginal covariance.\n'
        'FW: frozen beta as FP; white residual as AW.\n'
        'Interaction = RMSE(AP)-RMSE(AW)-RMSE(FP)+RMSE(FW), in ppb. '
        'Policy calendars are not reoptimized in this controlled computational cross.\n',
        encoding='utf-8')


def matrix_descriptive_and_forcing():
    metrics = read_csv(MATRIX/'matrix_aggregate_metrics.csv')
    metrics['policy'] = metrics['method']
    rows = []
    variance_rows = []
    for shared in (True, False):
        for delay in DELAYS:
            for policy in POLICIES:
                for site in SITES:
                    path = MATRIX/f'{site}_B48_{policy}_d{delay}_shared{int(shared)}.npz'
                    run = run_file(path)
                    with np.load(path, allow_pickle=False) as z:
                        var = z['var'].copy()
                        finite_scores = z['candidate_scores'][np.isfinite(z['candidate_scores'])] if 'candidate_scores' in z else np.array([])
                        forced = z['forced_query'].astype(bool) if 'forced_query' in z else None
                        starts = z['start_mask'].astype(bool) if 'start_mask' in z else None
                    paid = int(run.query.sum())
                    forced_n = int((forced & run.query).sum()) if forced is not None else None
                    zero = bool(len(finite_scores) and np.all(finite_scores == 0))
                    rows.append(dict(site=site, delay=delay, shared=shared, policy=policy,
                        paid_attempts=paid, forced_paid_attempts=forced_n,
                        forced_paid_fraction=forced_n/paid if forced_n is not None and paid else np.nan,
                        starts=int(starts.sum()) if starts is not None else None,
                        candidate_score_evaluations=len(finite_scores),
                        positive_candidate_score_evaluations=int((finite_scores > 0).sum()),
                        all_finite_candidate_scores_zero=zero,
                        candidate_score_max=float(finite_scores.max()) if len(finite_scores) else np.nan,
                        informative_acquisition=(False if zero else True) if policy != 'periodic' else None,
                        interpretation=('Zero rollout benefit and quota fallback: not evidence of successful value-based acquisition'
                            if zero else 'Not applicable to periodic calendar' if policy == 'periodic'
                            else 'Mixture of positive-score decisions and quota fallback; fraction is attempt-weighted')))
                    for identity in sorted({r['instrument'] for r in run.metadata}):
                        ids = np.array([k for k, r in enumerate(run.metadata) if r['instrument'] == identity])
                        mask = run.score_masks[ids]
                        count = mask.sum(axis=0)
                        active = count > 0
                        den = np.maximum(count, 1)
                        se = np.where(mask, (run.pred[ids]-run.y[None, :])**2, 0).sum(axis=0)/den
                        pv = np.where(mask, var[ids], 0).sum(axis=0)/den
                        assert np.isfinite(se[active]).all() and np.isfinite(pv[active]).all()
                        assert (pv[active] > 0).all()
                        rho = float(spearmanr(pv[active], se[active]).statistic)
                        variance_rows.append(dict(site=site, instrument=identity,
                            family=run.metadata[ids[0]]['family'], delay=delay, shared=shared, policy=policy,
                            instrument_hours=int(active.sum()), mean_predictive_observation_variance=float(pv[active].mean()),
                            observed_clipped_prediction_mse=float(se[active].mean()),
                            observed_mse_to_mean_variance=float(se[active].mean()/pv[active].mean()),
                            variance_squared_error_spearman=rho,
                            analysis_role='descriptive observed-device diagnostic, no IID-hour correlation p-value',
                            variance_scope='pre-update h^T P h + R; squared error uses reported zero-clipped prediction'))
    forcing = write_csv('matrix_forcing_site', rows)
    agg_forcing = []
    for keys, f in forcing.groupby(['shared', 'delay', 'policy'], sort=False):
        applicable = f.forced_paid_attempts.notna().all()
        total = int(f.paid_attempts.sum())
        forced = int(f.forced_paid_attempts.sum()) if applicable else None
        agg_forcing.append(dict(shared=keys[0], delay=keys[1], policy=keys[2],
            paid_attempts=total, forced_paid_attempts=forced,
            forced_paid_fraction=forced/total if applicable and total else np.nan,
            all_sites_all_finite_candidate_scores_zero=bool(f.all_finite_candidate_scores_zero.all()),
            analysis_role='attempt-weighted engineering diagnostic, not a superiority test'))
    fa = write_csv('matrix_forcing_aggregate', agg_forcing)
    combined = metrics.merge(fa, on=['shared', 'delay', 'policy'], how='left', validate='one_to_one')
    write_csv('matrix_all_cells_descriptive', combined)
    write_csv('matrix_metrics', combined)
    variance = write_csv('variance_error_common_support_devices', variance_rows)
    aggregate = variance.groupby(['shared', 'delay', 'policy'], as_index=False)[[
        'mean_predictive_observation_variance', 'observed_clipped_prediction_mse',
        'observed_mse_to_mean_variance', 'variance_squared_error_spearman']].mean()
    aggregate['aggregation'] = 'unweighted mean of 18 observed-device summaries; ratio is mean of device ratios'
    aggregate['analysis_role'] = 'descriptive, no hour-IID significance claim'
    write_csv('variance_error_common_support_aggregate', aggregate)
    legacy = read_csv(ROOT/'results/uncertainty_diagnostics.csv')
    legacy['variance_scope'] = 'Original full-support unclipped pre-update Gaussian prediction diagnostics'
    legacy['analysis_role'] = 'Previously computed descriptive result, reused without rerun or tuning'
    write_csv('variance_error_legacy_full_support', legacy)
    zeros = forcing[forcing.policy.eq('aware_unretuned') & forcing.delay.eq(168)]
    assert len(zeros) == 6 and zeros.all_finite_candidate_scores_zero.all()
    assert zeros.forced_paid_fraction.eq(1.).all()
    print('MATRIX: 120 site forcing rows, 720 device variance rows; H168/d168 degenerate score verified', flush=True)


def actual_context_benefits(phi, valid, y, P0, F, Q, R, t, score, horizon=168):
    """Offline oracle inputs/availability, not oracle label values or a policy.

    Like the old local branch, predict before assimilation at every hour.
    Return both fixed-horizon and exact-common-score means. Missing input or
    target prevents assimilation. Future feature/availability access is only
    for this explicitly offline diagnostic, never historical acquisition.
    """
    covs = np.stack([P0.copy() for _ in range(3)])
    gains = np.zeros((2, horizon))
    for j in range(horizon+1):
        if j > 0:
            covs = covs*F[None, :, None]*F[None, None, :]+np.diag(Q)[None, :, :]
        k = t+j
        h = np.ones(len(F)); h[:-1] = phi[k] if valid[k] else phi[t]
        if j > 0:
            values = np.einsum('i,aij,j->a', h, covs, h)
            gains[:, j-1] = np.maximum(values[2]-values[:2], 0.)
        if valid[k] and np.isfinite(y[k]):
            for a in range(2):
                start = a*72
                if start <= j < start+12:
                    ph = covs[a]@h
                    covs[a] -= np.outer(ph, ph)/max(float(R+h@ph), 1e-12)
    return gains.mean(axis=1), gains[:, score].mean(axis=1)


def local_window_diagnostics():
    branch = ROOT/'results/revision/branch'
    points = read_csv(branch/'all_points.csv')
    summary = read_csv(branch/'summary.csv')
    write_csv('local_window_existing_summary', summary)
    from site_data import make_site
    from revision_branch import snapshot_history
    freeze_path = ROOT/'evidence/FREEZE.json'
    SOURCES.add(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    SOURCES.update([ROOT/'datasets/no2_hourly_oob.parquet',
                    ROOT/'datasets/reference_no2_ratified.parquet'])
    SOURCES.update(ROOT/'scripts'/name for name in ('site_data.py', 'study_data.py',
        'calibration_engine.py', 'acquisition_controller.py', 'revision_branch.py'))
    rows = []
    for site in SITES:
        fleet = make_site(site, **freeze['calibrator'])
        periodic = run_file(ROOT/f'results/main/{site}_B48_periodic.npz')
        access = periodic.query | fleet['initialization_mask']
        for d, (model, meta) in enumerate(zip(fleet['models'], fleet['metadata'])):
            group = points[points.site.eq(site) & points.deployment.eq(meta['deployment'])]
            if group.empty:
                continue
            indices = group['index'].to_numpy(np.int64)
            means, covs, pred = snapshot_history(model['phi'], model['valid'], fleet['y'], access,
                model['m'], model['P'], model['F'], model['Q'], model['r'], int(fleet['ready'][d]), indices)
            np.testing.assert_allclose(pred, periodic.pred[d], rtol=0, atol=0, equal_nan=True)
            for k, (_, record) in enumerate(group.iterrows()):
                t = int(record['index'])
                score = fleet['score_masks'][d, t+1:t+169]
                horizon, common = actual_context_benefits(model['phi'], model['valid'], fleet['y'],
                    covs[k], model['F'], model['Q'], model['r'], t, score)
                if k == 0:
                    changed = fleet['y'].copy()
                    finite = np.isfinite(changed[t:])
                    tail = changed[t:]
                    tail[finite] += 1000.
                    test_horizon, test_common = actual_context_benefits(model['phi'], model['valid'],
                        changed, covs[k], model['F'], model['Q'], model['r'], t, score)
                    np.testing.assert_array_equal(test_horizon, horizon)
                    np.testing.assert_array_equal(test_common, common)
                row = record.to_dict()
                row.update(oracle_context_variance_difference_horizon=float(horizon[0]-horizon[1]),
                    oracle_context_variance_difference_scored=float(common[0]-common[1]),
                    oracle_context_benefit_now_scored=float(common[0]),
                    oracle_context_benefit_late_scored=float(common[1]),
                    candidate_quota='Equal 12 attempts in both local branches; no fleet quota coupling in diagnostic',
                    oracle_scope='Future actual inputs and label availability only in offline covariance rollout; no future target values in score',
                    analysis_role='retrospective descriptive diagnostic, not retuned acquisition')
                rows.append(row)
    extended = write_csv('local_window_context_diagnostics', rows)
    assert len(extended) == len(points) == 208
    comparisons = []
    columns = ('predicted_difference', 'oracle_context_variance_difference_horizon',
               'oracle_context_variance_difference_scored')
    for name, group in [('pooled', extended)] + list(extended.groupby('instrument')):
        for column in columns:
            predicted = group[column].to_numpy(float)
            realized = group.realized_mse_difference.to_numpy(float)
            non_ties = (predicted != 0) & (realized != 0)
            rho = float(spearmanr(predicted, realized).statistic)
            choice = np.where(predicted >= 0, group.mse_now, group.mse_late)
            oracle = np.minimum(group.mse_now, group.mse_late)
            comparisons.append(dict(instrument=name, criterion=column, n=len(group),
                variance_value_realized_mse_spearman=rho,
                sign_agreement=float((np.sign(predicted[non_ties]) == np.sign(realized[non_ties])).mean()),
                non_ties=int(non_ties.sum()), predicted_now_fraction=float((predicted >= 0).mean()),
                mean_chosen_mse=float(choice.mean()), mean_always_now_mse=float(group.mse_now.mean()),
                mean_best_realized_branch_mse=float(oracle.mean()),
                mean_choice_regret=float((choice-oracle).mean()),
                uncertainty='Dependent weekly cases, four observed long-record devices; descriptive, no IID p-value',
                inference='Input/availability oracle isolates one diagnostic mismatch; not a deployable policy or proof all acquisition objectives fail'))
    result = write_csv('local_window_context_summary', comparisons)
    original = result[result.instrument.eq('pooled') & result.criterion.eq('predicted_difference')].iloc[0]
    np.testing.assert_allclose(original.variance_value_realized_mse_spearman,
        summary[summary.instrument.eq('pooled')].spearman_mse.iloc[0], rtol=1e-12)
    print('LOCAL: 208 old cases reproduced; context/availability-only oracle diagnostic complete', flush=True)


def exploratory_prior_model_contrasts():
    # Existing intervals are reused and explicitly tagged as exploratory.
    files = [ROOT/'results/revision/diagnostics/matrix_shared_selected_paired_intervals.csv',
             ROOT/'results/revision4/gp_active/paired_contrasts.csv',
             ROOT/'results/revision4/gp_delay/paired_contrasts.csv',
             ROOT/'results/revision6/gp_active/paired_contrasts.csv']
    frames = []
    for path in files:
        if path.exists():
            frame = read_csv(path)
            frame['source_table'] = path.relative_to(ROOT).as_posix()
            frame['analysis_role'] = 'Earlier retrospective exploratory baseline contrast; outside principal Holm family'
            frame['inference_boundary'] = 'No new confirmation claim or unadjusted significance conclusion'
            frames.append(frame)
    if frames:
        write_csv('prior_model_exploratory_contrasts', pd.concat(frames, ignore_index=True))
    sensitivity = read_csv(ROOT/'results/revision6/model_sensitivity/aggregate_metrics.csv')
    adverse = sensitivity[sensitivity.variant.eq('drift_high') & sensitivity.delay.eq(72)]
    rmse = adverse.set_index('method').macro_instrument_rmse
    aware = float(rmse['drift_high_aware_retained_d72'])
    periodic = float(rmse['drift_high_periodic_d72'])
    assert aware > periodic
    write_csv('preserved_adverse_model_boundary', [dict(variant='drift_high', delay=72,
        policy='aware_retained', comparator='periodic', policy_rmse=aware,
        comparator_rmse=periodic, estimate=aware-periodic,
        analysis_role='prespecified model sensitivity, adverse result retained',
        interpretation='Delivery-aware scheduling advantage is conditional on process-model assumptions')])


def report():
    SOURCES.update(ROOT/'scripts'/name for name in ('statistical_analysis.py',
                                                  'revision4_components.py'))
    SOURCES.add(ROOT/'REVISION6_STATE.md')
    principal = read_csv(OUT/'principal_contrasts_7d.csv')
    local = read_csv(OUT/'local_window_context_summary.csv')
    full = read_csv(OUT/'immediate_full_support_contrast.csv').query('block_days == 7').iloc[0]
    common = principal.query('delay == 0').iloc[0]
    devices = read_csv(OUT/'device_contrasts_full_support_7d.csv')
    provenance = []
    # Inputs produced by this script are omitted from external source ledger.
    for path in sorted(p for p in SOURCES if OUT not in p.parents):
        provenance.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    payload = dict(stage='retrospective scientific revision, no GitHub upload',
        status='PASS', principal_family_count=7, source_files=provenance,
        software_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matrix_directory=MATRIX.relative_to(ROOT).as_posix(),
        shared_principal_reused_after_permission_fix=SHARED_REUSED,
        preserved_support=dict(common_instrument_hours=59674, full_instrument_hours=62652, instruments=18),
        primary_pvalue_formula='(1 + count(abs(delta_star - delta_hat) >= abs(delta_hat))) / (B + 1)',
        adjustment='Holm step-down across seven specified principal contrasts; marginal percentile CIs',
        inference_scope=SCOPE, no_new_training_experiments=True,
        existing_commissioning_reconstructed_for_offline_covariance=True, old_outputs_modified=False,
        checks=dict(primary_family_seven=True, common_support_equal=True,
            old_factorial_interaction_exact=True, old_window_correlation_reproduced=True,
            local_covariance_score_future_target_value_invariance=True,
            H168_delay168_zero_scores_and_all_forced=True, adverse_drift_high72_preserved=True))
    (OUT/'verification.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    lines = ['# Revision 6 statistical evidence', '',
        'This analysis is retrospective. The later-period records were examined in previous revisions. '
        'The expanded grid and principal contrast family were fixed before these new computations, '
        'but this does not create independent confirmation.', '',
        '## Existing evidence and necessary additions', '',
        '- Common support: earlier delayed comparisons already used ready+168 and 59,674 instrument-hours; '
        'the new complete matrix extends that exact support to every policy, delay, and sharing condition. '
        f'The original immediate full-support contrast is {full.estimate:.6f} ppb on 62,652 hours; '
        f'the matched-support contrast is {common.estimate:.6f} ppb on 59,674 hours.',
        '- The complete adaptation-by-residual 2x2 and seven-/fourteen-day interaction intervals already '
        'exist. They are reused, not duplicated. Equations and constraint implementation are documented '
        'in factorial_cell_equations.txt. White residuals preserve stationary marginal variance; '
        'frozen means retain coefficient uncertainty through a constrained Joseph update.',
        '- The principal family newly receives centered paired bootstrap p-values and Holm adjustment '
        'across seven contrasts. Site-quarter circular blocks preserve observed quarter lengths. '
        'All eleven required policy streams share each bootstrap draw. Intervals are marginal '
        'percentile intervals, not simultaneous family intervals. No hour-IID test is used.',
        '- Earlier full-support draws already contain all 18 device RMSEs. Device confidence intervals '
        'are extracted from those saved paired draws, without repeating training or claiming '
        'prediction intervals for a new device. London and York each remain confounded with their '
        'single physical instrument.',
        '- Existing Gaussian coverage/variance-error diagnostics and the 208-case local window '
        'diagnostic are reused. New matched-support diagnostics and an explicitly offline actual-input '
        'and availability covariance rollout help distinguish forecasting mismatch from quota coupling. '
        'No future target values enter the oracle covariance score; future features and availability '
        'are used only in this offline diagnosis.', '', '## Principal seven-day results', '',
        '| Delay (h) | Contrast | RMSE difference (ppb) | Marginal 95% CI | Centered p | Holm p |',
        '| --- | --- | ---: | --- | ---: | ---: |']
    for r in principal.itertuples():
        lines.append(f'| {r.delay} | {r.policy} minus {r.comparator} | {r.estimate:.6f} | '
            f'[{r.ci_low:.6f}, {r.ci_high:.6f}] | {r.p_centered_bootstrap:.6f} | {r.p_holm_7:.6f} |')
    lines.extend(['', 'The p-values approximate centered sampling distributions conditional on this '
        'observed cohort and fixed policies. They do not eliminate retrospective design decisions or '
        'establish performance in a new deployment. Fourteen-day sensitivity changes which marginal '
        'thresholds cross 0.05: notably retained minus original at 168 h is 0.039992 after Holm with '
        'seven-day blocks, but 0.063957 with fourteen-day blocks. Do not choose a block length '
        'based on this threshold.', '', '## Device intervals', '',
        f'{len(devices)} conditional device contrasts; '
        f'{int((devices.ci_high < 0).sum())} intervals entirely below zero, '
        f'{int((devices.ci_low > 0).sum())} entirely above, '
        f'{int(((devices.ci_low <= 0) & (devices.ci_high >= 0)).sum())} spanning zero. '
        'These are descriptive marginal intervals and are not 18 independent hypothesis tests.', '',
        '## Local window-ranking diagnosis', '',
        '| Offline criterion | Spearman vs realized MSE difference | Sign agreement | Mean chosen MSE |',
        '| --- | ---: | ---: | ---: |'])
    for r in local[local.instrument.eq('pooled')].itertuples():
        lines.append(f'| {r.criterion} | {r.variance_value_realized_mse_spearman:.6f} | '
                     f'{r.sign_agreement:.6f} | {r.mean_chosen_mse:.6f} |')
    lines.extend(['', 'Both local branches spend exactly 12 attempted reference hours. Failure of '
        'their value ranking therefore cannot be attributed solely to the fleet quota constraint. '
        'The input/availability oracle is a diagnosis, not a deployable selector; remaining mismatch '
        'can reflect mean-model error, noise realizations, clipping, and conditional covariance '
        'assumptions. No universal failure claim about information gain or acquisition is warranted.',
        '', '## Preserved adverse boundaries', '',
        '- Unretuned H168 at 168 h delay has a first affected output beyond the horizon. All finite '
        'candidate scores are zero and all paid attempts are quota-forced at all three sites, '
        'with and without sharing. Its RMSE is not evidence of successful value-based scheduling.',
        '- High coefficient process noise at 72 h delay gives retained delivery-aware RMSE '
        'greater than periodic. This adverse prespecified sensitivity remains in the evidence.', '',
        '## Output interfaces', '',
        'CSV files in results/revision6/statistics contain the seven principal contrasts and '
        'fourteen-day sensitivity; all 40 aggregate matrix cells with forced fractions; 120 '
        'site forcing rows; 720 common-support device variance-error rows; 18 full-support '
        'device intervals at each old block length; exact common-support device intervals; '
        'existing component interactions; and all 208 augmented local diagnostic cases. '
        'verification.json hashes every external input and records the statistical scope.'])
    path = ROOT/'evidence/revision6/statistics.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('STATISTICS PASS; seven principal contrasts and scientific boundaries exported', flush=True)


def main():
    global MATRIX, SHARED_REUSED
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('all', 'principal', 'existing', 'matrix', 'local', 'report'), default='all')
    parser.add_argument('--matrix-directory', type=Path, default=DEFAULT_MATRIX,
                        help='Fixed-delay matrix input; matrix-only updates reuse unchanged shared inference')
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--sensitivity-replicates', type=int, default=1500)
    args = parser.parse_args()
    MATRIX = args.matrix_directory.resolve()
    if ROOT not in MATRIX.parents:
        raise ValueError('Matrix inputs must remain inside the D-drive project')
    if not MATRIX.is_dir():
        raise ValueError(f'Matrix input directory does not exist: {MATRIX}')
    if args.replicates < 1000 or args.sensitivity_replicates < 300:
        raise ValueError('Insufficient requested bootstrap precision')
    if shutil.disk_usage(ROOT).free < 100_000_000:
        raise RuntimeError('Insufficient space on D computation drive')
    OUT.mkdir(parents=True, exist_ok=True)
    # Partial updates retain the ledger of unchanged analyses. Superseded
    # matrix inputs are removed except the 33 shared streams whose old draws
    # are explicitly checked and reused after a permissions-only correction.
    receipt_path = OUT/'verification.json'
    if args.stage != 'all' and receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding='utf-8'))
        if args.stage == 'report':
            SHARED_REUSED = bool(previous.get('shared_principal_reused_after_permission_fix', False))
        primary_names = {f'{site}_B48_{policy}_d{delay}_shared1.npz'
            for site in SITES for policy, delay in ([('periodic', 0), ('original', 0)] + [
                (p, d) for d in DELAYS[1:] for p in ('periodic', 'original', 'aware_retained')])}
        for source in previous['source_files']:
            path = ROOT/source['path']
            if (MATRIX != DEFAULT_MATRIX and path.parent == DEFAULT_MATRIX
                    and path.name not in primary_names):
                continue
            SOURCES.add(path)
    if args.stage in ('all', 'existing'):
        existing_device_and_factorial()
    if args.stage in ('all', 'principal'):
        principal_family(args.replicates, args.sensitivity_replicates)
    if args.stage in ('all', 'matrix'):
        if args.stage == 'matrix':
            assert_shared_principal_unchanged()
        matrix_descriptive_and_forcing()
        exploratory_prior_model_contrasts()
    if args.stage in ('all', 'local'):
        local_window_diagnostics()
    if args.stage in ('all', 'matrix', 'report'):
        report()


if __name__ == '__main__':
    main()
