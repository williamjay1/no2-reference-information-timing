"""Portable public validation without modifying frozen scientific programs."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def manifest_check():
    file=ROOT/'REPRODUCTION_MANIFEST.json'
    if not file.exists():raise FileNotFoundError('Use the complete reproduction release archive, which includes its byte manifest.')
    manifest=json.loads(file.read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=(ROOT/row['path']).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(),row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    return dict(status='PASS',files=len(manifest['files']))


def environment(output):
    for name in ('cache/numba','temp/matplotlib'):(output/name).mkdir(parents=True,exist_ok=True)
    values={'NUMBA_CACHE_DIR':output/'cache/numba','MPLCONFIGDIR':output/'temp/matplotlib',
            'TEMP':output/'temp','TMP':output/'temp'}
    for key,value in values.items():os.environ[key]=str(value)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'


def aggregate(paths):
    """Independently compute instrument-hour averaged squared residuals."""
    import numpy as np
    instruments=[];support=0
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            pred,y,mask=z['pred'],z['y'],z['score_masks'].astype(bool)
            meta=json.loads(z['metadata_json'].item())
            assert pred.shape==mask.shape==(len(meta),len(y))
            assert np.isfinite(pred[mask]).all()
            for identity in sorted({row['instrument'] for row in meta}):
                ix=[i for i,row in enumerate(meta) if row['instrument']==identity]
                count=mask[ix].sum(axis=0);active=count>0
                residual=np.where(mask[ix],pred[ix]-y,0.)
                mse=np.square(residual).sum(axis=0)/np.maximum(count,1)
                instruments.append(float(np.sqrt(mse[active].mean())))
                support+=int(active.sum())
    return dict(instruments=len(instruments),instrument_hours=support,rmse=float(np.mean(instruments)))


def key_numbers():
    import numpy as np
    import pandas as pd
    checks=[]
    directory=ROOT/'results/revision6/fixed_delay_corrected'
    table=pd.read_csv(directory/'matrix_aggregate_metrics.csv')
    for delay in (0,24,72,168):
        for policy in ('periodic','original','aware_retained'):
            paths=[directory/f'{site}_B48_{policy}_d{delay}_shared1.npz' for site in ('Manchester','London','York')]
            actual=aggregate(paths)
            expected=table[(table['delay']==delay)&table['shared'].eq(True)&table.method.eq(policy)]
            assert len(expected)==1
            np.testing.assert_allclose(actual['rmse'],expected.macro_instrument_rmse.iloc[0],rtol=1e-12,atol=1e-12)
            assert actual['instruments']==18 and actual['instrument_hours']==59674
            checks.append(dict(condition='fixed_contract',delay=delay,policy=policy,**actual))
    directory=ROOT/'results/revision6/online_delivery_corrected'
    table=pd.read_csv(directory/'aggregate_metrics.csv')
    for seed in (610001,610002,610003):
        for policy in ('periodic','original_online','aware_online'):
            method=f'{policy}_random{seed}'
            actual=aggregate([directory/f'{site}_B48_{method}.npz' for site in ('Manchester','London','York')])
            expected=table[table.method.eq(method)];assert len(expected)==1
            np.testing.assert_allclose(actual['rmse'],expected.macro_instrument_rmse.iloc[0],rtol=1e-12,atol=1e-12)
            assert actual['instruments']==18 and actual['instrument_hours']==59674
            checks.append(dict(condition='online_random_corrected',seed=seed,policy=policy,**actual))
    return checks


def quick(output):
    # Keep every synthetic write in a disposable copy, including frozen
    # functions that derive their own output root from __file__.
    sandbox=output/'synthetic_sandbox';sandbox.mkdir()
    (sandbox/'scripts').mkdir();(sandbox/'evidence/revision6').mkdir(parents=True)
    for path in (ROOT/'scripts').glob('*.py'):shutil.copy2(path,sandbox/'scripts'/path.name)
    shutil.copy2(ROOT/'evidence/FREEZE.json',sandbox/'evidence/FREEZE.json')
    for relative in ('temp','cache/numba','results'):(sandbox/relative).mkdir(parents=True,exist_ok=True)
    commands=[['revision6_random_delivery.py','--stage','test'],
              ['revision6_online_delivery.py','--stage','test'],
              ['revision6_contract_controller.py','--contract-test']]
    env=os.environ.copy()
    env.update(NUMBA_CACHE_DIR=str(sandbox/'cache/numba'),TEMP=str(sandbox/'temp'),TMP=str(sandbox/'temp'))
    records=[]
    for arguments in commands:
        done=subprocess.run([sys.executable,str(sandbox/'scripts'/arguments[0]),*arguments[1:]],
                            cwd=sandbox,env=env,text=True,capture_output=True)
        (output/(arguments[0]+'.stdout.txt')).write_text(done.stdout+done.stderr,encoding='utf-8')
        if done.returncode:raise RuntimeError(f'Synthetic verification failed: {arguments[0]}: {done.stderr[-2000:]}')
        records.append(dict(program=arguments[0],status='PASS'))
    numbers=key_numbers()
    result=dict(status='PASS',synthetic_tests=records,numerical_groups_recomputed=len(numbers),
        numerical_groups=numbers,scope='Byte manifest, three actual synthetic reruns, and independent aggregation of 21 empirical stream groups. No full policy refitting or field validation is claimed.')
    write_json(output/'quick_validation.json',result);print(json.dumps({k:result[k] for k in ('status','synthetic_tests','numerical_groups_recomputed')},indent=2))


def from_derived(output):
    # A portable counterpart of the legacy D-only CLI, using the unchanged
    # frozen computational functions and data; no guard bypass or source edit.
    import numpy as np
    from site_data import make_site
    from fixed_schedules import fixed_site_mask
    from acquisition_controller import run_site_fixed
    config=json.loads((ROOT/'evidence/FREEZE.json').read_text(encoding='utf-8'))
    checks=[]
    for site in config['sites']:
        fleet=make_site(site,**config['calibrator'])
        for budget in config['budgets']:
            query,_=fixed_site_mask(len(fleet['y']),int(min(fleet['ready'])),fleet['initialization_mask'],
                budget=budget,phase=config['periodic_phase'])
            z=run_site_fixed(fleet['models'],fleet['y'],fleet['ready'],fleet['initialization_mask'],query)
            pred=np.maximum(z['pred'],0.)
            with np.load(ROOT/'results/main'/f'{site}_B{budget}_periodic.npz',allow_pickle=False) as saved:
                np.testing.assert_array_equal(query,saved['query'])
                np.testing.assert_array_equal(fleet['score_masks'],saved['score_masks'])
                np.testing.assert_array_equal(pred,saved['pred'])
            checks.append(dict(site=site,budget=budget,status='PASS',prediction_parity='EXACT',calendar_parity='EXACT'))
            print('FROM DERIVED PASS',site,budget,flush=True)
    write_json(output/'from_derived_validation.json',dict(status='PASS',checks=checks,
        note='Nine periodic streams reconstructed from public-data derivatives. Saved predictions entered only the final equality checks. Other policy families are separate reproducibility commands.'))


def stream_baselines():
    """Call unchanged computational functions in a fresh, portable checkout."""
    from run_main_baselines import _identity,_run_group
    from site_data import make_site
    directory=ROOT/'results/main'
    identity=_identity(ROOT/'evidence/FREEZE.json')
    environment(ROOT)
    for site in ('Manchester','London','York'):
        fleet=make_site(site,stage='test',q_scale=.001,use_ar=True,
                        weather=True,ar_share=.8,guard_width=8.)
        for budget in (12,24,48):
            _run_group(site,budget,'periodic',fleet,identity,directory,False)
        _run_group(site,48,'risk',fleet,identity,directory,False)


def fresh(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Choose a new computation destination.')
    destination.mkdir(parents=True)
    for folder in ('scripts','datasets'):
        shutil.copytree(ROOT/folder,destination/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    (destination/'evidence').mkdir();(destination/'results').mkdir()
    for name in ('FREEZE.json','freeze_technical_addendum.json','REVISION_ACCESS_PROTOCOL.json'):
        path=ROOT/'evidence'/name
        if path.exists():shutil.copy2(path,destination/'evidence'/name)
    for path in (ROOT/'results').glob('*.csv'):
        if 'dev' in path.name:shutil.copy2(path,destination/'results'/path.name)
    for name in ('README.md','requirements_reproduction.txt','LICENSE','DATA_LICENSE.md','CITATION.cff','REVISION6_STATE.md'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,destination/name)
    for folder in ('results/main','results/tables','results/figures','temp','cache/numba','evidence/revision6'):
        (destination/folder).mkdir(parents=True,exist_ok=True)
    write_json(destination/'FRESH_COMPUTATION.json',dict(saved_evaluation_predictions_copied=False,
        originating_published_manifest_sha256=sha(ROOT/'REPRODUCTION_MANIFEST.json')))
    print('Fresh code/data/development-only computation checkout:',destination)


def hydrate(archive,destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Choose a new extraction destination.')
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in z.infolist():assert (destination/item.filename).resolve().is_relative_to(destination)
        z.extractall(destination)
    manifests=list(destination.glob('*/REPRODUCTION_MANIFEST.json'))
    assert len(manifests)==1
    print('Extracted complete reproduction project:',manifests[0].parent)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-manifest',action='store_true');ap.add_argument('--quick',action='store_true')
    ap.add_argument('--from-derived',action='store_true');ap.add_argument('--output',type=Path)
    ap.add_argument('--stream-baselines',action='store_true',help='Portable entry to the unchanged baseline functions; use a fresh computation checkout after run_main.py.')
    ap.add_argument('--fresh',type=Path);ap.add_argument('--hydrate',type=Path);ap.add_argument('--destination',type=Path)
    a=ap.parse_args()
    if a.hydrate:
        if not a.destination:ap.error('--hydrate requires --destination')
        hydrate(a.hydrate,a.destination);return
    if a.fresh:fresh(a.fresh);return
    if a.stream_baselines:stream_baselines();return
    if a.check_manifest or a.quick:print('MANIFEST',json.dumps(manifest_check()),flush=True)
    if a.quick or a.from_derived:
        if not a.output:ap.error('Use an explicit new --output computation directory.')
        output=a.output.resolve()
        if output.exists():raise FileExistsError('Choose a new validation output directory.')
        output.mkdir(parents=True);environment(output)
        if a.quick:quick(output)
        if a.from_derived:from_derived(output)
    elif not a.check_manifest:ap.error('Choose --check-manifest, --quick, --from-derived, --stream-baselines, --fresh or --hydrate.')


if __name__=='__main__':main()
