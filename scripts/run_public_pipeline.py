"""Recorded fresh-computation phases, separate from sealed-release validation.

Only a new checkout made by public_reproduction.py --fresh is accepted. This
runner does not download data, upload anything, or modify the published archive.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]


def commands(phase):
    legacy=[['run_main.py'],['public_reproduction.py','--stream-baselines'],
            ['run_component_checks.py'],['statistical_analysis.py']]
    legacy += [['revision_access.py','--stage',name] for name in
               ('test','arrival','setup','components','memory','quality')]
    legacy += [['revision_competitive.py','--stage',name] for name in
               ('plan','smoke','cross','development','selected')]
    legacy += [['revision_diagnostics.py','--phase',name] for name in
               ('existing','random','supplemental','quality_pair','matrix',
                'verify_random','random_costs','coverage','operational_contrasts')]
    legacy += [['revision_branch.py'],['revision_environment.py','--phase','freeze'],
               ['revision_environment.py','--phase','all'],['uncertainty_diagnostics.py']]
    r4=[['revision4_delay.py','--stage','test'],['revision4_delay.py','--stage','run'],
        ['revision4_delay_audit.py']]
    r4 += [['revision4_gp_active.py','--stage',name] for name in
           ('smoke','development','run','summarize','verify')]
    r4 += [['revision4_gp_delay.py','--stage',name] for name in ('smoke','run','summarize','verify')]
    r4 += [['revision4_components.py','--stage','all']]
    r6=[['revision6_contract_controller.py','--contract-test'],
        ['revision6_fixed_delay.py','--stage','all'],['revision6_correct_contract.py'],
        ['revision6_model_sensitivity.py'],['revision6_model_sensitivity_audit.py'],
        ['revision6_random_delivery.py','--stage','all'],
        ['revision6_online_delivery.py','--stage','all']]
    r6 += [['revision6_gp_active.py',name] for name in ('smoke','run','summarize','verify')]
    r6 += [['revision6_gp_intervals.py'],['revision6_statistics.py','--stage','all',
           '--matrix-directory','results/revision6/fixed_delay_corrected']]
    return {'legacy':legacy,'revision4':r4,'revision6':r6}[phase]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('legacy','revision4','revision6'),required=True)
    ap.add_argument('--list-only',action='store_true')
    args=ap.parse_args();sequence=commands(args.phase)
    if args.list_only:print(json.dumps(sequence,indent=2));return
    if not (ROOT/'FRESH_COMPUTATION.json').exists() or (ROOT/'REPRODUCTION_MANIFEST.json').exists():
        raise RuntimeError('Use a new computation checkout created by --fresh, not the sealed release.')
    for folder in ('cache/numba','temp/matplotlib','results/pipeline','evidence/revision6'):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    for key,value in {'NUMBA_CACHE_DIR':ROOT/'cache/numba','MPLCONFIGDIR':ROOT/'temp/matplotlib',
                      'TEMP':ROOT/'temp','TMP':ROOT/'temp'}.items():os.environ[key]=str(value)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONUTF8']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
    rows=[];receipt=ROOT/'results/pipeline'/f'{args.phase}.json'
    if receipt.exists():raise FileExistsError('Phase receipt already exists; use a new fresh checkout for a clean rebuild.')
    for index,arguments in enumerate(sequence):
        program=ROOT/'scripts'/arguments[0]
        if not program.exists():raise FileNotFoundError(program)
        print('RUN',args.phase,index+1,'/',len(sequence),' '.join(arguments),flush=True)
        start=datetime.now(timezone.utc).isoformat()
        logfile=ROOT/'results/pipeline'/f'{args.phase}_{index+1:02d}_{program.stem}.log'
        with logfile.open('w',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,str(program),*arguments[1:]],cwd=ROOT,
                                  stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        rows.append(dict(command=arguments,started_utc=start,completed_utc=datetime.now(timezone.utc).isoformat(),
                         returncode=result.returncode,log=logfile.relative_to(ROOT).as_posix()))
        receipt.write_text(json.dumps(dict(phase=args.phase,records=rows,
            status='PASS' if len(rows)==len(sequence) and result.returncode==0 else 'RUNNING_OR_FAILED'),indent=2),encoding='utf-8')
        if result.returncode:raise RuntimeError(f'Computation failed; see {logfile}')
    print('PHASE COMPLETE',args.phase,flush=True)


if __name__=='__main__':main()
