"""Verify an accompanying archive and rebuild errors from hourly predictions.

Run inside the supplied project after extracting it to a computation drive.
Never writes into the original prediction or raw-data directories.
"""
from pathlib import Path
import argparse,hashlib,json
from collections import defaultdict
import numpy as np
import pandas as pd
from statistical_analysis import load_run,build_site_bundle,metric_tables

ROOT=Path(__file__).resolve().parents[1]


def main(output,verify_manifest):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if verify_manifest:
        manifest=json.loads((ROOT/'REPRODUCTION_MANIFEST.json').read_text(encoding='utf-8'))
        for row in manifest['files']:
            p=ROOT/row['path']
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Checksum mismatch: '+row['path'])
        print('All supplied file checksums agree.')
    directories=['results/main','results/revision/access_v2',
        'results/revision/competitive/frozen','results/revision/competitive/shared_selected',
        'results/revision/diagnostics/random_predictions']
    grouped=defaultdict(list);archives=0;failed=[]
    for folder in directories:
        for path in sorted((ROOT/folder).rglob('*.npz')):
            with np.load(path,allow_pickle=False) as z:
                if not {'pred','score_masks','metadata_json','query','initmask'}.issubset(z.files):continue
            run=load_run(path)
            grouped[(folder,run.budget,run.method)].append(build_site_bundle({run.method:run}))
            archives+=1
    aggregates=[];instruments=[]
    for (folder,budget,method),bundles in sorted(grouped.items()):
        if {b['site'] for b in bundles}!={'Manchester','London','York'}:
            failed.append(dict(folder=folder,budget=budget,method=method,sites=[b['site'] for b in bundles]));continue
        tables=metric_tables(bundles)
        a=tables['aggregate_metrics'];a['result_group']=folder;aggregates.append(a)
        i=tables['instrument_metrics'];i['result_group']=folder;instruments.append(i)
    if failed:raise ValueError('Incomplete three-site prediction group: '+json.dumps(failed))
    pd.concat(aggregates,ignore_index=True).to_csv(output/'rebuilt_aggregate_errors.csv',index=False)
    pd.concat(instruments,ignore_index=True).to_csv(output/'rebuilt_instrument_errors.csv',index=False)
    (output/'validation.json').write_text(json.dumps(dict(status='PASS',prediction_archives=archives,complete_groups=len(grouped),note='Different groups may use different explicit scoring support. Direct paired contrasts require their documented common masks.'),indent=2),encoding='utf-8')
    print('Validated',archives,'hourly archives and rebuilt',len(grouped),'complete groups.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/reproduction_check'));p.add_argument('--verify-manifest',action='store_true')
    args=p.parse_args();main(args.output,args.verify_manifest)
