"""Build local scientific release and code staging; never publish externally.

Curates the original complete computational archive and the final sixth-revision
experiments. Frozen files are copied byte-for-byte. New manifests describe this
curated release rather than falsely asserting the old archive is complete.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'delivery/SensorsWindowCalibration_Information_Timing_Reproduction'
RAW_METADATA=Path('F:/AcademicData/SensorsWindowCalibration/raw/QUANT_v1_zenodo_10775692_metadata_20260930T015809Z.json')
EXCLUDED_PREFIXES=('assemble_','export_','package_','finalize_','verify_delivery')
EXCLUDED_SCRIPTS={'verify_revision4_documents.py','prepare_fresh_replay.py'}
EXTRA_SCRIPTS=('public_reproduction.py','package_revision6_reproduction.py',
    'revision6_contract_controller.py','revision6_correct_contract.py','revision6_fixed_delay.py',
    'revision6_gp_active.py','revision6_gp_intervals.py','revision6_model_sensitivity.py',
    'revision6_model_sensitivity_audit.py','revision6_random_delivery.py',
    'revision6_online_delivery.py','revision6_statistics.py',
    'make_nature_figures.py','verify_nature_figures.py','make_revision6_figures.py',
    'make_graphical_abstract.py','make_revision6_graphical_abstract.py')
BASE_EVIDENCE=('FREEZE.json','freeze_technical_addendum.json','REVISION_ACCESS_PROTOCOL.json',
    'COMPONENT_CHECKS.json','revision2_core_lineage.json','revision3_core_lineage.json',
    'revision4_core_lineage.json','revision4_delay_protocol.json','revision4_protocol.md',
    'revision4_components.md','revision4_gp_active.md','revision4_stream_reaggregation.json',
    'implementation_audit.md','data_boundary_audit.md','statistics_notes.md')
R6_EVIDENCE=('fixed_delay_protocol.json','model_sensitivity_protocol.json','contract_controller_checks.json',
    'contract_real_smoke.json','contract_controller.md','gp_active.md','model_sensitivity.md',
    'random_delivery.md','online_delivery_corrected.md','statistics.md','method_readonly_audit.md')
R6_FOLDERS=('fixed_delay','fixed_delay_corrected','gp_active','model_sensitivity',
            'random_delivery','online_delivery_corrected','statistics')
NUMERIC_EXTENSIONS={'.csv','.json','.npz','.txt'}

MIT='''MIT License

Copyright (c) 2026 Junjie Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
REQUIREMENTS='''numpy==2.5.1
pandas==3.0.5
scipy==1.18.0
scikit-learn==1.9.0
numba==0.67.0
threadpoolctl==3.6.0
pyarrow==25.0.0
'''
FIGURE_REQUIREMENTS='''-r requirements_reproduction.txt
matplotlib==3.11.1
pillow==12.3.0
pymupdf==1.28.2
'''
DATA_LICENSE='''# Data attribution and license

The observations and public-data derivatives are licensed under Creative
Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

Source: *QUANT: A Three-Year, Multi-City Air Quality Dataset of Commercial Air
Sensors and Reference Data for Performance Evaluation*, first hourly release,
https://doi.org/10.5281/zenodo.10775692 . The public record credits Sebastian Diez,
Stuart Lacy, Peter Edwards, Katie Read and Josefina Urquiza. Verbatim creator
metadata, including its original name ordering, is preserved in
SOURCE_ATTRIBUTION.json. Source article: Diez, Lacy, Urquiza and Edwards,
Scientific Data 11, 904 (2024), https://doi.org/10.1038/s41597-024-03767-2 .

This package supplies derived Parquet inputs rather than the original large
CSV files. Changes select hourly manufacturer-reported NO2 channels and their
sensor quality flags, separate Ratified reference targets and duplicate
reference measurements, and preserve source identity and hourly timestamps.
Scripts document eligibility, channel/deployment mapping, commissioning and
time splits. Prediction archives and analysis tables are derived computational
outputs. Credit the QUANT creators when reusing observations or their derivatives;
do not attribute the original observations to the software author.

The LICENSE file applies to author-written code and documentation (MIT,
Junjie Zhang); it does not replace the QUANT data license. Third-party Python
dependencies retain their respective licenses. No IEEE class/style files,
author photograph or unpublished article/supplement text is distributed here.
'''
CITATION='''cff-version: 1.2.0
message: "Please cite this software and credit the QUANT data source when reusing its derivatives."
type: software
title: "NO2 Reference Information Timing: Calibration and Reproduction Code"
version: 0.6.0
date-released: 2026-10-01
authors:
  - family-names: Zhang
    given-names: Junjie
    orcid: "https://orcid.org/0009-0004-8821-4018"
license: MIT
abstract: "Reproducible retrospective calibration and reference-service experiments on public QUANT NO2 data, including delayed-information planning and causally ordered event-specific delivery."
references:
  - type: data
    title: "QUANT: A Three-Year, Multi-City Air Quality Dataset of Commercial Air Sensors and Reference Data for Performance Evaluation"
    doi: 10.5281/zenodo.10775692
    year: 2024
    authors:
      - family-names: Diez
        given-names: Sebastian
      - family-names: Lacy
        given-names: Stuart
      - family-names: Edwards
        given-names: Peter
      - family-names: Read
        given-names: Katie
      - family-names: Urquiza
        given-names: Josefina
'''
README='''# NO2 reference information timing: reproducible calibration experiments

This source code and complete numerical archive support a retrospective sensor
calibration study on the public QUANT hourly release. The central question is
when event-time reference acquisition improves subsequently available outputs
under specified delivery and sharing contracts. This is not a new estimator,
prospective deployment, electrode mechanism or field-cost validation.

This local package is not evidence that a repository or software DOI already
exists. DOI 10.5281/zenodo.10775692 identifies the source QUANT dataset, not this
paper or software. CITATION.cff intentionally has no invented repository URL,
publication identifier or software DOI.

## Two delivery layers

The `revision6_publication_staging` source tree is small and suitable for a code
repository. It contains scientific programs, protocols, attribution and selected
summary tables, but no saved prediction archives or Parquet data. For numerical
validation and full execution use the **complete** `revision6_reproduction`
release archive. This is the computational package, not merely the figure files.
Hydrate a source checkout with a locally available complete release:

```text
python scripts/public_reproduction.py --hydrate /path/to/revision6_reproduction.zip --destination /new/computation/directory
```

Change into the single extracted project directory before the commands below.
Python 3.12 and the exact requirements were tested on Windows. All portable
entry points derive inputs from their own project root; Unix execution is
supported by path construction but has not been independently tested here.
Historical absolute paths in frozen metadata are provenance, not required
input locations. Preserve frozen Python bytes (`.gitattributes` disables line
ending conversion); do not silently change hashes to bypass provenance checks.

## Install and validate

Install numerical dependencies with `python -m pip install -r requirements_reproduction.txt`.
No GPU or network connection is needed for computation once the archive is
extracted. Use new working directories with adequate free space. On the
author's machine all computation, caches and temporary files belong on D:;
raw originals on F:/E: remain immutable. Other hosts should choose their own
computation storage, never silently overwrite a source archive.

```text
python scripts/public_reproduction.py --check-manifest
python scripts/public_reproduction.py --quick --output /new/path/quick_check
python scripts/public_reproduction.py --from-derived --output /new/path/from_data_check
```

`--quick` checks every published file's SHA-256, actually reruns three synthetic
controller/estimator suites in a disposable source copy, and independently
aggregates 21 three-site empirical stream groups. It does not refit the whole
study. `--from-derived` rebuilds the nine original periodic site/budget streams
from Parquet inputs using the unchanged frozen functions and compares every
prediction, calendar and scoring mask exactly. Saved predictions enter only
the final comparison. The original Windows-only CLI is preserved; the portable
adapter calls its unchanged computational functions and does not edit/bypass
its drive guard. The manifest excludes itself and is not an external timestamp
or registry signature.

## Scientific evidence and corrections

`results/main`, `results/revision` and `results/revision4` retain the original
numerical lineage. `results/revision6/fixed_delay_corrected` is the authoritative
120-cell fixed-delay matrix: physical commissioning allocation is kept fixed
while reference-information permission changes. `fixed_delay` remains only
because corrected generators and unchanged shared bootstrap lineage refer to
it; its nonshared rows are superseded and must not be used as final evidence.
`statistics` records the seven-comparison retrospective principal family,
conditional paired calendar-block intervals, Holm adjustment, device/context
diagnostics and descriptive sensitivities. All comparisons require documented
common supports, not a difference between unrelated summary rows.

`random_delivery` is a conditional replay of previously archived calendars
under simulated event-specific releases. Those transferred calendars are not
causal online decisions under their new delivery histories. In contrast,
`online_delivery_corrected` recomputes decisions from legally released history
and uses an exact out-of-sequence linear-Gaussian replay estimator. Delivery
draws are simulated contracts applied to real QUANT observations, not observed
field delivery logs. Three paired random seeds use delays {0,24,72,168} h with
probabilities {0.20,0.20,0.35,0.25}; the mean delay is 72 h.

Adaptive priors fitted from own commissioning labels and every fitted model's
planning contribution are unavailable until the declared `ready + 168 h`
bound. The final receipt audits actual delivery availability and absence of
adaptive decisions or scored predictions using fitted priors before that gate.
Earlier periodic-only predictions are unscored and never drive decisions. An earlier pre-fix online
experiment violated this gate and is excluded from this release; none of its
optimistic results are final. Random-contract mean instrument RMSE is 4.470307
ppb for delivery-aware, 4.472546 for the naive online objective, and 4.853746 for
periodic acquisition. The aware-minus-naive average is only -0.002239 ppb and
seed order reverses: these results do **not** establish an aware-over-naive
advantage under random release. The working fixed-72 h scheduler is conservative
and misspecified under random release, not an exact distribution-aware optimum.

Metrics first average concurrent channel squared errors within each physical
instrument-hour, then compute each instrument's RMSE and average instruments
equally. The fixed and corrected-random comparisons use 18 instruments and
59,674 common instrument-hours. Devices are not independent new-site samples;
intervals are conditional on the observed cohort and calendar-block procedure.
Expanded tuning uses pre-2021 Manchester only, but the revision is exploratory:
later evaluation data had already been inspected. No preregistration or
untouched independent confirmation is claimed.

## Full computation from source derivatives

Create a separate fresh checkout that copies code, derived inputs and original
early-development records, but **no saved evaluation prediction streams**:

```text
python scripts/public_reproduction.py --fresh /new/path/fresh_computation
```

Change into it. Set `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`, `TEMP`, and `TMP` to its
`cache/numba` and `temp` subdirectories; cap `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` at 2 and set
`PYTHONDONTWRITEBYTECODE=1`. The supplied `run_public_pipeline.py` creates
these directories/environment and runs the required computation stages:

```text
python scripts/run_public_pipeline.py --phase legacy
python scripts/run_public_pipeline.py --phase revision4
python scripts/run_public_pipeline.py --phase revision6
```

The phase runner records the exact commands and return codes. It refuses to
run without the fresh-checkout marker and refuses to treat a sealed release
as a mutable computation directory. A complete rebuild includes expensive GP
refits and thousands of bootstrap draws and takes substantially longer than
quick validation. The final scientific revision was computed in the original
project; the whole fresh pipeline has not additionally been rerun merely to
assemble this package. Legacy historical raw-download/development CLIs remain
unaltered and may have Windows storage guards; the documented pipeline uses
the portable baseline adapter and supplied derivatives, not new raw downloads.

The key individual revision-six commands (in a disposable computation copy) are:

```text
python scripts/revision6_contract_controller.py --contract-test
python scripts/revision6_fixed_delay.py --stage all
python scripts/revision6_correct_contract.py
python scripts/revision6_model_sensitivity.py
python scripts/revision6_model_sensitivity_audit.py
python scripts/revision6_random_delivery.py --stage all
python scripts/revision6_online_delivery.py --stage all
python scripts/revision6_gp_active.py run
python scripts/revision6_gp_active.py summarize
python scripts/revision6_gp_active.py verify
python scripts/revision6_gp_intervals.py
python scripts/revision6_statistics.py --stage all --matrix-directory results/revision6/fixed_delay_corrected
```

## Figures, separate from computation

Install `requirements_figures.txt`, then use a disposable full computation copy:

```text
python scripts/make_nature_figures.py --variant word
python scripts/make_nature_figures.py --variant latex
python scripts/make_revision6_figures.py
python scripts/make_revision6_graphical_abstract.py
```

Figures read completed numerical CSVs. The abstract adapter reads the two
explicitly different comparison/support rows from CSVs and reuses the retained
vector layout without unpublished manuscript text. PDF/SVG are editable/vector
masters and the raster exports are generated at actual print size. The legacy
figure-source verifier preserves the six frozen numerical source hashes; new
figure geometry checks run inside the revision-six renderer. Word/LaTeX
authoring and commercial journal template files are intentionally omitted.

See DATA_LICENSE.md for QUANT attribution (CC BY 4.0) and LICENSE for code (MIT).
RELEASE_CHANGES.md identifies authoritative corrections and the retained frozen
lineage. Manifests and prior runtime/path receipts are computational provenance,
not additional empirical evidence or a controlled speed comparison.
'''
CHANGES='''# Sixth-revision numerical release changes

This is a curated computational release from the original 1,109-file archive,
not a byte-identical repack of that entire article-delivery archive. Included
frozen scripts/inputs/outputs retain their bytes; FROZEN_SOURCE_IDENTITIES.json
records the exact retained subset against the prior source manifest. The new
REPRODUCTION_MANIFEST.json describes all actually delivered files.

Added: the 16-candidate earlier-development grid and selection stability;
120-cell delivery/sharing matrix; permissions-only commissioning allocation
correction; alternative residual/drift-process sensitivity; expanded GP
planning; conditional archived-calendar random-release replay; truly causal
online random-release experiment with commissioning-fit availability gates;
retrospective seven-contrast uncertainty/multiplicity analysis; independent
portable verification and lightweight source staging.

Authoritative fixed matrix: `results/revision6/fixed_delay_corrected`.
Authoritative random online experiment: `results/revision6/online_delivery_corrected`.
Old fixed nonsharing cells remain only as computational lineage; they are
superseded. The earlier online-delivery outputs used fitted commissioning
priors too early for random release and are excluded, together with their
superseded source snapshot. The final experiment uses the declared 168-hour
availability bound, with a separate passing availability audit.

Excluded from public distribution: unpublished article/supplement text,
Word/PDF manuscript files, author photograph, IEEE templates, assembly/export
scripts, editor correspondence, old layout/content QA, credentials and
account material, runtime logs/caches, original large raw CSV files. Scientific
negative and uncertain results are retained; exclusion is not based on sign.
'''


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def json_write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    before=digest(source);shutil.copy2(source,destination)
    assert before==digest(source)==digest(destination),f'Source changed during copy: {source}'


def identities(folder):
    return [dict(path=p.relative_to(folder).as_posix(),bytes=p.stat().st_size,sha256=digest(p))
            for p in sorted(folder.rglob('*')) if p.is_file()]


def prerequisites():
    if shutil.disk_usage(ROOT).free<2_000_000_000:raise RuntimeError('Insufficient D-drive staging space')
    assert len(list((ROOT/'results/revision6/fixed_delay_corrected').glob('*_shared*.npz')))==120
    online=ROOT/'results/revision6/online_delivery_corrected'
    assert len(list(online.glob('*_random*.npz')))==27
    for name in ('self_test.json','commissioning_availability.json'):
        record=json.loads((online/name).read_text());assert record['status']=='PASS',name
    receipt=json.loads((ROOT/'results/revision6/statistics/verification.json').read_text())
    assert receipt['status']=='PASS' and receipt['principal_family_count']==7
    for row in receipt['source_files']:
        source=ROOT/row['path'];assert source.is_file() and digest(source)==row['sha256'],row['path']
    assert (ROOT/'scripts/run_public_pipeline.py').exists()


def docs(destination,metadata):
    for name,text in {'README.md':README,'LICENSE':MIT,'DATA_LICENSE.md':DATA_LICENSE,
        'CITATION.cff':CITATION,'requirements_reproduction.txt':REQUIREMENTS,
        'requirements_figures.txt':FIGURE_REQUIREMENTS,'RELEASE_CHANGES.md':CHANGES,
        '.gitattributes':'# Preserve hashes of frozen scientific source bytes.\n* -text\n',
        '.gitignore':'__pycache__/\n*.pyc\n*.nbc\n*.nbi\ncache/\ntemp/\n*.zip\n*.docx\n*.tex\n*.cls\n*.sty\n*.bst\ndatasets/*.parquet\nresults/**/*.npz\nresults/**/*.png\nresults/**/*.pdf\nresults/**/*.svg\n'}.items():
        (destination/name).write_text(text,encoding='utf-8')
    json_write(destination/'SOURCE_ATTRIBUTION.json',metadata)


def prepare(full,staging):
    prerequisites()
    for target in (full,staging):
        assert target.resolve().is_relative_to((ROOT/'delivery').resolve()),target
        if target.exists():raise FileExistsError(f'Choose a new destination; no deletion: {target}')
    old=json.loads((BASE/'REPRODUCTION_MANIFEST.json').read_text())
    old_entries={item['path']:item for item in old['files']}
    record=json.loads(RAW_METADATA.read_text())['record']
    metadata=dict(dataset_title=record['metadata']['title'],doi=record['metadata']['doi'],
        license=record['metadata']['license'],creators_verbatim=record['metadata']['creators'],
        source_record_api='https://zenodo.org/api/records/10775692',
        source_paper_doi='10.1038/s41597-024-03767-2',
        software_author='Junjie Zhang',software_orcid='0009-0004-8821-4018',
        software_license='MIT',source_manifest_scope='Original observations CC BY 4.0; author code MIT')
    full.mkdir(parents=True);staging.mkdir(parents=True)
    retained=[]
    def frozen(path):
        relative=path.relative_to(BASE)
        if relative.as_posix() in old_entries:
            entry=old_entries[relative.as_posix()]
            assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],relative
            retained.append(entry)
        copy(path,full/relative)
    for path in (BASE/'scripts').glob('*.py'):
        if not path.name.startswith(EXCLUDED_PREFIXES) and path.name not in EXCLUDED_SCRIPTS:frozen(path)
    for name in EXTRA_SCRIPTS+('run_public_pipeline.py',):copy(ROOT/'scripts'/name,full/'scripts'/name)
    for path in (BASE/'datasets').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.parquet'):frozen(path)
    for name in BASE_EVIDENCE:
        path=BASE/'evidence'/name
        if path.exists():frozen(path)
    for path in (BASE/'results').rglob('*'):
        if not path.is_file() or path.suffix not in NUMERIC_EXTENSIONS:continue
        relative=path.relative_to(BASE/'results')
        if len(relative.parts)==1 or relative.parts[0] in ('main','revision','revision4','tables'):
            frozen(path)
    for folder in R6_FOLDERS:
        for path in (ROOT/'results/revision6'/folder).rglob('*'):
            if path.is_file() and path.suffix in NUMERIC_EXTENSIONS:
                copy(path,full/path.relative_to(ROOT))
    for name in R6_EVIDENCE:
        source=ROOT/'evidence/revision6'/name
        if source.exists():copy(source,full/source.relative_to(ROOT))
    copy(ROOT/'REVISION6_STATE.md',full/'REVISION6_STATE.md')
    docs(full,metadata);docs(staging,metadata)
    # Every source of final statistics must be present, with its original byte identity.
    receipt=json.loads((full/'results/revision6/statistics/verification.json').read_text())
    for row in receipt['source_files']:assert digest(full/row['path'])==row['sha256'],row['path']
    json_write(full/'FROZEN_SOURCE_IDENTITIES.json',dict(
        source_archive_manifest_sha256=digest(BASE/'REPRODUCTION_MANIFEST.json'),
        source_archive_total_files=1109,retained_files=len(retained),files=retained,
        statement='Retained subset only; excluded editorial/manuscript files are not claimed as preserved in this release.'))
    # Source repository: code, protocols, attribution, and small numerical summaries.
    for folder in ('scripts','evidence'):
        shutil.copytree(full/folder,staging/folder,dirs_exist_ok=True)
    for source in (full/'datasets').glob('*.json'):copy(source,staging/source.relative_to(full))
    copy(full/'REVISION6_STATE.md',staging/'REVISION6_STATE.md')
    for source in (full/'results/revision6').rglob('*.csv'):
        if source.stat().st_size<100_000:copy(source,staging/source.relative_to(full))
    for destination in (full,staging):
        for source in (destination/'scripts').glob('*.py'):
            compile(source.read_bytes(),str(source),'exec')
        assert not any(p.suffix in ('.docx','.tex','.cls','.sty','.bst') for p in destination.rglob('*') if p.is_file())
        assert not any(p.name in ('.env','id_rsa','credentials.json') for p in destination.rglob('*'))
    json_write(full/'REPRODUCTION_MANIFEST.json',dict(
        format_version=1,version='0.6.0',created_utc=datetime.now(timezone.utc).isoformat(),
        files=identities(full),notes='Scientific numerical archive; manifest excludes itself.'))
    json_write(staging/'CODE_STAGING_MANIFEST.json',dict(
        format_version=1,version='0.6.0',files=identities(staging),
        scope='Lightweight code staging; complete release archive required for numerical verification.'))
    return full,staging


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full',type=Path,default=ROOT/'delivery/revision6_reproduction')
    ap.add_argument('--staging',type=Path,default=ROOT/'delivery/revision6_publication_staging')
    ap.add_argument('--zip',type=Path,default=ROOT/'delivery/revision6_reproduction.zip')
    ap.add_argument('--refresh-documentation',action='store_true',help='Before external publication only: update this package script and README; preserve all scientific file bytes.')
    args=ap.parse_args()
    if args.refresh_documentation:
        full,staging=args.full.resolve(),args.staging.resolve()
        for folder,manifest_name in ((full,'REPRODUCTION_MANIFEST.json'),(staging,'CODE_STAGING_MANIFEST.json')):
            assert folder.is_relative_to((ROOT/'delivery').resolve()),folder
            old=json.loads((folder/manifest_name).read_text())
            for row in old['files']:assert digest(folder/row['path'])==row['sha256'],row['path']
            (folder/'README.md').write_text(README,encoding='utf-8')
            copy(Path(__file__),folder/'scripts'/Path(__file__).name)
            files=[row for row in identities(folder) if row['path']!=manifest_name]
            old['files']=files;old['documentation_updated_utc']=datetime.now(timezone.utc).isoformat()
            json_write(folder/manifest_name,old)
    else:
        if args.zip.exists():raise FileExistsError('Choose a new ZIP destination; frozen assets are not overwritten.')
        full,staging=prepare(args.full.resolve(),args.staging.resolve())
    args.zip.parent.mkdir(parents=True,exist_ok=True)
    zip_target=args.zip.with_name(args.zip.name+'.unpublished.tmp') if args.refresh_documentation else args.zip
    if zip_target.exists():raise FileExistsError(zip_target)
    with zipfile.ZipFile(zip_target,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for source in sorted(full.rglob('*')):
            if source.is_file():z.write(source,Path(full.name)/source.relative_to(full))
    with zipfile.ZipFile(zip_target) as z:assert z.testzip() is None
    if args.refresh_documentation:zip_target.replace(args.zip)
    rows=[]
    for folder in (full,staging):
        files=[p for p in folder.rglob('*') if p.is_file()]
        largest=max(files,key=lambda p:p.stat().st_size)
        rows.append(dict(path=str(folder),files=len(files),bytes=sum(p.stat().st_size for p in files),
            largest_file=largest.relative_to(folder).as_posix(),largest_bytes=largest.stat().st_size))
    result=dict(status='LOCAL_PACKAGED_NOT_UPLOADED',packages=rows,
        zip=dict(path=str(args.zip),bytes=args.zip.stat().st_size,sha256=digest(args.zip)))
    json_write(ROOT/'evidence/revision6/local_package_receipt.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
