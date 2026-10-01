# NO2 reference information timing: reproducible calibration experiments

This source code and complete numerical archive support a retrospective sensor
calibration study on the public QUANT hourly release. The central question is
when event-time reference acquisition improves subsequently available outputs
under specified delivery and sharing contracts. This is not a new estimator,
prospective deployment, electrode mechanism or field-cost validation.

Public source: https://github.com/williamjay1/no2-reference-information-timing

The complete numerical archive is the `revision6_reproduction.zip` asset for
release v0.6.0. The source checkout is deliberately lightweight; download the
complete archive from the repository's Releases section before running numerical
verification. DOI 10.5281/zenodo.10775692 identifies the source QUANT dataset,
not this paper or software. No software DOI or accepted publication identifier
is claimed. The archive's README and protocol preserve their pre-publication
creation state; this repository supplies the current public location.

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
