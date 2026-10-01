# QUANT data boundary audit

Audit date: 2026-09-30. This audit inspects identities, values and source preprocessing; it does not compare predictive performance.

## 1. IMB concurrent NO2 channels are not exact duplicates

The QUANT first-party [Bosch collation script](https://github.com/wacl-york/quant-tools/blob/main/setup-database/1-preprocess-lcs-data/collate_Bosch.R) explicitly handles three NO2 measurements, mapping measurement suffixes `_2` and `_3` to subsidiary channel names. The [replacement script](https://github.com/wacl-york/quant-tools/blob/main/setup-database/3-db-updates/sensor_replacement.R) states that its replacement handling affects Aeroqual and Zephyr from the main study. Therefore a universal interpretation of sensornumber as sequential sensor replacement is incorrect.

| Instrument | Version | Channels | Paired hours | Exact equal | Max absolute difference (ppb) | Pearson correlation |
|---|---|---:|---:|---:|---:|---:|
| IMB1 | out-of-box | 1-2 | 1772 | 0 | 3.501502 | 0.992596 |
| IMB1 | out-of-box | 1-3 | 1772 | 0 | 7.205465 | 0.947123 |
| IMB1 | out-of-box | 2-3 | 1772 | 0 | 5.823449 | 0.938985 |
| IMB1 | cal1 | 1-2 | 4510 | 0 | 4.948306 | 0.994650 |
| IMB1 | cal1 | 1-3 | 4510 | 0 | 7.497890 | 0.978877 |
| IMB1 | cal1 | 2-3 | 4510 | 0 | 9.401697 | 0.987313 |
| IMB2 | out-of-box | 1-2 | 1811 | 0 | 12.253849 | 0.989910 |
| IMB2 | out-of-box | 1-3 | 1811 | 0 | 9.610471 | 0.929019 |
| IMB2 | out-of-box | 2-3 | 1811 | 0 | 18.891059 | 0.929709 |
| IMB2 | cal1 | 1-2 | 9459 | 0 | 6.146486 | 0.993034 |
| IMB2 | cal1 | 1-3 | 9459 | 0 | 17.099896 | 0.973836 |
| IMB2 | cal1 | 2-3 | 9459 | 0 | 20.014836 | 0.954750 |

Every pair differs at every jointly observed hour. Identical coverage is consistent with simultaneous channels, not duplicated values. Retain channel records if the protocol permits, but aggregate/cluster uncertainty by physical instrument and site; IMB1/IMB2 constitute two instruments, not six independent instruments. The version cal1 is shown solely to establish identity and is excluded from model inputs.

Temperature and humidity records occur under IMB sensornumber 1 only. The current exact channel join preserves that missingness for channels 2 and 3; no weather is copied between channels. An x-only fallback or initial-only imputation keeps these channels without using reference meteorology.

## 2. Temperature scale and range

The first-party [CEDA CSV export script](https://github.com/wacl-york/quant-tools/blob/main/export-for-CEDA/export_lcs_worker_csv.R) defines `celsius_to_kelvin <- function(x) x + 273.15` and applies it to the exported temperature column. The [netCDF worker](https://github.com/wacl-york/quant-tools/blob/main/export-for-CEDA/export_lcs_worker.R) does likewise. The Zenodo CSV values below are already Celsius-scale; the publication/CEDA Kelvin description must not be blindly applied to this CSV. No Kelvin subtraction was performed.

The following descriptive quantiles use all out-of-box Temperature rows, grouped by instrument prefix (including families without eligible NO2 observations).

| Family | n | Min | 1% | Median | 99% | Max |
|---|---:|---:|---:|---:|---:|---:|
| AP | 35848 | -2.5242 | 1.3907 | 14.1433 | 29.1418 | 40.8517 |
| AQM | 134416 | -3.9817 | 0.3500 | 12.8017 | 29.4497 | 42.5000 |
| AQY | 83549 | -4.1997 | 0.4430 | 12.5850 | 29.3983 | 40.9930 |
| Ari | 88570 | -1.1379 | 3.8501 | 15.8178 | 34.5873 | 48.1217 |
| Atm | 2480 | 0.0000 | 0.0000 | 18.2627 | 33.4280 | 8020.8644 |
| IMB | 17554 | -1.2055 | 1.7581 | 14.6473 | 29.2481 | 42.4867 |
| Mod | 35836 | -1.7892 | 2.2089 | 15.2227 | 30.9357 | 43.3690 |
| NS | 36262 | -0.8464 | 3.0616 | 15.5862 | 30.4098 | 41.7355 |
| PA | 174131 | -8.3333 | 2.1264 | 13.8624 | 35.8594 | 59.7817 |
| Poll | 20363 | -3.3267 | 0.8764 | 14.6167 | 41.8959 | 49.3200 |
| Prax | 23702 | -0.1133 | 3.6934 | 15.9267 | 29.7733 | 40.8717 |
| Zep | 67622 | -130.0000 | -130.0000 | 12.2417 | 35.2966 | 331.0000 |

Among the 149,202 Ratified paired NO2 records, 516 temperature values lie outside [-40,65] and 528 RH values lie outside [0,100]. These are covariate QC findings, not outcome-based exclusions. The main loader masks such weather values and retains valid concentration observations; sensor Error flags and non-Ratified targets are also masked. Warning flags remain available for sensitivity analysis.

## 3. Time axis

The raw hourly CSV stores timestamps without an offset. The first-party export worker writes coverage dates with `format_ISO8601(..., usetz="Z")`, and [reference collation](https://github.com/wacl-york/quant-tools/blob/main/setup-database/1-preprocess-lcs-data/collate_Reference.R) uses lubridate `as_datetime`, whose default is UTC. These establish a UTC-oriented upstream convention, although the CSV strings alone do not identify a timezone. Analysis uses the supplied hour axis unchanged, joining sensor and reference records on exact timestamp/site. No DST correction, one-hour shift, local-time calendar conversion, or invented timezone offset is applied. Duration budgets use elapsed source hours.

## 4. Calibration and reference boundary

Only out-of-box sensor concentration products enter the model table. Both cal1 and cal2 are excluded because manufacturers had reference data for retrospective recalibration. These are manufacturer concentration outputs, not electrode voltages or entirely unprocessed measurements. The [Zephyr collation script](https://github.com/wacl-york/quant-tools/blob/main/setup-database/1-preprocess-lcs-data/collate_Zephyr.R) explicitly confines the first calibration to NO and relabels other measurands as out-of-box. Upstream factory processing and manufacturer quality controls remain part of the data-generating system.

Use no2_hourly_oob.parquet as the full sensor-availability boundary, because the paired Ratified-only derivative discards hours with missing or unratified reference. Reindex each deployment to a complete hourly grid before applying budgets or 90-day rolling windows. Expose reference values to model updates only at requested times; missing targets must consume the scheduled window time without triggering a retry using future knowledge.

## 5. Identity and evaluation constraints

instrument identifies a system; sensornumber distinguishes either simultaneous sensing channels (IMB has three concurrent NO2 measurements) or replacement segments (documented for Aeroqual and Zephyr). device=instrument+sensornumber is a channel/segment key, not an independent-instrument count; it remains the same across sites. deployment additionally includes site. Group uncertainty by instrument/site as appropriate; do not treat channels, replacement segments, moves, or hourly rows as independent instruments.

Eligibility is based only on minimum initialization pairs and minimum available scoring pairs, not prediction errors. Development is Manchester observations before 2021-01-01; no 2021+ performance is examined by this audit. Main evaluation must report its actual site-year coverage: London has no eligible Ratified paired observations in 2022, although its reference series extends into 2022. Three cities across a multiyear archive does not imply a fully crossed city-year panel.

The Zenodo metadata declares CC BY 4.0. CEDA publication/distribution licensing should be cited separately if used; do not replace the license recorded for the downloaded Zenodo version.
