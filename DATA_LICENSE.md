# Data attribution and license

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
