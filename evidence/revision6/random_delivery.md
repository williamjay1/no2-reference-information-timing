# Random reference delivery and delay mismatch

These are retrospective fixed-calendar transfer tests on real QUANT sensor and reference observations. They do not implement or validate an online distribution-aware planner. The transferred adaptive calendars were generated under their original fixed-delay contracts and remain fully committed during these replays.

Per-event delays are 0, 24, 72, or 168 hours with probabilities 0.20, 0.20, 0.35, and 0.25 (expected 72 hours), using three fixed seeds. A potential delay is drawn for every site/event hour before applying any calendar, making delivery draws paired across the three calendars. Simulated delivery is not an observed release log. All outputs precede same-hour releases. The engine inserts out-of-sequence observations at event time and refilters a maximum 168-hour suffix using only labels already delivered. A 170-state ring preserves the pre-insertion anchor. Historical outputs are never revised.

Nine constant-72-hour real-data parity checks match the frozen fixed-delay streams. Synthetic implementation tests separately establish constant-delay parity at 0/24/72/168 hours, hidden-target and future-label invariance, output-before-release ordering, and agreement with an independent dense joint-Gaussian conditioning solution for reverse arrivals and simultaneous releases. Synthetic tests are not empirical results.

Random-delivery results contain 27 site streams; delay mismatch contains 12. Every comparison uses 18 physical instruments and 59,674 common instrument-hours after ready+168. Random seed ranges are conditional simulation variability and must not be described as deployment confidence intervals. Delays assumed by the retained-aware calendar (0/24/72/168 hours) are separately transferred to actual 72-hour delivery.

## Aggregate random-delivery RMSE (ppb)

seed,policy,macro_instrument_rmse
610001,aware_retained,4.2919677442240465
610001,original,4.983069014551742
610001,periodic,4.926763579448756
610002,aware_retained,4.222137272988402
610002,original,4.949148805965301
610002,periodic,4.827342089659511
610003,aware_retained,4.245162294375138
610003,original,4.997894697232164
610003,periodic,4.807132075884138

## Assumed-delay calendars under actual 72-hour delivery

policy,assumed_delay,macro_instrument_rmse
aware_retained,0,4.830534192982297
aware_retained,168,4.628837394691726
aware_retained,24,4.752648354520107
aware_retained,72,4.664218949132904
periodic,72,5.060452888471006
