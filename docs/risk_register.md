# Risk Register

Likelihood and impact are rated Low / Medium / High. Status is as of the end of the sprint (23 Oct 2026).

| ID | Risk | Likelihood | Impact | Mitigation | Owner | Status / outcome |
|---|---|---|---|---|---|---|
| R1 | Data access delay | Medium | High | Confirm access terms before starting | Chrysl | **Occurred.** PhysioNet direct download estimated at 1 day. Switched to the official AWS mirror; ~4.5 h lost; schedule re-planned, scope kept (D8) |
| R2 | Preprocessing overruns (timestamps, gaps, meal alignment) | High | High | Starter pipeline tested on synthetic data in the same layout; fallback to 60-min horizon only | Chrysl | **Avoided.** Real data ran with minor changes |
| R3 | Data quality problems in meal logs | Medium | Medium | Plausibility checks on macros vs calories | Chrysl | **Occurred.** 46 implausible meals found and excluded (D11) |
| R4 | Too few clean meals per subgroup | Medium | Medium | Report n per subgroup; flag exploratory results | Chrysl | **Partly occurred.** Groups are balanced (412–520 meals), but sex and age splits are small and labelled exploratory |
| R5 | Model fails to beat baselines | Medium | Low | Treat a negative result as valid; report honestly | Chrysl | **Did not occur.** LightGBM beat both baselines at every horizon with CIs above 0 |
| R6 | Data leakage inflating results | Low | High | Pre-meal features only; participant-grouped CV; rules written in README | Chrysl | **Controlled** by design |
| R7 | Overstating results in write-up | Medium | High | Report CIs, subgroup gaps, and limitations prominently | Chrysl | **Controlled.** Modest gain and T2D gap stated in headline |
| R8 | Write-up squeezed by overruns | Medium | High | Hard stop on analysis; write-up time protected | Chrysl | Managed via re-plan after R1 |
| R9 | License / attribution breach | Low | Medium | Cite dataset and paper; no raw data in repo; non-commercial use | Chrysl | **Controlled.** Raw data excluded via .gitignore |
| R10 | Scope creep (neural net, microbiome, photos) | High | Medium | Out-of-scope list in charter; changes logged | Chrysl | **Controlled.** All logged as future work |
