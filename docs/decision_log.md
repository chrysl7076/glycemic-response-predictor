# Decision Log

| # | Date / time | Decision | Reason | Impact |
|---|---|---|---|---|
| D1 | 23 Oct, 07:00 | Solo project, not a 3-person team | 1-day timeline; leadership already shown via E-Cell role; clearer ownership | Smaller scope, faster delivery |
| D2 | 23 Oct, 07:00 | Drop the temporal neural network | Not feasible to do well in 1 day | Logged as future work |
| D3 | 23 Oct, 07:00 | Drop sleep features | Not present in CGMacros (checked data dictionary) | One fewer planned feature group |
| D4 | 23 Oct, 07:00 | Add carbs-only linear baseline | A group-mean baseline is too easy to beat; tests whether personal factors add value beyond carbs | Directly answers the research question |
| D5 | 23 Oct, 07:00 | Add 2-hour iAUC as a target | Standard metric in postprandial glucose research | Comparable to literature |
| D6 | 23 Oct, 07:00 | Assign groups from HbA1c (<5.7 / 5.7–6.4 / ≥6.5 %) | Group label not in data files | Verify against CGMacros paper |
| D7 | 23 Oct, 07:00 | Interpolate CGM gaps ≤20 min only; exclude meals with another meal within 2 h | Avoid inventing data; avoid overlapping responses | Fewer meals, cleaner targets |
| D8 | 23 Oct, 08:00 | Download from PhysioNet's AWS mirror | Direct download estimated at 1 day (risk materialized) | ~4.5 h delay; analysis window compressed, scope kept |
| D9 | 23 Oct, 13:15 | Use Libre sensor as glucose source | Available for all 45 participants at 1-min resolution; Dexcom has gaps | Consistent target across participants |
| D10 | 23 Oct, 13:15 | Keep snacks as meals; also use them to exclude overlapping windows | Snacks change glucose; ignoring them would contaminate other meals' responses | 207 snacks in data; 250 meals excluded for overlap |
| D11 | 23 Oct, 13:20 | Exclude meals >3000 kcal or where macro energy ≠ logged calories (ratio outside 0.5–2) | Clear logging errors found (e.g., 2283 g carbs in a 3090 kcal meal) | 46 meals excluded; 1404 meals kept |
| D12 | 23 Oct, 13:20 | Confirmed HbA1c grouping | Produces 15 / 16 / 14 participants, matching the CGMacros paper | Assumption from D6 verified |
| D13 | 25 Oct | Add personal history features (from each person's earlier meals only) | "Personalised" model never used a person's own past responses; key gap an expert would flag | 60-min error −11% vs v1; largest gains for T2D and men |
| D14 | 25 Oct | Add temporal CNN predicting the full 2-h curve, plus per-person fine-tuning | Test whether a deep temporal model beats trees; enable full-curve output | CNN best at 30 min; LightGBM best at 60/120 min; fine-tuning +18% |
| D15 | 25 Oct | Evaluate personalisation on each person's later half of meals | Fair test: adapt on earlier meals, score on later ones, same meals for every model | Personalised LightGBM best (−38% vs carbs only) |
| D16 | 25 Oct | No hyperparameter tuning; report all ensemble combinations tried | Avoid optimistic results on a small cohort | Best spike AUROC 0.89 reported as approximate |
| D17 | 25 Oct | Sensitivity analysis (Dexcom sensor, no overlap rule, no plausibility filter) | Check findings don't depend on cleaning choices | Improvement 23–25% in all settings |
