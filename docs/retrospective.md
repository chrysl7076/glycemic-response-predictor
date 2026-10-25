# Retrospective: One-Day Sprint

*Draft. Edit in your own words before publishing.*

## Outcome vs plan
| Success criterion (charter) | Result |
|---|---|
| S1 Beat carbs-only baseline at 60 min, CI above 0 | ✅ 15% lower error (95% CI 7–23%) |
| S2 Report accuracy by group | ✅ Plus sex and age splits |
| S3 SHAP drivers match physiology | ✅ Carbs, fasting glucose, HbA1c up; protein, fiber, activity down |
| S4 Reproducible in 2 commands | ✅ `preprocess.py` → `train.py` |
| S5 Delivered same day | ✅ Delivered same day, ~1 h later than the original plan |

## What went well
- **Scoping early paid off.** Freezing scope in the charter (no neural net, no microbiome, no photos) kept the day focused when time ran short.
- **Testing on synthetic data before the real data arrived** meant the pipeline worked almost first time once the download finished, even after the delay.
- **A stronger baseline made the result meaningful.** Adding a carbs-only baseline turned "the model beats an average" into "personal factors add value beyond carbs."
- **Subgroup analysis found the most important insight:** the model helps type 2 diabetes participants least.

## What didn't go well
- **Data download delay (~4.5 h).** I didn't check download speed or mirrors before the day started.
- **Meal logs were messier than expected.** Some entries were clearly wrong (e.g., 2,283 g of carbs), which I only found by inspecting distributions.
- **Unequal subgroup performance** is a limitation I can describe but couldn't fix in one day.

## What I'd do differently
1. Download and inspect data the day **before** a time-boxed sprint. Treat data access as a dependency, not a task.
2. Add data-quality checks as a named step in the plan, not something discovered mid-analysis.
3. Set a subgroup accuracy threshold in the charter, not only an overall one.

## Lessons for project management
- A risk register is only useful if you act on it. R1 (data delay) was listed in the charter, so switching to a mirror and re-planning was quick.
- Logging decisions as they happen (12 entries) took seconds each and made the write-up much faster.
- Cutting scope, not quality, was the right trade-off under a fixed deadline.
