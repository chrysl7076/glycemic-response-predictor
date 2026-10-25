# Project Charter — Glycemic Response Predictor

**Owner:** Chrysl Sheckina · **Type:** Solo, time-boxed 1-day sprint · **Date:** 23 Oct 2026

## 1. Problem
People eat the same meal and get very different blood glucose responses, so generic diet advice often fails. This project tests whether personal and meal-level data can predict an individual's post-meal glucose response.

## 2. Research question
Can meal composition, timing, recent activity, and metabolic markers predict an individual's post-meal glucose change better than simple baselines, and does accuracy hold across healthy, prediabetic, and type 2 diabetic participants?

## 3. Success criteria
| # | Criterion | Measure |
|---|---|---|
| S1 | Beats a carbs-only baseline | LightGBM MAE lower at 60 min, 95% CI of improvement above 0 |
| S2 | Holds across groups | MAE reported per group, with sample sizes |
| S3 | Explainable | SHAP drivers checked against known physiology |
| S4 | Reproducible | Results regenerate from a clean clone with 2 commands |
| S5 | On time | Public repo and write-up delivered by 19:00 IST, 23 Oct |

## 4. Scope
**In scope:** CGMacros dataset (PhysioNet, open access); meal-level prediction at 30 / 60 / 120 min plus 2-hour iAUC; two baselines (group mean, carbs-only linear) vs LightGBM; participant-grouped cross-validation; subgroup analysis; SHAP; README, limitations, ethics, clinic pilot considerations.

**Out of scope (future work):** temporal neural network, full-curve prediction, hyperparameter tuning, microbiome features, food photos, external validation, any clinical use.

## 5. Deliverables
1. Public GitHub repo: code, environment file, README
2. Results tables: overall and by subgroup, with uncertainty
3. SHAP plots and interpretation
4. Limitations, ethics, and clinic pilot section
5. PM artifacts: this charter, risk register, decision log, retrospective

## 6. Milestones
| Time (IST) | Milestone |
|---|---|
| 07:30 | Charter signed off, scope frozen |
| 08:30 | Repo and environment set up, data downloaded |
| 11:00 | Clean meal-level dataset validated |
| 12:30 | Model results produced |
| 14:00 | Headline finding written |
| 17:45 | Write-up complete |
| 19:00 | Repo public, resume updated |

## 7. Constraints and assumptions
- **Time:** 1 working day. Scope is cut, not quality.
- **Data:** about 45 participants, about 10 days each, free-living (not controlled) meals, macros estimated from logs.
- **Assumption:** groups are assigned from baseline HbA1c (ADA cut-offs), to be checked against the CGMacros paper.
- **License:** CC BY-NC-SA 4.0. Attribution required, non-commercial only.

## 8. Top risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Preprocessing overruns (timestamps, gaps) | High | High | Starter code tested on synthetic data; fall back to the 60-min horizon only |
| Too few clean meals per subgroup | Medium | Medium | Report n per group; flag low-confidence results |
| Model doesn't beat baselines | Medium | Low | A negative result is still a valid finding; report it honestly |
| Data leakage | Low | High | Pre-meal features only; participant-level splits; rules in README |
| Write-up squeezed by overruns | Medium | High | Hard stop on analysis at 14:00 |

## 9. Ethics
De-identified public data, used under its license. Results are not medical advice and not validated for clinical use. AI assistance is disclosed in the README.
