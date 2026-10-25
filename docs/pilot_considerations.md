# From Model to Clinic: What a Pilot Would Need

This project is a research prototype. This note sets out what would have to happen before a model like this could be piloted, for example in a diabetes prevention programme or a dietitian's practice.

## 1. Who would use it, and for what
| Stakeholder | Need | Main concern |
|---|---|---|
| Patients / participants | Know which meals spike *their* glucose | Accuracy, privacy, not being blamed |
| Dietitians / diabetes educators | Personalise advice faster | Trust, explainability, workload |
| Clinicians | Support, not replace, judgement | Safety, liability, regulatory status |
| Programme / hospital managers | Better outcomes at acceptable cost | Integration, cost, evidence |
| Regulators | Patient safety | Intended use, validation, monitoring |

**Recommended intended use:** educational, low-risk decision support (e.g., "this meal is likely to raise your glucose more than usual"). It should not be used for insulin dosing or diagnosis.

## 2. Evidence gaps to close first
1. **External validation** on a larger, more diverse cohort, including the target population.
2. **Acceptable accuracy in every subgroup.** Today's performance for type 2 diabetes participants (6% gain over carbs alone) is too weak.
3. **Clinically meaningful thresholds:** agree with clinicians what error level is useful (e.g., correctly flagging high vs low response meals).
4. **Prospective test:** compare against standard dietary advice on a real outcome (time-in-range, HbA1c).

## 3. Regulatory and governance
- Depending on claims and jurisdiction, the tool may count as software as a medical device. Intended-use wording decides the regulatory pathway.
- Data protection: CGM and health data are sensitive. A pilot needs consent, data minimisation, secure storage, and a data-sharing agreement.
- Ethics approval for any prospective study.
- Model governance: version control, monitoring for drift, and a process for pulling the model if performance drops.

## 4. Practical integration
- **Inputs:** CGM data plus meal logging. Meal logging is the weakest link, since self-estimated macros are noisy. Photo-based logging may help.
- **Workflow:** how results reach the dietitian (dashboard, report) without adding workload.
- **Cost:** CGM sensors are the main cost driver. Short CGM periods used for "learning" a person's response may be enough.

## 5. Suggested pilot plan (high level)
| Phase | Duration | Goal | Exit criterion |
|---|---|---|---|
| 0. Validation | 3 months | Retrain and test on a new dataset | Accuracy holds in all subgroups |
| 1. Usability | 2 months | 10–20 users, dietitian feedback | Users understand and trust outputs |
| 2. Feasibility pilot | 6 months | 50–100 participants vs usual care | Engagement and safety acceptable |
| 3. Effectiveness trial | 12+ months | Measure time-in-range / HbA1c | Clinically meaningful improvement |

## 6. Key risks
- Over-trust in predictions by users. Mitigate with clear uncertainty display and education.
- Widening health gaps if the tool works worse for some groups. Mitigate with subgroup monitoring as a release criterion.
- Poor meal logging quality. Mitigate with simpler logging and photo support.
