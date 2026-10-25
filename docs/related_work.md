# Related Work

## 1. Personalised glycemic response prediction
**Zeevi et al. (2015), *Cell*.** Monitored 800 Israeli adults for a week (46,898 meals) and built a gradient-boosting model using blood measures, diet, anthropometrics, activity, and gut microbiome. It predicted individual post-meal responses (iAUC) far better than carbohydrate or calorie counting, and was validated in an independent cohort of 100 people. This study established that responses to identical meals vary widely between people and that ML can predict this variation. *(Their reported validation correlation is about 0.7. Check the exact figure in the paper before citing it.)*

**Berry et al. (2020), *Nature Medicine* (PREDICT 1).** Studied 1,002 UK twins and unrelated adults, validated in 100 US participants. They found large differences between people in responses to identical meals (coefficient of variation 68% for glucose). Meal macronutrients explained more glucose-response variance (15.4%) than the gut microbiome (6.0%), and genetics explained about 9.5%. Their ML model predicted glucose responses with **r = 0.77**.

## 2. Prediction in diabetes and with limited data
**Shen, Choi & Kleinberg (2025/2026), *Journal of Diabetes Science and Technology*.** Predicted 2-hour postprandial response and peak rise in 397 people with type 1 diabetes (T1DEXI) and 100 with type 2 diabetes (ShanghaiT2DM), without microbiome data. They reported **r = 0.61–0.73**, found that food-category features beat macronutrients alone, that time of day and menstrual phase mattered, and **concluded that personalised training data was not required**.

## 3. The CGMacros dataset
**Das et al. (*Scientific Data*); Gutierrez-Osuna et al. (2025, PhysioNet).** CGMacros is a multimodal pilot dataset with two CGM sensors, Fitbit activity, meal macronutrients and photos, blood labs, and microbiome data for 45 adults across the metabolic spectrum. It was released for research on personalised nutrition and diet monitoring.

## 4. How this project compares

| Study | People | Population | Microbiome? | Best glucose-response correlation |
|---|---|---|---|---|
| Zeevi et al. 2015 | 800 (+100) | Israeli adults, mostly non-diabetic | Yes | ≈ 0.7 (verify) |
| Berry et al. 2020 | 1,002 (+100) | UK / US healthy adults | Yes | 0.77 |
| Shen et al. 2025 | 397 / 100 | T1D / T2D | No | 0.61–0.73 |
| **This project, new people (M2)** | 45 | Healthy, prediabetes, T2D (US) | No | **0.63** (iAUC) |
| **This project, personalised (M2b)** | 45 | Healthy, prediabetes, T2D (US) | No | **0.78** (iAUC) |

The carbs-only baseline in this project reached r = 0.12.

## 5. What this project adds and how to read it
1. **Comparable accuracy with far fewer people and no microbiome data, but only when personalised.** The non-personalised model (r ≈ 0.63) is below the large cohort studies. Adding a few days of the person's own meal responses brings it into the same range (r ≈ 0.78). The comparison is indicative only: the cohorts, meals, metrics, and validation designs all differ.
2. **A different conclusion on personal data than Shen et al.** They found personalised training data unnecessary, whereas here it gave the largest single improvement. Possible reasons:
   - this cohort is much smaller, so population-level patterns are learned less well and personal data compensates
   - CGMacros spans healthy to T2D, where between-person differences are larger
   - this project measures the benefit within the same short (about 10-day) window

   This disagreement is worth testing on a larger cohort.
3. **Model comparison.** Like the large studies, a gradient-boosting model was the strongest single model. A temporal CNN added value for short horizons and full-curve prediction, and combining the two gave the best spike detection.
4. **Fairness across metabolic groups.** Personalisation narrowed, but did not close, the accuracy gap for type 2 diabetes. This mirrors the broader observation that responses are more variable in people with impaired glucose control.

## References
- Zeevi D, Korem T, Zmora N, et al. Personalized nutrition by prediction of glycemic responses. *Cell*. 2015;163(5):1079–1094.
- Berry SE, Valdes AM, Drew DA, et al. Human postprandial responses to food and potential for precision nutrition. *Nature Medicine*. 2020;26:964–973. doi:10.1038/s41591-020-0934-0
- Shen Y, Choi E, Kleinberg S. Predicting postprandial glycemic responses with limited data in type 1 and type 2 diabetes. *Journal of Diabetes Science and Technology*. 2026;20(4):1252–1265 (online 2025).
- Das A, Kerr D, Glanz N, Bevier W, Santiago R, Gutierrez-Osuna R, Mortazavi B. CGMacros: a scientific dataset for personalized nutrition and diet monitoring. *Scientific Data*.
- Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. CGMacros (version 1.0.0). PhysioNet. 2025. doi:10.13026/3z8q-x658
