# Model card: 30-day readmission risk for diabetic patients

This card summarises what the model is, what it was trained on, how well it works and where it
must not be used. It follows the idea of a "model card": a short, honest description that travels
with the model. Details and figures are in the [README](README.md) and in `reports/`.

> Educational project. **Not a clinical tool.** It must not be used to make decisions about real
> patients.

## Model details
- **Task:** binary classification. For one hospital stay of a patient with diabetes, estimate the
  probability that the patient is readmitted to hospital within 30 days of discharge.
- **Model:** gradient-boosted decision trees (LightGBM, 300 trees, learning rate 0.03, 15 leaves,
  strong regularisation) inside a scikit-learn pipeline that also does the preprocessing. No class
  weights are used, so the probabilities are not artificially inflated.
- **Version:** `v0.1.0`, published as a GitHub Release asset (`model.joblib`, under 1 MB). The file is
  checked against a SHA-256 digest when downloaded, and `GET /health` reports which version is
  loaded. Random seed 42.
- **Inputs:** the information available when the patient leaves hospital: age, sex, how the
  patient was admitted and discharged, length of stay, counts of tests, procedures and
  medications, visits in the previous year, diagnoses (grouped from ICD-9 codes into broad
  groups), HbA1c and glucose results, diabetes medication, insurance type and medical specialty.
- **Output:** a probability between 0 and 1, and a flag `high_risk` that is true when the
  probability is above a fixed threshold (0.1971).
- **Owner:** Antonio Gil Mouzo, the author of the repository (GitHub: agilmouzo).
  Personal portfolio project, MIT licence.

## Intended use
- **In scope:** learning, demonstration, and discussion of how to build, evaluate, explain and
  serve a risk model carefully.
- **Out of scope:** any clinical, operational or administrative decision about real patients; use
  with data from other countries or health systems (including Spain's); use with patients that
  were discharged to death or hospice (the API rejects those codes); use before discharge.

## Training data
- **Source:** *Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning Repository
  (Clore, Cios, DeShazo and Strack, 2014; CC BY 4.0). 101,766 hospital stays from 71,518 patients
  in 130 US hospitals.
- **Cleaning:** stays that ended in death or hospice are removed (those patients cannot be
  readmitted). The data is not stored in this repository.
- **Target:** readmitted within 30 days (about 11% of stays). The data cannot tell whether a
  readmission was planned, avoidable or related to the care received.
- **Split:** by patient, so that stays of the same person never appear in both training and test.
  80% train, 20% test (19,802 stays).
- **Not used as inputs:** race (kept only to compare performance across groups), patient and
  encounter identifiers, and the weight, which is almost always missing.

## Performance
Test set, 95% confidence intervals from resampling patients:

| Metric | Value |
|---|---|
| AUC-ROC | 0.676 (0.660–0.690) |
| AUC-PR (random = 0.112) | 0.234 (0.211–0.257) |
| Brier score | 0.094 (0.090–0.098) |
| Readmission rate in the 10% highest-risk patients | 28.5% vs. 11.2% overall (lift 2.5x) |

At the high-risk threshold the model flags 9.7% of test patients, with precision 28.6% and recall
24.9%. Calibration is good for the group as a whole (slope 1.06, expected calibration error
0.007), but that does not make the figure for one patient equally precise. Compared with a
logistic regression the gain is small (about 0.01 AUC-ROC).

## Performance by subgroup
Discrimination is similar by gender and between the two large race groups (AUC-ROC 0.667 for
African American and 0.675 for Caucasian patients, overlapping intervals). It is clearly lower
for patients aged 80 and over (0.632) than for patients under 40 (0.777). Small groups have too
few cases to conclude anything. A gap in AUC alone does not prove unfair treatment, and a small gap
does not prove its absence. Full tables: `reports/subgroups.csv`.

## Limitations and risks
- Old, US-only data (1999-2008). Practice, coding and patient mix have changed. There is no
  temporal validation and no test on external hospitals.
- Modest predictive power: most readmissions are not flagged and most flags are false alarms. It
  ranks groups of patients better than it predicts individuals.
- Predicts at discharge only.
- Leaving race out does not remove proxies: insurance type and medical specialty can carry
  socioeconomic information.
- The explanations (SHAP) show associations learned from the data, not causes.
- Not tuned extensively. The API has no authentication or rate limit.

## Ethical considerations
The model predicts a health-related outcome and could, if misused, disadvantage groups of
patients, for example by age or by type of insurance. For that reason it is published only as an
educational example, with its limits stated next to the results and in the demo.

## How to cite and contact
Code and model: MIT licence, https://github.com/agilmouzo/readmission-risk. If you use the data, cite
the dataset (https://doi.org/10.24432/C5230J) and the original study (Strack et al., 2014,
*Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database
Patient Records*, BioMed Research International).
