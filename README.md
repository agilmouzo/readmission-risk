# Readmission Risk — 30-day hospital readmission for diabetic patients

> Educational project. **Not a clinical tool.**

End-to-end machine learning project: data exploration, a gradient-boosted model,
explainability (SHAP), subgroup analysis, and a containerised prediction API.

## Dataset
[Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296), UCI Machine
Learning Repository: ~100k hospital encounters, de-identified. The data is not committed;
it is downloaded by a script.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
pytest
python -m readmission.train      # compares models with patient-level CV, saves models/model.joblib
python -m readmission.evaluate   # opens the test set ONCE; writes tables and figures to reports/
python -m readmission.explain    # SHAP explanations on a sample of the training set
```

## Project status
- [x] Repo skeleton, data download, CI
- [x] EDA
- [x] Cleaning and features
- [x] Modelling (baseline + LightGBM)
- [x] Evaluation (AUC-PR, calibration, threshold, subgroups)
- [x] Explainability (SHAP)
- [ ] API (FastAPI) + Docker
- [ ] Final README: results, limitations

## Test-set results
Hold-out test set: 19,802 encounters from patients never seen in training. 95% intervals come from
resampling patients, not rows.

| Metric | Value (95% CI) |
|---|---|
| AUC-ROC | 0.676 (0.660–0.690) |
| AUC-PR (random = 0.112) | 0.234 (0.211–0.257) |
| Brier score | 0.094 (0.090–0.098) |
| Readmission rate in the 10% highest-risk patients | 28.5% vs. 11.2% overall (lift 2.5×) |
| Share of all readmissions found in that 10% | 25% (top 30%: 51%) |

Operating points (threshold chosen on out-of-fold predictions over the training set):

| Rule | Patients flagged | Precision | Recall |
|---|---|---|---|
| Top 10% highest risk (threshold 0.197) | 9.7% | 28.6% | 24.9% |
| F1-optimal (threshold 0.135) | 26.2% | 20.1% | 47.1% |

Calibration is good (slope 1.06, expected calibration error 0.007): a predicted 20% risk
corresponds to about 20% of those patients being readmitted.

**Subgroups.** No meaningful difference by gender or between the two large race groups
(African American AUC-ROC 0.667 vs. Caucasian 0.675, overlapping intervals); smaller race groups
have too few cases to conclude anything. Discrimination is clearly lower in patients aged 80+
(0.632) than under 40 (0.777). Full tables and figures are in `reports/`.

![Precision-recall curve](reports/figures/pr_curve.png)
![Calibration](reports/figures/calibration.png)
![Risk deciles](reports/figures/risk_deciles.png)
![AUC-ROC by subgroup](reports/figures/subgroup_auc.png)

## What the model looks at (SHAP)
SHAP splits each prediction into one contribution per variable: starting from the average risk
(10%), every variable pushes the patient's risk up or down. Computed on 5,000 training patients;
contributions are in log-odds, so only their sign and relative size matter.

- The most influential variable is the **number of previous inpatient stays** (20% of the total
  average impact), followed by the **discharge destination** (17%) and the **total number of
  prior visits** (9%). Inpatient stays are part of that total, so prior hospital use as a whole
  weighs about 29%.
- Risk grows steadily with the number of previous inpatient stays and with the number of
  diagnoses, and is higher from the age of 60.
- These are associations learned from data, not causes. The insurance type (`payer_code`, 5%) is
  also used by the model and may act as a proxy for socioeconomic status.

![Most influential variables](reports/figures/shap_importance.png)
![Effect of the main numeric variables](reports/figures/shap_numeric.png)
![Two example patients](reports/figures/shap_patients.png)

## Design decisions
- Binary target: readmitted in < 30 days vs. everything else.
- Train/test split **by patient** (`patient_nbr`) to avoid leakage between encounters.
- `race` is excluded from the features and used only for subgroup performance analysis.

## Limitations
_To be completed._