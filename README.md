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
```

## Project status
- [x] Repo skeleton, data download, CI
- [x] EDA
- [x] Cleaning and features
- [x] Modelling (baseline + LightGBM)
- [ ] Evaluation (AUC-PR, calibration, threshold, subgroups)
- [ ] Explainability (SHAP)
- [ ] API (FastAPI) + Docker
- [ ] Final README: results, limitations

## Design decisions
- Binary target: readmitted in < 30 days vs. everything else.
- Train/test split **by patient** (`patient_nbr`) to avoid leakage between encounters.
- `race` is excluded from the features and used only for subgroup performance analysis.

## Limitations
_To be completed._
