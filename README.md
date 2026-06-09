# Credit Risk Modelling — Home Credit Default Risk

A full credit-risk modelling project built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset. The goal was to go beyond a notebook model and build something closer to how a lending team would actually use data science — from raw data through to a deployable scoring system with monitoring and business decision tools.

**Live demo:** https://credit-risk-modelling-home-credit.streamlit.app

---

## What this project does

The model predicts the probability that a loan applicant will default. That probability is then fed into a three-band decision framework — approve, manual review, or decline — with adjustable thresholds that can be tuned against risk appetite and expected loss targets.

The project covers:

- Exploratory analysis across eight relational tables
- Feature engineering from bureau history, instalment payments, POS cash records, credit card balances, and previous applications
- Logistic Regression baseline and LightGBM final model
- SHAP explainability — both global feature importance and single-applicant explanations
- Business decision strategy with expected loss analysis (PD × LGD × EAD)
- Model monitoring pipeline tracking PSI, AUC, Gini, KS and ECE across monthly windows
- Credit policy A/B test framework with power calculation, z-test and P&L analysis
- FastAPI scoring endpoint
- Streamlit dashboard with interactive threshold simulator and applicant scoring

---

## Results

| | LightGBM | Logistic Regression |
|---|---:|---:|
| ROC-AUC | 0.7900 | 0.7755 |
| Gini | 0.5800 | 0.5511 |
| KS statistic | 0.4423 | 0.4201 |

At the selected thresholds (approve below 10% PD, decline above 30%):

| Decision | Applicant share | Observed default rate |
|---|---:|---:|
| Approve | 9.9% | 1.0% |
| Manual Review | 38.0% | 2.5% |
| Decline | 52.1% | 13.5% |

Top decile lift: 3.84× — the highest-risk 10% of applicants account for 38.4% of all defaults.

The three most predictive features by SHAP value are EXT_SOURCE_2, EXT_SOURCE_3, and EXT_SOURCE_1 — bureau-derived creditworthiness scores where higher values indicate lower default risk.

---

## Project structure

```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_explainability.ipynb
│   └── 05_business_strategy.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── data_audit.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── explain_model.py
│   └── decision_strategy.py
├── api/
│   ├── main.py
│   └── schemas.py
├── dashboard/
│   └── streamlit_app.py
├── monitoring/
│   └── model_monitoring.py
├── experiments/
│   └── ab_test_credit_policy.py
├── reports/
├── models/
├── PRODUCTION_SCALING.md
└── requirements.txt
```

---

## Data

Download the dataset from Kaggle and place the CSV files in `data/raw/`. The raw data is not included in this repository.

```
data/raw/
├── application_train.csv
├── application_test.csv
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── POS_CASH_balance.csv
├── installments_payments.csv
└── credit_card_balance.csv
```

---

## Running the project

```bash
pip install -r requirements.txt

# Run notebooks 01 through 05 in order

# Dashboard
streamlit run dashboard/streamlit_app.py

# API
uvicorn api.main:app --reload

# Monitoring
python monitoring/model_monitoring.py

# A/B test
python experiments/ab_test_credit_policy.py
```

---

## Production notes

`PRODUCTION_SCALING.md` covers what would change at 100× data volume — distributed feature engineering, feature store design, real-time scoring latency requirements, drift detection, and the champion/challenger retraining framework.
