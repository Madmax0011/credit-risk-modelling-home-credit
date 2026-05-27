# Credit Risk Modelling Platform

An end-to-end machine learning project for credit-risk modelling using the Home Credit Default Risk dataset. The project simulates how a fintech lender could use data science to support underwriting, risk segmentation, applicant-level explainability, and credit decision strategy.

The aim of this project is not only to train a predictive model, but to build a complete credit-risk workflow covering data understanding, feature engineering, model training, model evaluation, explainability, and business decision analysis.

---

## Project Overview

This project predicts the probability that a loan applicant may experience repayment difficulty. It uses multiple Home Credit data tables, including application data, bureau history, previous applications, instalment payments, POS cash balance records, and credit card balance history.

The final workflow includes:

- Exploratory data analysis
- Applicant-level feature engineering from multiple relational tables
- Missing-value handling and categorical encoding
- Logistic Regression baseline model
- LightGBM credit-risk model
- Model evaluation using ROC-AUC, Gini, KS statistic, precision, recall and F1-score
- Decile, gain and lift analysis
- SHAP-based model explainability
- Business decision strategy using approve, manual review and decline bands
- Expected loss estimation using PD, EAD and LGD
- FastAPI scoring service
- Streamlit dashboard for business-facing review

---

## Business Problem

Credit-risk modelling is central to lending decisions. A lender must balance growth and profitability against default risk. Approving too many high-risk applicants can increase losses, while rejecting too many low-risk applicants can reduce revenue and customer growth.

This project addresses the following business question:

> Can we use applicant, bureau, repayment and credit-history data to predict default risk and support more structured underwriting decisions?

The model output is converted into a practical decision framework:

| Risk Band | Probability of Default | Decision |
|---|---:|---|
| Low Risk | < 10% | Approve |
| Medium Risk | 10% to 30% | Manual Review |
| High Risk | > 30% | Decline |

These thresholds are adjustable and are analysed in the business strategy notebook.

---

## Dataset

Dataset used:

**Home Credit Default Risk**  
Source: Kaggle competition dataset

The dataset contains several related tables:

| Table | Description |
|---|---|
| `application_train.csv` | Main training applicant data with target variable |
| `application_test.csv` | Main test applicant data |
| `bureau.csv` | Previous credit bureau records |
| `bureau_balance.csv` | Monthly bureau balance history |
| `previous_application.csv` | Previous Home Credit loan applications |
| `POS_CASH_balance.csv` | Point-of-sale and cash loan balance records |
| `installments_payments.csv` | Repayment instalment history |
| `credit_card_balance.csv` | Credit card balance history |

The raw data is not included in this repository due to dataset size and Kaggle usage requirements. Users should download it from Kaggle and place the CSV files inside:

```text
data/raw/