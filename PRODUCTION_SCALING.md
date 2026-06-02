# Production Scaling Notes

This document answers directly: *"How would this change at 100x the data
and with leaky, drifting features?"* — a standard question in credit DS
technical interviews.

---

## What changes at 100x scale

### Data layer

| This project (prototype) | Production at 100x |
|--------------------------|-------------------|
| Single CSV, ~300k rows | Distributed warehouse (BigQuery, Snowflake, Redshift) |
| Load once, train offline | Streaming ingestion (Kafka / Pub-Sub) |
| DuckDB local query | Spark / dbt for feature engineering |
| Full reload to retrain | Incremental feature materialisation |

At 30M+ applications per year, the bottleneck shifts from model training to
**feature computation** — joins across bureau, transaction, and behavioural
tables that span terabytes. The model itself (LightGBM, ~200 trees) still
trains in minutes; feature engineering takes hours unless managed carefully.

### Feature store
At scale, raw features computed ad-hoc at training time cause **training-serving
skew** — the single most common cause of silent model degradation in production.
The fix is an offline/online feature store (Feast, Tecton, or a custom DynamoDB
cache) that:
- Computes features once on a schedule
- Serves the same feature values at both training and real-time scoring
- Provides point-in-time correct lookups for historical training (prevents
  look-ahead leakage)

### Real-time vs batch scoring
This project scores in batch (Streamlit on-demand). Production typically needs
both:
- **Batch** (nightly): pre-score existing customers for line management,
  pre-approval campaigns
- **Real-time** (< 100ms): score applications at submission, fraud checks
  during transaction authorisation

Real-time serving requires a lightweight model wrapper (FastAPI + joblib/ONNX),
a feature retrieval path, and strict latency budgets.

---

## Feature leakage at scale

Leakage is significantly harder to catch at scale because:

1. **Temporal leakage** — features computed using data "from the future" relative
   to the loan decision date. Example: using a borrower's total revol_bal
   measured at end-of-year for a loan taken out in March. Fix: always use
   point-in-time lookups keyed on application date.

2. **Label leakage** — default indicator populated before the observation
   window closes. Example: including `recoveries > 0` as a feature (it's only
   non-zero *after* the loan has already charged off).

3. **Proxy leakage** — features that are downstream effects of the label.
   Example: `last_payment_date` or `payment_plan_flag` — these change *because*
   a loan is defaulting, not before. Standard check: null rate of the feature
   should not differ between training positives and negatives.

At prototype scale these are visible in feature importance (the leaky feature
dominates). At 100x they are subtler — requires a disciplined temporal
validation framework:

```
Training  |  Validation  |  Out-of-time test
2015–2017 |  2018 Q1–Q2  |  2018 Q3–Q4
```

OOT (out-of-time) AUC significantly below validation AUC is the first signal
of temporal leakage.

---

## Model drift at scale

This project includes `monitoring/model_monitoring.py`, which computes PSI,
AUC, Gini, KS and ECE across monthly windows. At scale, the same logic sits
inside a scheduled monitoring job (Airflow DAG or Vertex AI pipeline) that:

1. Scores the last 30 days of new applications using the current production model
2. Computes PSI for the score distribution and top 20 features
3. Compares AUC / Gini against the development baseline
4. Writes metrics to a dashboard (Grafana, Looker, or custom)
5. Triggers a Slack/email alert if PSI > 0.25 or AUC drops > 3 percentage points

**Concept drift vs covariate drift**

| Type | What shifts | Signal | Response |
|------|-------------|--------|----------|
| Covariate drift | Input feature distributions (e.g. DTI rising during recession) | High PSI on features | Investigate; may or may not affect model |
| Concept drift | Relationship between features and target (e.g. FICO less predictive post-2020) | AUC degradation | Retrain required |
| Label drift | Base default rate shifts | Rising default rate in monitoring windows | Recalibrate or retrain |

The monitoring script simulates all three: rising default rate (label drift),
score compression (concept drift), and rising utilisation (covariate drift).

---

## Retraining strategy

| Trigger | Action |
|---------|--------|
| PSI > 0.25 on score | Retrain with recent data; evaluate on OOT |
| AUC drops > 3 pp vs baseline | Retrain + feature investigation |
| Default rate rises > 20% relative | Recalibrate (temperature scaling); consider retrain |
| Scheduled (quarterly) | Champion/challenger: retrain on rolling 24-month window |

A champion/challenger framework runs the new model in shadow mode (scores
applications but doesn't make decisions) for 4–6 weeks, comparing AUC and
calibration before promoting to champion.

---

## MLOps stack (what this project would need in production)

| Concern | Tool options |
|---------|-------------|
| Pipeline orchestration | Airflow, Prefect, Vertex AI Pipelines |
| Feature store | Feast, Tecton, Redis + DynamoDB |
| Model registry | MLflow, Weights & Biases, Vertex AI Model Registry |
| Serving | FastAPI + joblib, BentoML, Vertex AI Prediction |
| Monitoring | Custom PSI/AUC pipeline (see `monitoring/`), Evidently AI |
| A/B testing | Internal framework (see `experiments/`), Statsig, Growthbook |
| CI/CD for models | GitHub Actions + pytest on model artefacts |

The monitoring and experimentation scripts in this project are simplified but
structurally equivalent to what any of these tools provide — they demonstrate
the concepts without the infrastructure dependency.

---

## What stays the same at 100x

- The model architecture (LightGBM with calibration is still the right choice
  for tabular credit data at any scale)
- The evaluation framework: Gini, KS, lift tables, calibration (ECE)
- The business logic: approve / refer / decline thresholds, P&L framing
- The experimentation approach: power calculation, z-test, sensitivity analysis
- The explainability layer: SHAP values are efficient even at very large N
  (use `shap.TreeExplainer` with `approximate=True` for speed)
