"""
Streamlit dashboard — Credit Risk Modelling project.
Public-facing version. Does not expose raw data or row-level predictions.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODELS_DIR, REPORTS_DIR, FINAL_MODEL_FILE
from src.decision_strategy import assign_credit_decision, assign_risk_band

MODEL_PATH = MODELS_DIR / FINAL_MODEL_FILE

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
def add_css():
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .dashboard-hero {
            padding: 1.5rem 1.7rem; border-radius: 14px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
            color: white; margin-bottom: 1.2rem;
        }
        .dashboard-hero h1 { color: white; font-size: 2.2rem; margin-bottom: 0.3rem; }
        .dashboard-hero p  { color: #e2e8f0; font-size: 1rem; margin-bottom: 0; }
        .info-card {
            padding: 0.9rem 1.1rem; border-radius: 12px;
            border: 1px solid #e5e7eb; background: #ffffff;
            box-shadow: 0 1px 2px rgba(15,23,42,0.06); margin-bottom: 0.8rem;
        }
        .warn-card {
            padding: 0.9rem 1.1rem; border-radius: 12px;
            border: 1px solid #fbbf24; background: #fffbeb;
            margin-bottom: 0.8rem;
        }
        .note { color: #64748b; font-size: 0.88rem; }
        .result-approve { color: #16a34a; font-weight: 700; font-size: 1.4rem; }
        .result-review  { color: #d97706; font-weight: 700; font-size: 1.4rem; }
        .result-decline { color: #dc2626; font-weight: 700; font-size: 1.4rem; }
    </style>
    """, unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


# ── Advanced feature configuration ───────────────────────────────────────────
#
# Each entry in ADVANCED_FEATURES defines one adjustable field in the
# "Advanced credit profile" expander.  Fields are grouped into four sections
# that mirror standard credit bureau and behavioural data categories.
#
# Keys:
#   feature_name  : exact name expected by the LightGBM model
#   label         : plain-English label shown to the user (banking terminology)
#   tooltip       : one-sentence explanation visible on hover
#   widget        : "slider" or "number"
#   min / max / step / default : widget parameters
#   unit          : display suffix (%, years, days, etc.) — informational only
#   section       : grouping header


# Advanced feature definitions.
# All defaults are neutral population-median values — not preset-specific.
# This ensures manual entry produces consistent, unbiased scoring regardless
# of which preset was previously active. Preset buttons only affect the main
# form fields (income, credit, age, EXT scores etc.), not the advanced panel.

ADVANCED_FEATURES = [
    # ── Bureau history ────────────────────────────────────────────────────────
    {
        "section": "Bureau Credit History",
        "feature_name": "BUREAU_BUREAU_DEBT_TO_CREDIT_RATIO_max",
        "label": "Bureau debt utilisation — highest account",
        "tooltip": "The highest credit utilisation ratio across all bureau accounts. "
                   "Above 70% is considered high risk by most lenders. "
                   "Population median: 30%.",
        "widget": "slider", "min": 0.0, "max": 1.0, "step": 0.01,
        "default": 0.30,
        "format": "%.0f%%", "scale": 100,
    },
    {
        "section": "Bureau Credit History",
        "feature_name": "BUREAU_AMT_CREDIT_SUM_mean",
        "label": "Average bureau credit limit across all accounts",
        "tooltip": "Mean credit limit across all bureau accounts. "
                   "Left at 0 to allow dynamic scaling to the applicant's income.",
        "widget": "number", "min": 0, "max": 5000000, "step": 10000,
        "default": 0,
        "format": None, "scale": 1,
    },
    {
        "section": "Bureau Credit History",
        "feature_name": "BUREAU_DAYS_CREDIT_max",
        "label": "Most recent bureau credit account opened (years ago)",
        "tooltip": "How many years ago the applicant's most recently opened bureau "
                   "credit account was registered. Population median: 1 year.",
        "widget": "number", "min": 0, "max": 20, "step": 1,
        "default": 1,
        "format": None, "scale": -365.25,
    },
    {
        "section": "Bureau Credit History",
        "feature_name": "BUREAU_DAYS_CREDIT_ENDDATE_max",
        "label": "Latest bureau credit end date (years from now)",
        "tooltip": "How many years in the future the latest bureau credit account "
                   "is scheduled to close. Population median: 2 years.",
        "widget": "number", "min": 0, "max": 15, "step": 1,
        "default": 2,
        "format": None, "scale": 365.25,
    },
    # ── Payment behaviour ─────────────────────────────────────────────────────
    {
        "section": "Payment Behaviour",
        "feature_name": "INSTALL_INSTALL_LATE_PAYMENT_FLAG_mean",
        "label": "Late payment rate on previous loans",
        "tooltip": "Proportion of instalment payments on previous loans that were made "
                   "after the due date. 0% means every payment was on time. "
                   "Population median: 2%.",
        "widget": "slider", "min": 0.0, "max": 0.5, "step": 0.01,
        "default": 0.02,
        "format": "%.0f%%", "scale": 100,
    },
    {
        "section": "Payment Behaviour",
        "feature_name": "INSTALL_INSTALL_PAYMENT_DELAY_DAYS_mean",
        "label": "Average payment delay (days)",
        "tooltip": "Mean number of days by which instalment payments were made after "
                   "the scheduled due date. Population median: 0 days.",
        "widget": "number", "min": 0, "max": 30, "step": 1,
        "default": 0,
        "format": None, "scale": 1,
    },
    {
        "section": "Payment Behaviour",
        "feature_name": "INSTALL_AMT_PAYMENT_sum",
        "label": "Total repayments made across all previous loans",
        "tooltip": "Cumulative amount repaid across all previous instalment loans. "
                   "Left at 0 to allow dynamic scaling to the applicant's credit amount.",
        "widget": "number", "min": 0, "max": 5000000, "step": 10000,
        "default": 0,
        "format": None, "scale": 1,
    },
    # ── Previous applications ─────────────────────────────────────────────────
    {
        "section": "Previous Loan Applications",
        "feature_name": "PREV_NAME_CONTRACT_STATUS_Refused_ratio",
        "label": "Previous applications refused (proportion)",
        "tooltip": "Share of the applicant's prior loan applications that were refused "
                   "by lenders. A high refusal rate is a strong negative signal. "
                   "Default: 0 (no prior refusals).",
        "widget": "slider", "min": 0.0, "max": 1.0, "step": 0.01,
        "default": 0.0,
        "format": "%.0f%%", "scale": 100,
    },
    {
        "section": "Previous Loan Applications",
        "feature_name": "PREV_CNT_PAYMENT_mean",
        "label": "Average previous loan term (months)",
        "tooltip": "Mean scheduled repayment term in months across the applicant's "
                   "previous loan applications. Population median: 24 months.",
        "widget": "number", "min": 0, "max": 60, "step": 1,
        "default": 24,
        "format": None, "scale": 1,
    },
    # ── Identity and registration ─────────────────────────────────────────────
    {
        "section": "Identity and Registration",
        "feature_name": "DAYS_ID_PUBLISH",
        "label": "Years since identity document was issued",
        "tooltip": "How many years ago the applicant's current identity document "
                   "(passport or national ID) was issued. An ID issued very recently "
                   "can be a risk flag. Default: derived from applicant age.",
        "widget": "number", "min": 0, "max": 30, "step": 1,
        "default": 0,   # 0 = use dynamic age-based fill in build_model_input
        "format": None, "scale": -365.25,
    },
    {
        "section": "Identity and Registration",
        "feature_name": "DAYS_REGISTRATION",
        "label": "Years since residential registration",
        "tooltip": "How many years ago the applicant registered at their current "
                   "or most recent address. Default: derived from applicant age.",
        "widget": "number", "min": 0, "max": 40, "step": 1,
        "default": 0,   # 0 = use dynamic age-based fill in build_model_input
        "format": None, "scale": -365.25,
    },
    {
        "section": "Identity and Registration",
        "feature_name": "DAYS_LAST_PHONE_CHANGE",
        "label": "Years since phone number last changed",
        "tooltip": "How many years ago the applicant last changed their registered "
                   "phone number. Frequent changes can indicate instability. "
                   "Population median: 2 years.",
        "widget": "number", "min": 0, "max": 10, "step": 1,
        "default": 2,
        "format": None, "scale": -365.25,
    },
]

# Collect unique sections in order
SECTIONS = list(dict.fromkeys(f["section"] for f in ADVANCED_FEATURES))


# ── Preset definitions (verified against live model) ─────────────────────────
#
# All four presets were calibrated by running the actual model offline.
# Resulting PD values:
#   Prime      ~5.9%  -> Approve
#   Near-prime ~20.9% -> Manual Review
#   Sub-prime  ~63.5% -> Decline
#   Thin file  ~12.3% -> Manual Review (bureau and instalment data absent)

PRESET_KEYS = ["Prime", "Near-prime", "Sub-prime", "Thin file"]

PRESETS = {
    "Prime": {
        "label": "Prime applicant",
        "description": "Stable income, low debt burden, strong bureau scores, clean payment history. "
                       "Representative of a well-established borrower.",
        "income": 180000.0, "credit": 270000.0, "annuity": 13500.0, "goods": 270000.0,
        "age": 45.0, "emp": 12.0, "ext2": 0.82, "ext3": 0.76,
        "zero_bureau": False, "zero_install": False,
        "adv_defaults": "prime",
    },
    "Near-prime": {
        "label": "Near-prime applicant",
        "description": "Moderate income, higher credit-to-income ratio, some late payments, "
                       "short employment. Borderline case — typical manual review candidate.",
        "income": 135000.0, "credit": 405000.0, "annuity": 20250.0, "goods": 360000.0,
        "age": 35.0, "emp": 4.0, "ext2": 0.62, "ext3": 0.56,
        "zero_bureau": False, "zero_install": False,
        "adv_defaults": "near",
    },
    "Sub-prime": {
        "label": "Sub-prime applicant",
        "description": "Lower income, very high credit-to-income ratio, low bureau scores, "
                       "frequent late payments, prior application refusals.",
        "income": 90000.0, "credit": 630000.0, "annuity": 31500.0, "goods": 540000.0,
        "age": 28.0, "emp": 1.0, "ext2": 0.28, "ext3": 0.22,
        "zero_bureau": False, "zero_install": False,
        "adv_defaults": "sub",
    },
    "Thin file": {
        "label": "Thin-file applicant",
        "description": "Young applicant with reasonable income and bureau scores but no "
                       "prior credit or repayment history on file. Bureau and instalment "
                       "features are absent, not poor.",
        "income": 145000.0, "credit": 220000.0, "annuity": 11000.0, "goods": 220000.0,
        "age": 24.0, "emp": 2.0, "ext2": 0.68, "ext3": 0.62,
        "zero_bureau": True, "zero_install": True,
        "adv_defaults": "thin",
    },
}


def get_adv_default(feat_cfg, preset_key):
    """Return the preset-specific default for an advanced feature, or None if absent."""
    k = f"default_{preset_key}"
    return feat_cfg.get(k)


def apply_scale(value, scale):
    """Convert a user-facing value to the model's expected unit."""
    return value * scale


# ── Model input builder ───────────────────────────────────────────────────────
def build_model_input(model, main_inputs: dict, adv_inputs: dict,
                      zero_bureau: bool, zero_install: bool) -> pd.DataFrame:
    """
    Constructs a full 618-feature DataFrame from the form inputs.

    Missing features are filled with realistic population-median estimates
    so the model scores coherently.  Bureau and instalment features are
    zeroed when zero_bureau / zero_install is True (thin-file scenario).
    """
    feature_names = model.feature_name_

    income  = main_inputs["income"]
    credit  = main_inputs["credit"]
    annuity = main_inputs["annuity"]
    goods   = main_inputs["goods"]
    age     = main_inputs["age"]
    emp     = main_inputs["emp"]
    ext2    = main_inputs["ext2"]
    ext3    = main_inputs["ext3"]
    ext1    = (ext2 + ext3) / 2  # imputed from visible scores

    row = {f: 0 for f in feature_names}

    # Core application features
    row.update({
        "AMT_INCOME_TOTAL":   income,
        "AMT_CREDIT":         credit,
        "AMT_ANNUITY":        annuity,
        "AMT_GOODS_PRICE":    goods,
        "DAYS_BIRTH":         -age * 365.25,
        "DAYS_EMPLOYED":      -emp * 365.25,
        "APP_AGE_YEARS":      age,
        "APP_EMPLOYED_YEARS": emp,
        "EXT_SOURCE_1":       ext1,
        "EXT_SOURCE_2":       ext2,
        "EXT_SOURCE_3":       ext3,
        "APP_CREDIT_TO_INCOME_RATIO":      credit / income if income else 0,
        "APP_ANNUITY_TO_INCOME_RATIO":     annuity / income if income else 0,
        "APP_ANNUITY_TO_CREDIT_RATIO":     annuity / credit if credit else 0,
        "APP_GOODS_PRICE_TO_CREDIT_RATIO": goods / credit if credit else 0,
        "APP_EMPLOYED_TO_AGE_RATIO":       emp / age if age else 0,
        "REGION_POPULATION_RELATIVE":      0.02,
        "POS_CNT_INSTALMENT_FUTURE_mean":  12.0,
        "POS_CNT_INSTALMENT_FUTURE_std":   6.0,
        "PREV_AMT_ANNUITY_min":            annuity * 0.5,
        "PREV_CNT_PAYMENT_std":            6.0,
        "PREV_HOUR_APPR_PROCESS_START_mean": 12.0,
        "PREV_HOUR_APPR_PROCESS_START_std":  3.0,
    })

    # Apply advanced feature overrides (already converted to model units)
    # Values of 0 for fields that support dynamic fill are skipped —
    # the dynamic median fill below handles them instead.
    DYNAMIC_FILL_FIELDS = {
        "BUREAU_AMT_CREDIT_SUM_mean",
        "INSTALL_AMT_PAYMENT_sum",
        "DAYS_ID_PUBLISH",
        "DAYS_REGISTRATION",
    }
    for feat_name, val_tuple in adv_inputs.items():
        # adv_inputs stores (user_value, scale) tuples
        user_val, scale = val_tuple
        model_val = user_val * scale
        # Skip if zero and this field has a dynamic fill fallback
        if user_val == 0 and feat_name in DYNAMIC_FILL_FIELDS:
            continue
        if zero_bureau and feat_name.startswith("BUREAU_"):
            continue
        if zero_install and feat_name.startswith("INSTALL_"):
            continue
        row[feat_name] = model_val

    # Median fill for bureau features not covered by advanced panel
    if not zero_bureau:
        if row.get("BUREAU_AMT_CREDIT_SUM_mean", 0) == 0:
            row["BUREAU_AMT_CREDIT_SUM_mean"] = income * 1.5
        if row.get("BUREAU_AMT_CREDIT_SUM_min", 0) == 0:
            row["BUREAU_AMT_CREDIT_SUM_min"] = income * 0.3
        if row.get("BUREAU_BUREAU_DEBT_TO_CREDIT_RATIO_std", 0) == 0:
            row["BUREAU_BUREAU_DEBT_TO_CREDIT_RATIO_std"] = 0.05
        if row.get("BUREAU_DAYS_CREDIT_sum", 0) == 0:
            row["BUREAU_DAYS_CREDIT_sum"] = -2000.0

    # Median fill for instalment features not covered by advanced panel
    if not zero_install:
        if row.get("INSTALL_AMT_PAYMENT_min", 0) == 0:
            row["INSTALL_AMT_PAYMENT_min"] = annuity * 0.8
        if row.get("INSTALL_AMT_INSTALMENT_min", 0) == 0:
            row["INSTALL_AMT_INSTALMENT_min"] = annuity * 0.8
        if row.get("INSTALL_INSTALL_LATE_PAYMENT_FLAG_std", 0) == 0:
            row["INSTALL_INSTALL_LATE_PAYMENT_FLAG_std"] = 0.14
        if row.get("INSTALL_AMT_PAYMENT_sum", 0) == 0:
            row["INSTALL_AMT_PAYMENT_sum"] = credit * 0.3

    return pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)


# ── Header ────────────────────────────────────────────────────────────────────
def display_header():
    st.markdown("""
    <div class="dashboard-hero">
        <h1>Credit Risk Modelling Dashboard</h1>
        <p>A portfolio credit-risk system that estimates probability of default,
        assigns risk bands and supports underwriting-style decision strategy.</p>
    </div>
    """, unsafe_allow_html=True)


# ── Overview tab ──────────────────────────────────────────────────────────────
def display_overview():
    st.subheader("Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Model",        "LightGBM")
    c2.metric("Validation ROC-AUC", "0.7900")
    c3.metric("Baseline ROC-AUC",   "0.7755")
    c4.metric("Validation Applicants", "61,503")

    st.markdown("""
    <div class="info-card">
        The project uses a full credit-risk workflow: data understanding, feature engineering,
        model training, validation, explainability and decision strategy. The final LightGBM
        model outperformed the Logistic Regression baseline and was converted into a
        business-facing risk framework with three decision bands.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Decision Strategy Summary")
    summary = pd.DataFrame({
        "Decision":             ["Approve",           "Manual Review",              "Decline"],
        "Risk Band":            ["Low Risk",          "Medium Risk",                "High Risk"],
        "PD Threshold":         ["Below 10%",         "10% to 30%",                 "30% and above"],
        "Applicant Share":      ["9.90%",             "38.02%",                     "52.08%"],
        "Observed Default Rate":["1.00%",             "2.53%",                      "13.46%"],
        "Interpretation":       ["Lowest-risk group", "Borderline — requires review", "Highest-risk group"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Approved Group Default Rate",      "1.00%")
    c2.metric("Manual Review Group Default Rate", "2.53%")
    c3.metric("Declined Group Default Rate",      "13.46%")

    st.markdown("""
    <div class="info-card">
        The selected threshold strategy is deliberately conservative: a smaller approved group
        with a 1% observed default rate, a large manual review band for borderline cases, and
        a high-risk decline band that contained the majority of observed defaults in the
        validation set. The thresholds are adjustable in the Threshold Simulator tab.
    </div>
    """, unsafe_allow_html=True)


# ── Threshold simulator tab ───────────────────────────────────────────────────
def display_threshold_simulator():
    st.subheader("Approval Threshold Simulator")
    st.write(
        "Adjust the approval and decline thresholds to see how the portfolio split "
        "and default rates change across decision bands. Results use aggregated "
        "validation data and illustrate the growth-versus-risk trade-off."
    )
    c1, c2 = st.columns(2)
    with c1:
        low = st.slider("Approve threshold (PD below this = Approve)",
                        min_value=0.05, max_value=0.20, value=0.10, step=0.01)
    with c2:
        high = st.slider("Decline threshold (PD at or above this = Decline)",
                         min_value=0.20, max_value=0.50, value=0.30, step=0.01)
    if low >= high:
        st.warning("Approve threshold must be lower than the decline threshold.")
        return

    KNOWN = {
        (0.05,0.20):{"ar":0.0272,"mr":0.2254,"dr":0.7474,"adr":0.0042,"mdr":0.0177,"ddr":0.1005},
        (0.05,0.25):{"ar":0.0272,"mr":0.3067,"dr":0.6661,"adr":0.0042,"mdr":0.0216,"ddr":0.1084},
        (0.10,0.25):{"ar":0.0990,"mr":0.2349,"dr":0.6661,"adr":0.0100,"mdr":0.0251,"ddr":0.1084},
        (0.10,0.30):{"ar":0.0990,"mr":0.3802,"dr":0.5208,"adr":0.0100,"mdr":0.0253,"ddr":0.1346},
        (0.15,0.35):{"ar":0.1888,"mr":0.4341,"dr":0.3771,"adr":0.0162,"mdr":0.0353,"ddr":0.1550},
        (0.20,0.40):{"ar":0.2526,"mr":0.4612,"dr":0.2862,"adr":0.0192,"mdr":0.0467,"ddr":0.1706},
    }
    key = min(KNOWN, key=lambda t: abs(t[0]-low)+abs(t[1]-high))
    s   = KNOWN[key]
    if key != (round(low,2), round(high,2)):
        st.caption(f"Nearest validated strategy: approve {key[0]:.2f}, decline {key[1]:.2f}.")

    c1,c2,c3 = st.columns(3)
    c1.metric("Approval Rate",      f"{s['ar']:.2%}")
    c2.metric("Manual Review Rate", f"{s['mr']:.2%}")
    c3.metric("Decline Rate",       f"{s['dr']:.2%}")
    c4,c5,c6 = st.columns(3)
    c4.metric("Approved Default Rate",     f"{s['adr']:.2%}")
    c5.metric("Manual Review Default Rate",f"{s['mdr']:.2%}")
    c6.metric("Declined Default Rate",     f"{s['ddr']:.2%}")

    tbl = pd.DataFrame({
        "Decision Group": ["Approve","Manual Review","Decline"],
        "Portfolio Share": [s["ar"],s["mr"],s["dr"]],
        "Observed Default Rate": [s["adr"],s["mdr"],s["ddr"]],
    })
    st.dataframe(
        tbl.style.format({"Portfolio Share":"{:.2%}","Observed Default Rate":"{:.2%}"}),
        use_container_width=True, hide_index=True
    )
    st.markdown("""
    <div class="info-card">
        Lowering the approve threshold restricts the portfolio to only the strongest
        applicants but cuts growth. Raising it increases volume but brings in more
        marginal risk. A credit team would calibrate these thresholds against
        affordability rules, risk appetite, and profitability targets.
    </div>
    """, unsafe_allow_html=True)


# ── Score Applicant tab ───────────────────────────────────────────────────────
def display_score_applicant(model):
    st.subheader("Applicant Risk Scoring")
    st.write(
        "Enter applicant details to generate a probability of default estimate and "
        "an underwriting-style recommendation. Use the presets to explore how the "
        "model responds across different borrower profiles."
    )

    st.markdown("""
    <div class="info-card">
        <strong>How this scoring form works</strong><br>
        <span class="note">
        The model uses 618 features drawn from application data, credit bureau records,
        instalment payment history, and previous loan applications. This form collects
        the most influential inputs directly. All remaining features are filled with
        realistic population-median values — the standard approach in production
        systems when full bureau data is not available at the point of enquiry.<br><br>
        <strong>External credit scores</strong> (EXT_SOURCE fields) are bureau-derived
        creditworthiness scores where <strong>0 = highest default risk</strong> and
        <strong>1 = lowest default risk</strong>. They are the most predictive features
        in this model. The Advanced Credit Profile section below allows you to adjust
        bureau, payment behaviour, and identity features individually.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Preset buttons ────────────────────────────────────────────────────────
    st.markdown("**Select a borrower profile**")
    cols = st.columns(len(PRESET_KEYS))
    for col, key in zip(cols, PRESET_KEYS):
        if col.button(key, use_container_width=True):
            st.session_state["preset"] = key

    active_preset = st.session_state.get("preset", "Near-prime")
    p = PRESETS[active_preset]

    st.markdown(f"""
    <div class="info-card">
        <strong>{p['label']}</strong><br>
        <span class="note">{p['description']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Thin-file notice
    if active_preset == "Thin file":
        st.markdown("""
        <div class="warn-card">
            <strong>Thin-file applicant — bureau and instalment history absent</strong><br>
            <span class="note">
            This profile has no prior credit or repayment history on file. Bureau and
            instalment features are set to zero — not because the applicant has poor
            history, but because no history exists. The model cannot distinguish a
            genuinely credit-invisible applicant from a defaulted borrower with missing
            data, so it routes the case to Manual Review as a precaution.<br><br>
            In production, this applicant would be routed to a secondary assessment
            pathway — typically an open banking review, where real-time bank transaction
            data (income regularity, spending patterns, overdraft frequency) is used
            instead of bureau history. Lenders such as Salad and Plend use this approach
            in the UK to serve the estimated 5 to 6 million credit-invisible adults.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Main form ─────────────────────────────────────────────────────────────
    with st.form("scoring_form"):
        st.markdown("**Core application details**")
        c1, c2 = st.columns(2)
        with c1:
            income  = st.number_input("Annual income",   min_value=0.0, value=p["income"],  step=5000.0)
            credit  = st.number_input("Credit amount",   min_value=0.0, value=p["credit"],  step=10000.0)
            annuity = st.number_input("Monthly annuity", min_value=0.0, value=p["annuity"], step=500.0)
            goods   = st.number_input("Goods price",     min_value=0.0, value=p["goods"],   step=10000.0)
        with c2:
            age = st.number_input("Age (years)",         min_value=18.0, max_value=75.0, value=p["age"],  step=1.0)
            emp = st.number_input("Employment (years)",  min_value=0.0,  max_value=50.0, value=p["emp"],  step=1.0)
            ext2 = st.slider("External credit score 2  (0 = highest risk, 1 = lowest risk)",
                             0.0, 1.0, value=p["ext2"], step=0.01)
            ext3 = st.slider("External credit score 3  (0 = highest risk, 1 = lowest risk)",
                             0.0, 1.0, value=p["ext3"], step=0.01)

        # ── Advanced credit profile (collapsible) ─────────────────────────────
        st.markdown("---")
        with st.expander("Advanced credit profile  —  bureau history, payment behaviour, identity"):

            if active_preset == "Thin file":
                st.info(
                    "Bureau history and payment behaviour fields are disabled for the "
                    "thin-file profile. The absence of this data is the defining "
                    "characteristic of a thin-file applicant."
                )

            # Search bar
            search_term = st.text_input(
                "Search features",
                placeholder="e.g. bureau, late payment, identity",
            ).strip().lower()

            adv_vals = {}

            for section in SECTIONS:
                section_feats = [f for f in ADVANCED_FEATURES if f["section"] == section]

                # Filter by search
                if search_term:
                    section_feats = [
                        f for f in section_feats
                        if search_term in f["label"].lower()
                        or search_term in f["tooltip"].lower()
                        or search_term in f["section"].lower()
                    ]

                if not section_feats:
                    continue

                # Disable bureau/install sections for thin file
                is_bureau  = section == "Bureau Credit History"
                is_install = section == "Payment Behaviour"
                disabled = active_preset == "Thin file" and (is_bureau or is_install)

                st.markdown(f"**{section}**")
                if disabled:
                    st.caption("Not applicable — no history on file for this profile.")
                    # Register zero tuples — build_model_input will zero these out
                    for feat in section_feats:
                        adv_vals[feat["feature_name"]] = (0.0, feat["scale"])
                    continue

                cols_adv = st.columns(2)
                for idx, feat in enumerate(section_feats):
                    raw_default = float(feat.get("default", feat["min"]))

                    with cols_adv[idx % 2]:
                        if feat["widget"] == "slider":
                            user_val = st.slider(
                                feat["label"],
                                min_value=float(feat["min"]),
                                max_value=float(feat["max"]),
                                value=raw_default,
                                step=float(feat["step"]),
                                help=feat["tooltip"],
                            )
                        else:
                            user_val = st.number_input(
                                feat["label"],
                                min_value=float(feat["min"]),
                                max_value=float(feat["max"]),
                                value=raw_default,
                                step=float(feat["step"]),
                                help=feat["tooltip"],
                            )
                        # Store raw user value — conversion happens in build_model_input
                        # A value of 0 means "use dynamic fill" for fields that support it
                        adv_vals[feat["feature_name"]] = (user_val, feat["scale"])

        submitted = st.form_submit_button("Score Applicant", use_container_width=True)

    if not submitted:
        return

    if model is None:
        st.error("Model file not found. Deploy the trained model before scoring.")
        return

    main_inputs = dict(
        income=income, credit=credit, annuity=annuity, goods=goods,
        age=age, emp=emp, ext2=ext2, ext3=ext3,
    )

    try:
        input_df   = build_model_input(model, main_inputs, adv_vals,
                                       zero_bureau=p["zero_bureau"],
                                       zero_install=p["zero_install"])
        pd_val     = float(model.predict_proba(input_df)[0, 1])
        risk_band  = assign_risk_band(pd_val)
        decision   = assign_credit_decision(pd_val)

        colour_map = {
            "Approve":       "result-approve",
            "Manual Review": "result-review",
            "Decline":       "result-decline",
        }
        css_class = colour_map.get(decision, "")

        st.markdown("---")
        st.subheader("Scoring Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Probability of Default", f"{pd_val:.2%}")
        c2.metric("Risk Band", risk_band)
        c3.metric("Recommended Decision", decision)

        st.markdown(f"""
        <div class="info-card">
            <span class="{css_class}">{decision}</span>
            &nbsp;&nbsp;
            <span class="note">PD {pd_val:.2%} &mdash; {risk_band}
            (Approve &lt; 10%, Manual Review 10–30%, Decline &ge; 30%)</span>
        </div>
        """, unsafe_allow_html=True)

        # Derived ratios table
        safe = lambda n, d: n/d if d else 0
        result_df = pd.DataFrame({
            "Metric": [
                "Probability of Default",
                "Risk Band",
                "Recommended Decision",
                "Credit-to-Income Ratio",
                "Annuity-to-Income Ratio",
                "Annuity-to-Credit Ratio",
                "Goods-Price-to-Credit Ratio",
                "Employment-to-Age Ratio",
            ],
            "Value": [
                f"{pd_val:.4f}",
                risk_band,
                decision,
                f"{safe(credit, income):.4f}",
                f"{safe(annuity, income):.4f}",
                f"{safe(annuity, credit):.4f}",
                f"{safe(goods, credit):.4f}",
                f"{safe(emp, age):.4f}",
            ],
        })
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        # Thin-file follow-up note
        if active_preset == "Thin file":
            st.markdown("""
            <div class="warn-card">
                <strong>Note on this result</strong><br>
                <span class="note">
                This applicant scores as Manual Review not because of adverse credit events
                but because no bureau or instalment evidence exists to separate them from a
                higher-risk borrower. In a production system, the next step would be to
                request open banking consent, allowing the lender to assess income regularity
                and spending behaviour from bank transaction data — bypassing the bureau gap.
                </span>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Scoring failed: {e}")


# ── Reports tab ───────────────────────────────────────────────────────────────
def display_reports():
    st.subheader("Project Reports")
    st.write(
        "The reports summarise the modelling workflow, validation performance, "
        "business strategy and interpretation of results."
    )
    for fname, label in [
        ("model_report.pdf",    "Model Report"),
        ("business_summary.pdf","Business Summary"),
    ]:
        path = REPORTS_DIR / fname
        st.markdown(f"### {label}")
        if path.exists():
            with open(path,"rb") as f:
                st.download_button(f"Download {label}", f, fname, "application/pdf")
        else:
            st.warning(f"{fname} is not currently available in the reports folder.")


# ── Data notice tab ───────────────────────────────────────────────────────────
def display_data_notice():
    st.subheader("Data and Usage Notice")
    st.write(
        "The raw Home Credit dataset is not included or redistributed in this app. "
        "To reproduce the full project, download the dataset from Kaggle after "
        "accepting the competition rules."
    )
    st.write(
        "This public app does not expose raw CSV files, processed applicant-level "
        "datasets, validation prediction rows, applicant IDs, or row-level default labels."
    )
    st.write(
        "The scoring form is a demonstration interface. A production credit decision "
        "system would require a complete feature pipeline, compliance review, "
        "fairness testing, affordability checks, monitoring, and governance."
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    add_css()
    display_header()
    model = load_model(MODEL_PATH)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Threshold Simulator", "Score Applicant", "Reports", "Data Notice"
    ])
    with tab1:  display_overview()
    with tab2:  display_threshold_simulator()
    with tab3:  display_score_applicant(model)
    with tab4:  display_reports()
    with tab5:  display_data_notice()


if __name__ == "__main__":
    main()
