"""
Streamlit dashboard for the Credit Risk Modelling project.

This public-facing version presents the project as a professional credit-risk
dashboard. It does not expose raw data, processed applicant-level data, or
row-level validation predictions.
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
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: Path):
    """
    Loads the saved credit-risk model if it exists.
    """
    if not model_path.exists():
        return None

    return joblib.load(model_path)


def prepare_model_input(model, input_features: dict) -> pd.DataFrame:
    """
    Converts form inputs into a model-ready one-row DataFrame.
    """
    input_df = pd.DataFrame([input_features])

    if hasattr(model, "feature_name_"):
        input_df = input_df.reindex(columns=model.feature_name_, fill_value=0)

    return input_df


def add_custom_css() -> None:
    """
    Adds light dashboard styling to improve the public presentation.
    """
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .dashboard-hero {
                padding: 1.5rem 1.7rem;
                border-radius: 16px;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
                color: white;
                margin-bottom: 1.2rem;
            }
            .dashboard-hero h1 {
                color: white;
                font-size: 2.35rem;
                margin-bottom: 0.4rem;
            }
            .dashboard-hero p {
                color: #e2e8f0;
                font-size: 1.02rem;
                margin-bottom: 0rem;
            }
            .insight-card {
                padding: 1rem 1.2rem;
                border-radius: 14px;
                border: 1px solid #e5e7eb;
                background: #ffffff;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
                margin-bottom: 0.8rem;
            }
            .small-note {
                color: #64748b;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_header() -> None:
    """
    Displays the dashboard header.
    """
    st.markdown(
        """
        <div class="dashboard-hero">
            <h1>Credit Risk Modelling Dashboard</h1>
            <p>
                A portfolio credit-risk system that estimates probability of default,
                assigns risk bands and supports underwriting-style decision strategy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_project_summary() -> None:
    """
    Displays headline project results without exposing row-level data.
    """
    st.subheader("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Final Model", "LightGBM")
    col2.metric("Validation ROC-AUC", "0.7900")
    col3.metric("Baseline ROC-AUC", "0.7755")
    col4.metric("Validation Applicants", "61,503")

    st.markdown(
        """
        <div class="insight-card">
            The project uses a full credit-risk workflow: data understanding, feature engineering,
            model training, validation, explainability and decision strategy. The final LightGBM
            model outperformed the Logistic Regression baseline and was then converted into a
            business-facing risk framework.
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_business_summary() -> None:
    """
    Displays aggregated business strategy results.
    """
    st.subheader("Decision Strategy Summary")

    summary_df = pd.DataFrame(
        {
            "Decision": ["Approve", "Manual Review", "Decline"],
            "Risk Band": ["Low Risk", "Medium Risk", "High Risk"],
            "Applicant Share": ["9.90%", "38.02%", "52.08%"],
            "Observed Default Rate": ["1.00%", "2.53%", "13.46%"],
            "Interpretation": [
                "Lowest-risk applicants",
                "Borderline cases requiring review",
                "Highest-risk applicants",
            ],
        }
    )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Approved Group Default Rate", "1.00%")
    col2.metric("Manual Review Default Rate", "2.53%")
    col3.metric("Declined Group Default Rate", "13.46%")

    st.markdown(
        """
        <div class="insight-card">
            The selected threshold strategy is deliberately conservative. It approves a smaller
            low-risk group, routes medium-risk applicants to manual review and assigns the highest
            predicted-risk applicants to decline. In the validation analysis, the declined group
            contained the majority of observed defaults, while the approved group had a much lower
            observed default rate.
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_threshold_simulator() -> None:
    """
    Displays a public-safe approval threshold simulator using aggregated validation results.
    """
    st.subheader("Approval Threshold Simulator")

    st.write(
        "Adjust the low-risk and high-risk thresholds to see how the approval, manual review "
        "and decline mix would change. This simulator uses aggregated validation results from "
        "the project and is designed to show the business trade-off between growth and risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        low_threshold = st.slider(
            "Approve threshold",
            min_value=0.05,
            max_value=0.20,
            value=0.10,
            step=0.01,
            help="Applicants below this predicted probability of default are treated as low risk.",
        )

    with col2:
        high_threshold = st.slider(
            "Decline threshold",
            min_value=0.20,
            max_value=0.50,
            value=0.30,
            step=0.01,
            help="Applicants at or above this predicted probability of default are treated as high risk.",
        )

    if low_threshold >= high_threshold:
        st.warning("The approve threshold must be lower than the decline threshold.")
        return

    known_strategies = {
        (0.05, 0.20): {
            "approval_rate": 0.0272,
            "manual_review_rate": 0.2254,
            "decline_rate": 0.7474,
            "approved_default_rate": 0.0042,
            "manual_review_default_rate": 0.0177,
            "declined_default_rate": 0.1005,
        },
        (0.05, 0.25): {
            "approval_rate": 0.0272,
            "manual_review_rate": 0.3067,
            "decline_rate": 0.6661,
            "approved_default_rate": 0.0042,
            "manual_review_default_rate": 0.0216,
            "declined_default_rate": 0.1084,
        },
        (0.10, 0.25): {
            "approval_rate": 0.0990,
            "manual_review_rate": 0.2349,
            "decline_rate": 0.6661,
            "approved_default_rate": 0.0100,
            "manual_review_default_rate": 0.0251,
            "declined_default_rate": 0.1084,
        },
        (0.10, 0.30): {
            "approval_rate": 0.0990,
            "manual_review_rate": 0.3802,
            "decline_rate": 0.5208,
            "approved_default_rate": 0.0100,
            "manual_review_default_rate": 0.0253,
            "declined_default_rate": 0.1346,
        },
        (0.15, 0.35): {
            "approval_rate": 0.1888,
            "manual_review_rate": 0.4341,
            "decline_rate": 0.3771,
            "approved_default_rate": 0.0162,
            "manual_review_default_rate": 0.0353,
            "declined_default_rate": 0.1550,
        },
        (0.20, 0.40): {
            "approval_rate": 0.2526,
            "manual_review_rate": 0.4612,
            "decline_rate": 0.2862,
            "approved_default_rate": 0.0192,
            "manual_review_default_rate": 0.0467,
            "declined_default_rate": 0.1706,
        },
    }

    selected_key = min(
        known_strategies,
        key=lambda thresholds: abs(thresholds[0] - low_threshold) + abs(thresholds[1] - high_threshold),
    )

    strategy = known_strategies[selected_key]

    if selected_key != (round(low_threshold, 2), round(high_threshold, 2)):
        st.caption(
            f"Using the closest validated strategy from the project: "
            f"approve threshold {selected_key[0]:.2f}, decline threshold {selected_key[1]:.2f}."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Approval Rate", f"{strategy['approval_rate']:.2%}")
    col2.metric("Manual Review Rate", f"{strategy['manual_review_rate']:.2%}")
    col3.metric("Decline Rate", f"{strategy['decline_rate']:.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Approved Default Rate", f"{strategy['approved_default_rate']:.2%}")
    col5.metric("Review Default Rate", f"{strategy['manual_review_default_rate']:.2%}")
    col6.metric("Declined Default Rate", f"{strategy['declined_default_rate']:.2%}")

    strategy_df = pd.DataFrame(
        {
            "Decision Group": ["Approve", "Manual Review", "Decline"],
            "Portfolio Share": [
                strategy["approval_rate"],
                strategy["manual_review_rate"],
                strategy["decline_rate"],
            ],
            "Observed Default Rate": [
                strategy["approved_default_rate"],
                strategy["manual_review_default_rate"],
                strategy["declined_default_rate"],
            ],
        }
    )

    st.dataframe(
        strategy_df.style.format(
            {
                "Portfolio Share": "{:.2%}",
                "Observed Default Rate": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="insight-card">
            Lower approval thresholds reduce risk but restrict growth. Higher approval thresholds
            increase the approved population, but they also allow more higher-risk applicants into
            the portfolio. This is the type of trade-off a credit-risk team would calibrate using
            profitability, affordability, regulation and portfolio risk appetite.
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_manual_scoring(model) -> None:
    """
    Displays a simplified public scoring form.
    """
    st.subheader("Applicant Risk Scoring")

    st.write(
        "Enter applicant details to estimate probability of default and generate an "
        "underwriting-style recommendation. The form is simplified for public demonstration."
    )

    with st.form("manual_scoring_form"):
        col1, col2 = st.columns(2)

        with col1:
            income = st.number_input(
                "Annual income",
                min_value=0.0,
                value=135000.0,
                step=5000.0,
            )

            credit_amount = st.number_input(
                "Credit amount",
                min_value=0.0,
                value=568800.0,
                step=10000.0,
            )

            annuity = st.number_input(
                "Annuity amount",
                min_value=0.0,
                value=20560.5,
                step=1000.0,
            )

            goods_price = st.number_input(
                "Goods price",
                min_value=0.0,
                value=450000.0,
                step=10000.0,
            )

        with col2:
            age_years = st.number_input(
                "Age in years",
                min_value=18.0,
                max_value=100.0,
                value=35.0,
                step=1.0,
            )

            employment_years = st.number_input(
                "Employment years",
                min_value=0.0,
                max_value=60.0,
                value=5.0,
                step=1.0,
            )

            ext_source_2 = st.slider(
                "External risk score 2",
                min_value=0.0,
                max_value=1.0,
                value=0.62,
                step=0.01,
            )

            ext_source_3 = st.slider(
                "External risk score 3",
                min_value=0.0,
                max_value=1.0,
                value=0.48,
                step=0.01,
            )

        submitted = st.form_submit_button("Score Applicant")

    if not submitted:
        return

    if model is None:
        st.error("The model is not available. Please add the trained model file before scoring.")
        return

    credit_to_income = credit_amount / income if income > 0 else 0
    annuity_to_income = annuity / income if income > 0 else 0
    annuity_to_credit = annuity / credit_amount if credit_amount > 0 else 0
    goods_price_to_credit = goods_price / credit_amount if credit_amount > 0 else 0

    input_features = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "DAYS_BIRTH": -age_years * 365.25,
        "DAYS_EMPLOYED": -employment_years * 365.25,
        "APP_AGE_YEARS": age_years,
        "APP_EMPLOYED_YEARS": employment_years,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "APP_CREDIT_TO_INCOME_RATIO": credit_to_income,
        "APP_ANNUITY_TO_INCOME_RATIO": annuity_to_income,
        "APP_ANNUITY_TO_CREDIT_RATIO": annuity_to_credit,
        "APP_GOODS_PRICE_TO_CREDIT_RATIO": goods_price_to_credit,
    }

    try:
        input_df = prepare_model_input(model, input_features)
        probability = float(model.predict_proba(input_df)[0, 1])

        risk_band = assign_risk_band(probability)
        decision = assign_credit_decision(probability)

        st.subheader("Scoring Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Probability of Default", f"{probability:.2%}")
        col2.metric("Risk Band", risk_band)
        col3.metric("Recommendation", decision)

        result_df = pd.DataFrame(
            {
                "Metric": [
                    "Probability of Default",
                    "Risk Band",
                    "Recommended Decision",
                    "Credit-to-Income Ratio",
                    "Annuity-to-Income Ratio",
                    "Annuity-to-Credit Ratio",
                ],
                "Value": [
                    f"{probability:.4f}",
                    risk_band,
                    decision,
                    f"{credit_to_income:.4f}",
                    f"{annuity_to_income:.4f}",
                    f"{annuity_to_credit:.4f}",
                ],
            }
        )

        st.dataframe(result_df, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error(f"Scoring failed: {error}")


def display_reports() -> None:
    """
    Displays links to generated PDF reports if available.
    """
    st.subheader("Project Reports")

    st.write(
        "The reports summarise the modelling workflow, validation performance, "
        "business strategy and interpretation of results."
    )

    model_report_path = REPORTS_DIR / "model_report.pdf"
    business_report_path = REPORTS_DIR / "business_summary.pdf"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Report")
        if model_report_path.exists():
            with open(model_report_path, "rb") as file:
                st.download_button(
                    label="Download Model Report",
                    data=file,
                    file_name="model_report.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning("model_report.pdf is not currently available in the reports folder.")

    with col2:
        st.markdown("### Business Summary")
        if business_report_path.exists():
            with open(business_report_path, "rb") as file:
                st.download_button(
                    label="Download Business Summary",
                    data=file,
                    file_name="business_summary.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning("business_summary.pdf is not currently available in the reports folder.")


def display_data_notice() -> None:
    """
    Displays the public data-sharing notice.
    """
    st.subheader("Data and Usage Notice")

    st.write(
        "The raw Home Credit dataset is not included or redistributed in this app. "
        "To reproduce the full project, users should download the dataset directly "
        "from Kaggle after accepting the competition rules."
    )

    st.write(
        "This public app does not expose raw CSV files, processed applicant-level "
        "datasets, validation prediction rows, applicant IDs or row-level default labels."
    )

    st.write(
        "The scoring form is a simplified demonstration interface. A production credit "
        "decision system would require a complete feature pipeline, compliance review, "
        "fairness testing, affordability checks, monitoring and governance."
    )


def main() -> None:
    """
    Runs the public Streamlit dashboard.
    """
    add_custom_css()
    display_header()

    model = load_model(MODEL_PATH)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Threshold Simulator",
            "Score Applicant",
            "Reports",
            "Data Notice",
        ]
    )

    with tab1:
        display_project_summary()
        display_business_summary()

    with tab2:
        display_threshold_simulator()

    with tab3:
        display_manual_scoring(model)

    with tab4:
        display_reports()

    with tab5:
        display_data_notice()


if __name__ == "__main__":
    main()  