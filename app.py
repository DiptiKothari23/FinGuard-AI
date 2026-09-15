import streamlit as st
import pandas as pd
import numpy as np

from src.fraud_detection import predict_fraud
from src.credit_scoring import predict_credit
from src.risk_analysis import calculate_risk
from src.personalization import analyze_spending


st.set_page_config(
    page_title="FinGuard AI",
    page_icon="💳",
    layout="wide"
)


# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("FinGuard AI")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Fraud Detection",
        "Credit Decision",
        "Risk Management",
        "Personalized Banking"
    ]
)


# ==============================
# DASHBOARD
# ==============================

if page == "Dashboard":

    st.title("FinGuard AI")

    st.subheader(
        "AI-Powered Banking & Financial Analysis"
    )

    st.write(
        """
        FinGuard AI is a FinTech application that uses
        Artificial Intelligence and data analysis to support
        fraud detection, credit decisions, financial risk
        management and personalized banking.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "AI Modules",
        "4"
    )

    col2.metric(
        "Fraud Detection",
        "Active"
    )

    col3.metric(
        "Credit Analysis",
        "Active"
    )

    col4.metric(
        "Risk Analysis",
        "Active"
    )

    st.divider()

    st.subheader("Application Features")

    st.markdown(
        """
        **Fraud Detection**

        Detect potentially fraudulent banking transactions
        using a machine learning model.

        **Credit Decision**

        Analyze customer financial information and provide
        an AI-assisted loan approval recommendation.

        **Risk Management**

        Calculate financial risk using income, debt,
        expenses, credit score and repayment history.

        **Personalized Banking**

        Analyze spending patterns and generate personalized
        financial insights.
        """
    )


# ==============================
# FRAUD DETECTION
# ==============================

elif page == "Fraud Detection":

    st.title("AI Fraud Detection")

    st.write(
        "Enter transaction details to detect potential fraud."
    )

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=1.0,
            value=5000.0
        )

        transaction_hour = st.slider(
            "Transaction Hour",
            0,
            23,
            14
        )

        customer_avg = st.number_input(
            "Customer Average Transaction (₹)",
            min_value=1.0,
            value=3000.0
        )

        previous_transactions = st.number_input(
            "Previous Transactions",
            min_value=0,
            value=50
        )

    with col2:

        international = st.selectbox(
            "International Transaction",
            ["No", "Yes"]
        )

        new_device = st.selectbox(
            "New Device",
            ["No", "Yes"]
        )

        location_changed = st.selectbox(
            "Location Changed",
            ["No", "Yes"]
        )

    if st.button(
        "Analyze Transaction",
        type="primary"
    ):

        transaction = {
            "amount": amount,
            "transaction_hour": transaction_hour,
            "customer_avg_transaction": customer_avg,
            "previous_transaction_count":
                previous_transactions,
            "is_international":
                1 if international == "Yes" else 0,
            "is_new_device":
                1 if new_device == "Yes" else 0,
            "location_changed":
                1 if location_changed == "Yes" else 0
        }

        result = predict_fraud(transaction)

        probability = result["fraud_probability"]

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "Fraud Probability",
            f"{probability:.1%}"
        )

        col2.metric(
            "Risk Level",
            result["risk_level"]
        )

        if result["risk_level"] == "HIGH":

            st.error(
                "High-risk transaction detected."
            )

        elif result["risk_level"] == "MEDIUM":

            st.warning(
                "Transaction requires review."
            )

        else:

            st.success(
                "Transaction appears safe."
            )


# ==============================
# CREDIT DECISION
# ==============================

elif page == "Credit Decision":

    st.title("AI Credit Decision")

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Annual Income (₹)",
            min_value=0,
            value=700000
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=720
        )

        existing_loan = st.number_input(
            "Existing Loan (₹)",
            min_value=0,
            value=100000
        )

    with col2:

        monthly_emi = st.number_input(
            "Monthly EMI (₹)",
            min_value=0,
            value=10000
        )

        employment_years = st.number_input(
            "Employment Years",
            min_value=0,
            value=5
        )

        missed_payments = st.number_input(
            "Missed Payments",
            min_value=0,
            value=0
        )

    if st.button(
        "Evaluate Credit",
        type="primary"
    ):

        customer = {
            "income": income,
            "credit_score": credit_score,
            "existing_loan": existing_loan,
            "monthly_emi": monthly_emi,
            "employment_years": employment_years,
            "missed_payments": missed_payments
        }

        result = predict_credit(customer)

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Decision",
            result["decision"]
        )

        col2.metric(
            "Approval Probability",
            f"{result['approval_probability']:.1%}"
        )

        col3.metric(
            "Risk",
            result["risk"]
        )


# ==============================
# RISK MANAGEMENT
# ==============================

elif page == "Risk Management":

    st.title("Financial Risk Management")

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Annual Income (₹)",
            min_value=1,
            value=700000,
            key="risk_income"
        )

        expenses = st.number_input(
            "Annual Expenses (₹)",
            min_value=0,
            value=300000
        )

        existing_loan = st.number_input(
            "Existing Loan (₹)",
            min_value=0,
            value=150000,
            key="risk_loan"
        )

    with col2:

        credit_score = st.number_input(
            "Credit Score",
            300,
            850,
            720,
            key="risk_credit"
        )

        missed_payments = st.number_input(
            "Missed Payments",
            min_value=0,
            value=0,
            key="risk_missed"
        )

    if st.button(
        "Calculate Risk",
        type="primary"
    ):

        result = calculate_risk(
            income,
            expenses,
            existing_loan,
            credit_score,
            missed_payments
        )

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "Risk Score",
            f"{result['risk_score']}/100"
        )

        col2.metric(
            "Risk Level",
            result["risk_level"]
        )

        st.write(
            f"Debt Ratio: "
            f"{result['debt_ratio']:.1%}"
        )

        st.write(
            f"Expense Ratio: "
            f"{result['expense_ratio']:.1%}"
        )


# ==============================
# PERSONALIZED BANKING
# ==============================

elif page == "Personalized Banking":

    st.title("Personalized Banking")

    st.write(
        "Upload a transaction CSV to analyze spending patterns."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file:

        data = pd.read_csv(
            uploaded_file
        )

        if "amount" not in data.columns or \
           "category" not in data.columns:

            st.error(
                "CSV must contain 'amount' and 'category' columns."
            )

        else:

            categories, insights = analyze_spending(
                data
            )

            st.subheader("Spending Summary")

            st.dataframe(
                categories
            )

            st.subheader("AI Banking Insights")

            for insight in insights:

                st.info(insight)

    else:

        st.info(
            "Upload a CSV containing transaction data."
        )