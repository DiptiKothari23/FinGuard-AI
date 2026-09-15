# FinGuard AI

## AI-Powered Banking & Financial Analysis Platform

FinGuard AI is a FinTech application that uses Artificial Intelligence and data analysis to support banking and financial activities.

The application provides four main modules:

- AI Fraud Detection
- AI Credit Decision
- Financial Risk Management
- Personalized Banking

## Features

### AI Fraud Detection
Uses a Random Forest machine learning model to estimate the probability that a banking transaction is fraudulent.

### AI Credit Decision
Analyzes customer financial information and provides an AI-assisted credit approval recommendation.

### Risk Management
Evaluates financial risk using income, expenses, existing loans, credit score and missed payments.

### Personalized Banking
Analyzes transaction history and identifies spending patterns and personalized financial insights.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly

## Project Structure

```text
FinGuard-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── fraud_detection.py
│   ├── credit_scoring.py
│   ├── risk_analysis.py
│   └── personalization.py
│
├── models/
│   ├── fraud_model.pkl
│   └── credit_model.pkl
│
├── dataset/
│   ├── transactions.csv
│   ├── customers.csv
│   └── spending.csv
│
└── tests/
|   |__ test_fraud.py

Installation

Create and activate a virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Running the Application

Run:

streamlit run app.py

The application will open in the browser.