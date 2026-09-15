import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


MODEL_PATH = "models/credit_model.pkl"

FEATURES = [
    "income",
    "credit_score",
    "existing_loan",
    "monthly_emi",
    "employment_years",
    "missed_payments"
]


def generate_credit_data(n_samples=3000, random_state=42):

    np.random.seed(random_state)

    income = np.random.randint(200000, 2000000, n_samples)

    credit_score = np.random.randint(300, 850, n_samples)

    existing_loan = np.random.randint(0, 1000000, n_samples)

    monthly_emi = np.random.randint(0, 60000, n_samples)

    employment_years = np.random.randint(0, 25, n_samples)

    missed_payments = np.random.randint(0, 8, n_samples)

    approval_score = (
        (credit_score > 650) * 3
        + (income > 500000) * 2
        + (employment_years > 2) * 1
        - (missed_payments * 1.5)
        - ((existing_loan / np.maximum(income, 1)) > 0.5) * 2
        - ((monthly_emi / np.maximum(income / 12, 1)) > 0.5) * 2
    )

    approved = (approval_score >= 3).astype(int)

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "existing_loan": existing_loan,
        "monthly_emi": monthly_emi,
        "employment_years": employment_years,
        "missed_payments": missed_payments,
        "approved": approved
    })

    return df


def train_model(df):

    X = df[FEATURES]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Credit Model Accuracy: {accuracy:.2%}")

    return model


def save_model(model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )


def load_model():

    return joblib.load(MODEL_PATH)


def predict_credit(customer):

    model = load_model()

    data = pd.DataFrame(
        [customer],
        columns=FEATURES
    )

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        decision = "APPROVED"
        risk = "LOW"
    else:
        decision = "REJECTED"
        risk = "HIGH"

    return {
        "decision": decision,
        "approval_probability": probability,
        "risk": risk
    }


if __name__ == "__main__":

    data = generate_credit_data()

    data.to_csv(
        "dataset/customers.csv",
        index=False
    )

    model = train_model(data)

    save_model(model)

    print("Credit model saved.")