import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


MODEL_PATH = "models/fraud_model.pkl"

FEATURES = [
    "amount",
    "transaction_hour",
    "customer_avg_transaction",
    "previous_transaction_count",
    "is_international",
    "is_new_device",
    "location_changed",
]


def generate_dataset(n_samples=5000, random_state=42):
    """
    Generate a synthetic banking transaction dataset.
    """

    np.random.seed(random_state)

    amounts = np.random.lognormal(
        mean=7.0,
        sigma=1.0,
        size=n_samples
    )

    transaction_hour = np.random.randint(
        0, 24, n_samples
    )

    customer_avg = np.random.lognormal(
        mean=6.5,
        sigma=0.6,
        size=n_samples
    )

    previous_transactions = np.random.randint(
        5, 200, n_samples
    )

    is_international = np.random.binomial(
        1, 0.15, n_samples
    )

    is_new_device = np.random.binomial(
        1, 0.20, n_samples
    )

    location_changed = np.random.binomial(
        1, 0.18, n_samples
    )

    # Amount relative to customer's normal spending
    amount_ratio = amounts / customer_avg

    # Generate fraud probability using realistic risk signals
    risk_score = (
        (amount_ratio > 4) * 2.5
        + (transaction_hour < 5) * 1.5
        + (is_international * 1.8)
        + (is_new_device * 1.5)
        + (location_changed * 1.7)
        + (amounts > 50000) * 1.5
    )

    probability = 1 / (1 + np.exp(-(risk_score - 4)))

    is_fraud = np.random.binomial(
        1,
        probability
    )

    df = pd.DataFrame({
        "amount": amounts.round(2),
        "transaction_hour": transaction_hour,
        "customer_avg_transaction": customer_avg.round(2),
        "previous_transaction_count": previous_transactions,
        "is_international": is_international,
        "is_new_device": is_new_device,
        "location_changed": location_changed,
        "is_fraud": is_fraud
    })

    return df


def train_model(df):
    """
    Train a Random Forest fraud detection model.
    """

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Fraud Detection Model")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model


def save_model(model):
    """
    Save trained model to the models directory.
    """

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(f"\nModel saved to: {MODEL_PATH}")


def load_model():
    """
    Load the trained fraud detection model.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Fraud model not found. Train the model first."
        )

    return joblib.load(MODEL_PATH)


def predict_fraud(transaction):
    """
    Predict fraud probability for a single transaction.

    Parameters
    ----------
    transaction : dict
        Transaction information.

    Returns
    -------
    dict
        Fraud probability and risk level.
    """

    model = load_model()

    input_data = pd.DataFrame(
        [transaction],
        columns=FEATURES
    )

    probability = model.predict_proba(
        input_data
    )[0][1]

    if probability >= 0.70:
        risk_level = "HIGH"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": probability,
        "risk_level": risk_level
    }


if __name__ == "__main__":

    print("Generating synthetic banking dataset...")

    dataset = generate_dataset()

    os.makedirs("dataset", exist_ok=True)

    dataset.to_csv(
        "dataset/transactions.csv",
        index=False
    )

    print(
        f"Dataset created with {len(dataset)} transactions."
    )

    print("\nTraining fraud detection model...")

    model = train_model(dataset)

    save_model(model)
    