import pandas as pd
import shap

from src.fraud_detection import FEATURES, load_model


FEATURE_NAMES = {
    "amount": "Transaction Amount",
    "transaction_hour": "Transaction Hour",
    "customer_avg_transaction": "Customer Average Transaction",
    "previous_transaction_count": "Previous Transaction Count",
    "is_international": "International Transaction",
    "is_new_device": "New Device",
    "location_changed": "Location Changed"
}


def explain_fraud_prediction(transaction):
    """
    Explain the fraud prediction for a transaction using SHAP.
    """

    model = load_model()

    input_data = pd.DataFrame(
        [transaction],
        columns=FEATURES
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_data)

    # SHAP output can differ between SHAP versions.
    # We need the contribution for the fraud class.
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    explanations = []

    for feature, contribution in zip(FEATURES, values):

        explanations.append({
            "feature": FEATURE_NAMES[feature],
            "contribution": float(contribution),
            "impact": (
                "Increases fraud risk"
                if contribution > 0
                else "Decreases fraud risk"
            )
        })

    explanations.sort(
        key=lambda x: abs(x["contribution"]),
        reverse=True
    )

    return explanations