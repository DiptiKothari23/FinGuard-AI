from src.fraud_detection import predict_fraud


def test_fraud_prediction():

    transaction = {
        "amount": 75000,
        "transaction_hour": 3,
        "customer_avg_transaction": 1200,
        "previous_transaction_count": 50,
        "is_international": 1,
        "is_new_device": 1,
        "location_changed": 1
    }

    result = predict_fraud(transaction)

    assert "fraud_probability" in result
    assert "risk_level" in result

    assert 0 <= result["fraud_probability"] <= 1
    assert result["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]