def calculate_risk(
    income,
    expenses,
    existing_loan,
    credit_score,
    missed_payments
):

    debt_ratio = existing_loan / max(income, 1)

    expense_ratio = expenses / max(income, 1)

    score = 0

    if credit_score < 600:
        score += 30
    elif credit_score < 700:
        score += 15

    if debt_ratio > 0.5:
        score += 25
    elif debt_ratio > 0.3:
        score += 15

    if expense_ratio > 0.7:
        score += 25
    elif expense_ratio > 0.5:
        score += 15

    score += min(missed_payments * 5, 20)

    score = min(score, 100)

    if score >= 60:
        risk_level = "HIGH"
    elif score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "debt_ratio": debt_ratio,
        "expense_ratio": expense_ratio
    }