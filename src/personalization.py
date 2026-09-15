import pandas as pd


def analyze_spending(data):

    total_spending = data["amount"].sum()

    category_spending = (
        data.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest_category = category_spending.index[0]

    highest_amount = category_spending.iloc[0]

    insights = []

    insights.append(
        f"Total spending: ₹{total_spending:,.2f}"
    )

    insights.append(
        f"Highest spending category: {highest_category}"
    )

    insights.append(
        f"Amount spent on {highest_category}: "
        f"₹{highest_amount:,.2f}"
    )

    if highest_amount / total_spending > 0.4:
        insights.append(
            f"Consider reducing spending in {highest_category}."
        )

    if "Food" in category_spending:
        insights.append(
            f"Food spending: ₹{category_spending['Food']:,.2f}"
        )

    if "Shopping" in category_spending:
        insights.append(
            f"Shopping spending: ₹{category_spending['Shopping']:,.2f}"
        )

    return category_spending, insights