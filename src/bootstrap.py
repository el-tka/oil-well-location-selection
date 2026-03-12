import numpy as np
import pandas as pd

BUDGET = 10_000_000_000
REVENUE_PER_UNIT = 450_000
N_WELLS = 200


def calculate_revenue(target, predictions):
    """Calculate profit from selected wells"""
    predictions = pd.Series(predictions)

    top_predictions = predictions.sort_values(ascending=False).head(N_WELLS)
    selected = target[top_predictions.index]

    total_product = selected.sum()
    revenue = total_product * REVENUE_PER_UNIT - BUDGET

    return revenue


def bootstrap_profit(target, predictions, n_samples=1000):

    state = np.random.RandomState(12345)

    values = []

    target = target.reset_index(drop=True)
    predictions = pd.Series(predictions)

    for _ in range(n_samples):

        target_subsample = target.sample(n=500, replace=True, random_state=state)
        preds_subsample = predictions[target_subsample.index]

        values.append(calculate_revenue(target_subsample, preds_subsample))

    values = pd.Series(values)

    mean_profit = values.mean()

    confidence_interval = (
        values.quantile(0.025),
        values.quantile(0.975)
    )

    risk = (values < 0).mean() * 100

    return mean_profit, confidence_interval, risk, values