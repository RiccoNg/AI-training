import numpy as np

def evaluate_performance(df, initial_capital):
    capital_curve = df["capital"]
    final_capital = capital_curve.iloc[-1]
    max_drawdown = max(capital_curve) - final_capital
    mdd_pct = max_drawdown / max(capital_curve)

    days = len(df)
    years = days / 252
    cagr = (final_capital / initial_capital) ** (1 / years) - 1

    daily_returns = df["strategy_return"]
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    return {
        "final_capital": final_capital,
        "max_drawdown": max_drawdown,
        "mdd_pct": mdd_pct,
        "cagr": cagr,
        "sharpe": sharpe
    }