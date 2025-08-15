# simulate_strategy.py

import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from utils.risk_control import apply_risk_control
from utils.performance_metrics import evaluate_performance
from utils.indicators import calculate_indicators


def simulate_strategy(
    df: pd.DataFrame,
    y_pred,
    y_prob=None,
    fee: float = 0.0004,
    stop_loss: float = -0.05,
    take_profit: float = 0.05,
    hold_days: int = 1,
    dynamic_position: bool = False,
    initial_capital: float = 30000,
    split_index: int = None,
    plot_curve: bool = False,
    verbose: bool = True
):
    """
    回傳：
      - metrics: dict 包含 final_capital, cagr, sharpe, max_drawdown, mdd_pct
      - test_df: DataFrame 模擬後的完整交易紀錄
    """

    # 1. 計算技術指標
    df = calculate_indicators(df)

    # 2. 切出測試集
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    test_df["y_pred"] = y_pred
    if y_prob is not None:
        test_df["y_prob"] = y_prob

    # 3. 套用風控模組（會在 test_df 裡面產生「持倉狀態」「交易提示」）
    test_df = apply_risk_control(
    test_df,
    initial_capital,
    cooldown_days = 2,
    max_loss_per_trade = stop_loss,   # 將 simulate_strategy 的 stop_loss map 到這裡
    take_profit = take_profit,
    hold_days = hold_days,
    dynamic_position = dynamic_position,
    min_capital_ratio = 0.5
    )


    # 4. 計算策略報酬與資金曲線
    capital = initial_capital
    capital_curve = []
    returns = []

    for i in range(len(test_df)):
        if test_df.loc[i, "持倉狀態"] == 1:
            ret = test_df.loc[i, "Return"] - fee
        else:
            ret = 0
        capital *= (1 + ret)
        capital_curve.append(capital)
        returns.append(ret)

    test_df["strategy_return"] = returns
    test_df["capital"] = capital_curve

    # 5. 績效分析
    metrics = evaluate_performance(test_df, initial_capital)

    # 6. 列印結果
    if verbose:
        print("\n📊 策略模擬結果（含交易提示）：")
        print(f"💰 初始資金：HK${initial_capital:,.2f}")
        print(f"📈 最終資金：HK${metrics['final_capital']:,.2f}")
        print(f"📉 最大回撤：HK${metrics['max_drawdown']:,.2f}（{metrics['mdd_pct']:.2%}）")
        print(f"📆 年化報酬率（CAGR）：{metrics['cagr']:.2%}")
        print(f"📈 夏普比率（Sharpe）：{metrics['sharpe']:.2f}")
        print("\n📌 最近 5 筆交易提示：")
        print(test_df[["Date", "交易提示", "持倉狀態", "strategy_return", "capital"]].tail())

    # 7. 繪製資金曲線
    if plot_curve:
        plt.figure(figsize=(10, 4))
        plt.plot(test_df["Date"], test_df["capital"], label="資金曲線")
        plt.title(f"資金曲線（持有 {hold_days} 天，初始 HK${initial_capital:,.0f}）")
        plt.xlabel("時間")
        plt.ylabel("資產（港元）")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return metrics, test_df


def optimize_strategy(
    df: pd.DataFrame,
    y_pred,
    y_prob=None,
    split_index: int = None,
    initial_capital: float = 30000,
    param_grid: dict = None
):
    """
    遍歷 param_grid 裡的所有參數組合，
    找出 Sharpe 比率最高者並回傳最佳參數與績效。
    """

    # 預設參數空間
    if param_grid is None:
        param_grid = {
            "fee": [0.0002, 0.0004],
            "stop_loss": [-0.03, -0.05],
            "take_profit": [0.03, 0.05],
            "hold_days": [1, 3, 5],
            "dynamic_position": [False, True]
        }

    best_sharpe = float("-inf")
    best_params = None
    best_metrics = None

    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))

        metrics, _ = simulate_strategy(
            df=df,
            y_pred=y_pred,
            y_prob=y_prob,
            fee=params["fee"],
            stop_loss=params["stop_loss"],
            take_profit=params["take_profit"],
            hold_days=params["hold_days"],
            dynamic_position=params["dynamic_position"],
            initial_capital=initial_capital,
            split_index=split_index,
            plot_curve=False,
            verbose=False
        )

        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_params = params
            best_metrics = metrics

    # 列印最佳結果
    print("\n🛠️ 優化結果：")
    print(f"最佳參數：{best_params}")
    print(f"最終資金：HK${best_metrics['final_capital']:,.2f}")
    print(f"年化報酬率：{best_metrics['cagr']:.2%}")
    print(f"最大回撤：HK${best_metrics['max_drawdown']:,.2f}（{best_metrics['mdd_pct']:.2%}）")
    print(f"夏普比率：{best_metrics['sharpe']:.2f}")

    return best_params, best_metrics


if __name__ == "__main__":
    # 1. 載入你的歷史資料
    df = pd.read_csv("data/data.csv", parse_dates=["Date"])

    # 2. 載入或預測你的 y_pred, y_prob
    #    這裡用一個簡單例子先隨機生成 y_pred：
    split_index = int(len(df) * 0.7)
    y_pred = (df["Close"].shift(-1) > df["Close"]).astype(int).iloc[split_index:].tolist()
    y_prob = None

    # 3. 單次回測（畫圖 + 列印）
    simulate_strategy(
        df, y_pred, y_prob,
        fee=0.0004,
        stop_loss=-0.05,
        take_profit=0.05,
        hold_days=1,
        dynamic_position=False,
        initial_capital=30000,
        split_index=split_index,
        plot_curve=True,
        verbose=True
    )

    # 4. 參數優化
    param_grid = {
        "fee": [0.0002, 0.0004],
        "stop_loss": [-0.03, -0.05],
        "take_profit": [0.03, 0.05],
        "hold_days": [1, 3, 5],
        "dynamic_position": [False, True]
    }
    optimize_strategy(
        df, y_pred, y_prob,
        split_index=split_index,
        initial_capital=30000,
        param_grid=param_grid
    )
