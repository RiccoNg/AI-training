import pandas as pd

def apply_risk_control(
    df: pd.DataFrame,
    initial_capital: float,
    cooldown_days: int = 3,
    max_loss_per_trade: float = -0.03,
    take_profit: float = 0.05,
    hold_days: int = 1,
    dynamic_position: bool = False,
    min_capital_ratio: float = 0.5
) -> pd.DataFrame:
    """
    新增參數：
      - take_profit: 停利門檻（正數），例如 0.05 表示 +5% 停利
      - hold_days: 固定持有天數
      - dynamic_position: 是否依 prob 強度動態調整倉位
    """

    signals = []
    positions = []
    cooldown = 0
    loss_streak = 0
    capital = initial_capital
    hold_counter = 0

    for i in range(len(df)):
        ret    = df["Return"].iloc[i]
        rsi    = df["rsi"].iloc[i]
        macd   = df["macd"].iloc[i]
        macd_sig = df["macd_signal"].iloc[i]
        pred   = df["y_pred"].iloc[i]
        prob   = df["y_prob"].iloc[i] if "y_prob" in df.columns else None

        signal = "觀望"
        position = 0

        # 基本條件檢查
        rsi_ok  = (rsi < 70) if pred==1 else (rsi > 30)
        macd_ok = (macd > macd_sig) if pred==1 else (macd < macd_sig)

        if cooldown==0 and capital >= initial_capital * min_capital_ratio:
            # 動態倉位例子
            if dynamic_position and prob is not None:
                if prob>0.7 and rsi_ok and macd_ok:
                    signal, position = "強烈買入", 1
                elif prob<0.3 and rsi_ok and macd_ok:
                    signal, position = "強烈賣出", 0
            else:
                if pred==1 and rsi_ok and macd_ok:
                    signal, position = "買入", 1
                elif pred==0 and rsi_ok and macd_ok:
                    signal, position = "賣出", 0

            # 停利判斷
            if position==1 and ret>take_profit:
                signal += "（停利）"
                position = 0
                cooldown = cooldown_days

            # 停損判斷
            if ret < max_loss_per_trade:
                signal += "（強制停損）"
                position = 0
                cooldown = cooldown_days

            # 連續虧損冷卻
            loss_streak = loss_streak+1 if ret<0 else 0
            if loss_streak>=3:
                signal += "（連續虧損冷卻）"
                cooldown = cooldown_days

            # 固定持有天數
            if position==1:
                hold_counter = hold_days
        else:
            # 冷卻或門檻未過
            signal, position = "冷卻期／停損中", 0
            cooldown = max(cooldown-1, 0)
            hold_counter = max(hold_counter-1, 0)

        signals.append(signal)
        positions.append(position)
        capital *= (1 + ret) if position==1 else 1

    df["交易提示"] = signals
    df["持倉狀態"] = positions
    return df