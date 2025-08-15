import yfinance as yf
import pandas as pd
import os


def download_and_process_data(ticker: str) -> pd.DataFrame:
    # 下載歷史股價資料
    df = yf.download(ticker, start="2018-01-01", end="2024-01-01")
    df = df[["Close"]]  # 只保留收盤價
    df.dropna(inplace=True)

    # 技術指標計算
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # 報酬率與波動率
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=10).std()

    # RSI（相對強弱指標）
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD 與訊號線
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 布林通道
    df["BB_upper"] = (df["MA10"] + 2 * df["Close"].rolling(window=10).std().squeeze())
    df["BB_lower"] = (df["MA10"] - 2 * df["Close"].rolling(window=10).std().squeeze())

    # 動能指標
    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    # 標籤欄位（漲跌）
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # 清理空值
    df.dropna(inplace=True)

    # 重設索引（避免 MultiIndex 錯誤）
    df.reset_index(inplace=True)

    # 扁平化欄位名稱
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 建立資料夾並儲存
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/data.csv", index=False)
    print("✅ 已儲存特徵資料至 data/data.csv")

    return df