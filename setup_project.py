import os

# 定義檔案內容
files = {
    "ai_trading_project/config.py": '''\
def map_to_yfinance_ticker(code: str) -> str:
    if code.startswith("US."):
        return code.replace("US.", "")
    elif code.endswith(".HK"):
        return code
    elif code.endswith(".SH"):
        return code.replace(".SH", ".SS")
    elif code.endswith(".SZ"):
        return code
    else:
        raise ValueError(f"無法識別的代碼格式：{code}")

TICKER_CODE = "US.QQQ"
''',

    "ai_trading_project/utils/data_loader.py": '''\
import yfinance as yf
import pandas as pd

def download_and_process_data(ticker: str, start="2020-01-01", end="2023-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=5).std()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df
''',

    "ai_trading_project/main.py": '''\
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import map_to_yfinance_ticker, TICKER_CODE
from utils.data_loader import download_and_process_data

os.makedirs("ai_trading_project/data", exist_ok=True)
os.makedirs("ai_trading_project/model", exist_ok=True)

yf_ticker = map_to_yfinance_ticker(TICKER_CODE)
print(f"下載資料：{TICKER_CODE} → {yf_ticker}")

df = download_and_process_data(yf_ticker)
features = ["MA5", "MA10", "Return", "Volatility"]
X = df[features]
y = df["Target"]

X.to_csv("ai_trading_project/data/data.csv", index=False)
print("已儲存特徵資料至 data/data.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

model.save_model("ai_trading_project/model/model.json")
print("已儲存模型至 model/model.json")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"模型準確率：{acc:.2f}")
'''
}

# 建立資料夾與檔案
for path, content in files.items():
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ 項目已建立完成：ai_trading_project/")