import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)
from xgboost import XGBClassifier
from utils.data_loader import download_and_process_data
from utils.strategy_simulator import simulate_strategy
from utils.strategy_simulator import optimize_strategy

# 建立 model 資料夾（如不存在）
os.makedirs("model", exist_ok=True)

# 1. 下載與處理資料
yf_ticker = "AAPL"  # 可替換成其他股票代碼
df = download_and_process_data(yf_ticker)

# 2. 指定使用的技術指標特徵
features = [
    "MA5", "MA10", "EMA12", "EMA26",
    "Return", "Volatility", "RSI",
    "MACD", "MACD_signal",
    "BB_upper", "BB_lower",
    "Momentum"
]

X = df[features]
y = df["target"]

# 3. 時間序列切分（前 80% 為訓練集）
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 4. 模型訓練（XGBoost）
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# 5. 模型儲存
joblib.dump(model, "model/xgb_model.pkl")
print("✅ 模型已儲存至 model/xgb_model.pkl")

# 6. 預測與準確率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📈 模型準確率：{accuracy:.2f}")

# 7. 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# 8. ROC 曲線與 AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 9. 分類報告
print("📊 分類報告：")
print(classification_report(y_test, y_pred))

# 10. 特徵重要性分析
importances = model.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 12. 策略模擬（單次回測 + 畫資金曲線）
simulate_strategy(
    df=df,
    y_pred=y_pred,
    y_prob=y_prob,
    fee=0.0004,           # 手續費
    stop_loss=-0.05,      # 停損 -5%
    take_profit=0.05,     # 停利 +5%
    hold_days=3,          # 持有天數
    dynamic_position=False,
    split_index=split_index,
    plot_curve=True,
    verbose=True
)

# 13. 參數優化
param_grid = {
    "fee": [0.0002, 0.0004],
    "stop_loss": [-0.03, -0.05],
    "take_profit": [0.03, 0.05],
    "hold_days": [1, 3, 5],
    "dynamic_position": [False, True]
}
best_params, best_metrics = optimize_strategy(
    df=df,
    y_pred=y_pred,
    y_prob=y_prob,
    split_index=split_index,
    initial_capital=30000,
    param_grid=param_grid
)

print("\n🎯 最佳優化結果：")
print(best_params)
print(best_metrics)


