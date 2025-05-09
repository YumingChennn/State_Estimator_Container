import pandas as pd
import matplotlib.pyplot as plt

# 重新讀取 CSV（假設在已上傳資料夾）
csv_path = "/home/ray/State_Estimator_Container/norm_tau_log.csv"
df = pd.read_csv(csv_path)

# 建立子圖，每隻腳一張圖
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()

# 顏色設定
colors = {"already contact": "tab:blue", "did not contact": "tab:red"}

# 每隻腳一張圖
for leg_id in range(4):
    ax = axes[leg_id]
    for status in ["already contact", "did not contact"]:
        mask = (df["Leg Number"] == leg_id) & (df["Contact Status"] == status)
        values = df[mask]["Norm Tau"].values
        ax.plot(values, label=status, color=colors[status])
    
    ax.set_title(f"Leg {leg_id}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Norm Tau")
    ax.grid(True)
    ax.legend()

plt.suptitle("Norm Tau per Leg with Contact Status")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
