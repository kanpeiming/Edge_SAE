import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 导入Savitzky-Golay滤波器

# 1. 读取数据
df_post = pd.read_csv("Post_TET_VCC.csv")
df_original = pd.read_csv("TET_VCC.csv")

# 2. 平滑处理（两种方法任选其一）
# 方法1：滑动平均（窗口大小window可调）
window = 5  # 窗口越大，曲线越平滑
df_post["Value_smooth"] = df_post["Value"].rolling(window=window, center=True).mean()
df_original["Value_smooth"] = df_original["Value"].rolling(window=window, center=True).mean()

# 方法2：Savitzky-Golay滤波器（更适合非线性数据）
# df_post["Value_smooth"] = savgol_filter(df_post["Value"], window_length=11, polyorder=2)
# df_original["Value_smooth"] = savgol_filter(df_original["Value"], window_length=11, polyorder=2)

# 3. 绘制折线图
plt.figure(figsize=(8, 5))
plt.plot(
    df_post["Step"],
    df_post["Value_smooth"],
    label="SDSTL (our method)",
    color="blue",
    linestyle="-",
    linewidth=2,
    alpha=0.8  # 适当降低透明度
)
plt.plot(
    df_original["Step"],
    df_original["Value_smooth"],
    label="Original TET",
    color="red",
    linestyle="-",
    linewidth=2,
    alpha=0.8
)

# 4. 定制图表
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("ACC(%)", fontsize=18)
plt.title("Comparison of SDSTL and Original TET on DVS-CIFAR10", fontsize=18, pad=20)
plt.legend(fontsize=18, loc="lower right")  # 图例位置调整到右下角，避免遮挡曲线
plt.grid(True, linestyle="--", alpha=0.4)  # 虚线网格更柔和

# 5. 设置坐标轴范围（根据实际数据调整）
plt.ylim(0.3, 0.9)  # 进一步调整纵轴范围，缩小上方空白区域

# 设置坐标轴刻度标签字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 6. 保存图表
plt.savefig("TET_vs_SDSTL_VCC.pdf", dpi=300, bbox_inches="tight")
plt.show()