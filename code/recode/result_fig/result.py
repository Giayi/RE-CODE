import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 配置数据 =====================
# 方法名称
methods = ["MetaGPT", "MetaGPT\n+EGRCM", "MetaGPT\n+PRQE", "RE-CODE"]
# 与 evaluation_comparison.png（fig1.py）一致的莫兰迪蓝系
colors = ["#90a4c5", "#93b0da", "#bfd8ec", "#dbdcee"]

# GPT-4o-mini 模型数据 (Pass@1 %)
gpt4o_mini_humaneval = [87.42, 88.05, 89.94, 91.82]
gpt4o_mini_mbpp = [76.11, 76.35, 82.20, 83.84]

# GPT-4 模型数据 (Pass@1 %)
gpt4_humaneval = [86.79, 84.91, 86.79, 93.08]
gpt4_mbpp = [80.09, 79.39, 85.95, 88.29]

x = np.arange(len(methods))  # x轴位置
width = 0.35  # 柱状图宽度

# ===================== 2. 绘图设置 =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
plt.figure(figsize=(16, 8))  # 画布大小

# ===================== 子图1：GPT-4o-mini 模型 =====================
ax1 = plt.subplot(1, 2, 1)
# HumanEval 柱状图
bars1 = ax1.bar(x - width/2, gpt4o_mini_humaneval, width, label="HumanEval", color=colors[0], edgecolor="white")
# MBPP 柱状图
bars2 = ax1.bar(x + width/2, gpt4o_mini_mbpp, width, label="MBPP", color=colors[2], edgecolor="white")

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    # 图表样式
ax1.set_title("GPT-4o-mini", fontsize=14, fontweight="bold", pad=20)
ax1.set_ylabel("Pass@1 (%)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=10)
ax1.legend(loc="upper left")
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(70, 95)

# ===================== 子图2：GPT-4 模型 =====================
ax2 = plt.subplot(1, 2, 2)
# HumanEval 柱状图
bars3 = ax2.bar(x - width/2, gpt4_humaneval, width, label="HumanEval", color=colors[1], edgecolor="white")
# MBPP 柱状图
bars4 = ax2.bar(x + width/2, gpt4_mbpp, width, label="MBPP", color=colors[3], edgecolor="white")

# 添加数值标签
for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

# 图表样式
ax2.set_title("GPT-4", fontsize=14, fontweight="bold", pad=20)
ax2.set_ylabel("Pass@1 (%)", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=10)
ax2.legend(loc="upper left")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(70, 95)

# 总标题
plt.suptitle("Ablation Study of MetaGPT on HumanEval and MBPP", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()  # 自动调整布局
plt.savefig("model_performance_comparison.png", dpi=300, bbox_inches="tight")  # 保存高清图片
plt.show()