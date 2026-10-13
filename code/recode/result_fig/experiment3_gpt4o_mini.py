import matplotlib.pyplot as plt
import numpy as np

# 配色与 full_evaluation_comparison.png（fig.py）一致
colors = {
    "HumanEval": "#8598c7",
    "MBPP": "#b3cce6",
}

methods = ["MetaGPT", "MetaGPT\n+EGRCM", "MetaGPT\n+PRQE", "RE-CODE"]

# GPT-4o-mini Pass@1（%）。HumanEval / MBPP 来自 results/experiment3_evaluate；
# MetaGPT+EGRCM+PRQE 的 HumanEval 按指定值使用 93.08。
humaneval = [87.42, 88.05, 89.94, 93.08]
mbpp = [76.11, 76.35, 82.20, 84.07]

x = np.arange(len(methods))
width = 0.35

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

bars_he = ax.bar(
    x - width / 2,
    humaneval,
    width,
    label="HumanEval",
    color=colors["HumanEval"],
    edgecolor="white",
    linewidth=0.6,
)
bars_mb = ax.bar(
    x + width / 2,
    mbpp,
    width,
    label="MBPP",
    color=colors["MBPP"],
    edgecolor="white",
    linewidth=0.6,
)

for bars in (bars_he, bars_mb):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

ax.set_title("GPT-4o-mini", fontsize=14, fontweight="bold", pad=16)
ax.set_ylabel("Pass@1 (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(loc="upper left", frameon=True, fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(70, 102)
ax.set_axisbelow(True)

fig.suptitle(
    "Ablation Study of MetaGPT on HumanEval and MBPP",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)
fig.tight_layout()
fig.savefig(
    "result_fig/experiment3_gpt4o_mini.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)
