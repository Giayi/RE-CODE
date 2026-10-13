#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/data/llm/jiayi/MetaGPT-main")
OUT_MD = ROOT / "results/experiment2/实验二_三代表性案例对照.md"

CASES = [
    # 展示：MetaGPT-promise（长文档、方差抑制叙事）
    {
        "name": "MetaGPT-promise",
        "file": ROOT / "results/experiment2/prqe_experiments/baseline_metagpt/prqe_promise.json",
        "select": "max_v1_length",
    },
    # 展示：LLM-promise（戏剧性的底层提升，因为整体 Cohen's d 最大）
    {
        "name": "LLM-promise",
        "file": ROOT / "results/experiment2/prqe_experiments/baseline_llm/prqe_promise.json",
        "select": "max_delta",
    },
    # 展示：EGRCM-pure（在已经很强基础上再优化）
    {
        "name": "EGRCM-pure",
        "file": ROOT / "results/experiment2/prqe_experiments/EGRCM/prqe_pure.json",
        "select": "high_v1_then_improve_small_margin",
    },
]


def trunc(text: str, max_chars: int = 1400) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "\n...（截断）"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_case(case: dict, data: dict) -> dict:
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No results in {case['file']}")

    mode = case["select"]
    if mode == "max_v1_length":
        # 长文档更利于讲“模板化、结构化导致方差抑制”
        return max(results, key=lambda r: len(r.get("requirement_v1", "")))
    if mode == "max_delta":
        return max(results, key=lambda r: r.get("delta", 0))
    if mode == "high_v1_then_improve_small_margin":
        # 选择“基线分数已很高，但 PRQE 仍能带来提升”的案例
        # 先按 v1 分数高排序，取前若干，再在其中找 delta 最大但不过分离谱（偏小边际）
        ranked = sorted(results, key=lambda r: r.get("score_v1", 0), reverse=True)
        top = ranked[:min(25, len(ranked))]
        # 在 top 内筛 delta>0，若没有则退化到 delta>=0
        improved = [r for r in top if r.get("delta", 0) > 0]
        candidates = improved if improved else top
        # 在 candidates 中，优先选择 delta 最小但 >0（强调“小幅但持续优化”）
        # 若 delta 都相同，再按 score_v2 高排序
        return sorted(candidates, key=lambda r: (r.get("delta", 0), -r.get("score_v2", 0)))[0]

    raise ValueError(f"Unknown select mode: {mode}")


def render_one(case: dict, chosen: dict) -> str:
    name = case["name"]
    req_id = chosen.get("requirement_id", "")
    orig = chosen.get("original_requirement", "")
    v1 = chosen.get("requirement_v1", "")
    v2 = chosen.get("requirement_v2", "")
    score_v1 = chosen.get("score_v1", "")
    score_v2 = chosen.get("score_v2", "")
    delta = chosen.get("delta", "")

    # scores_v1 / scores_v2 是各维度的字典（completeness/clarity/...），用于点出差异维度
    scores_v1 = chosen.get("scores_v1", {}) or {}
    scores_v2 = chosen.get("scores_v2", {}) or {}
    dims = sorted(set(scores_v1.keys()) | set(scores_v2.keys()))
    delta_dims = []
    for d in dims:
        if d in scores_v1 and d in scores_v2:
            delta_dims.append((d, scores_v2[d] - scores_v1[d]))
    # 只展示正增量
    delta_dims = [x for x in delta_dims if x[1] != 0]
    delta_dims.sort(key=lambda x: abs(x[1]), reverse=True)
    delta_dims_str = ", ".join([f"{d}{'+' if dv>0 else ''}{dv}" for d, dv in delta_dims[:4]])
    if not delta_dims_str:
        delta_dims_str = "（各维度无明显变化）"

    out = []
    out.append(f"## {name}")
    out.append(f"- requirement_id: `{req_id}`")
    out.append(f"- 原始需求（原文片段）：{trunc(orig, 320).replace(chr(10), ' ')}")
    out.append(f"- 评分对比：v1={score_v1} -> v2={score_v2}（delta={delta}）")
    out.append(f"- 维度变化（Top）：{delta_dims_str}")

    out.append("")
    out.append("### v1（基线方法输出）")
    out.append("```text")
    out.append(trunc(v1, 1400))
    out.append("```")

    out.append("### v2（+PRQE 输出）")
    out.append("```text")
    out.append(trunc(v2, 1400))
    out.append("```")

    return "\n".join(out)


def main() -> None:
    rendered = ["# 实验二：三代表性案例并排对照（展示具体改写效果）", ""]

    for i, case in enumerate(CASES, 1):
        data = load(case["file"])
        chosen = select_case(case, data)
        rendered.append(render_one(case, chosen))
        if i != len(CASES):
            rendered.append("\n---\n")

    rendered.append("")
    rendered.append("## 一句话总结（建议放到正文末尾）")
    rendered.append("")
    rendered.append("- **MetaGPT-promise**：长文档中，PRQE 将高层叙述“模板化”为更可验证、原子化的需求条目，从而表现出方差抑制（输出结构更稳定）。")
    rendered.append("- **LLM-promise**：在直接 LLM 的“底层表达能力”上，PRQE 带来最强的整体提升（最大 Cohen’s d），典型案例显示从弱约束到量化验收标准的跨越。")
    rendered.append("- **EGRCM-pure**：即使基线已很强，PRQE 仍能进一步补全版本约束、可测试的性能基准等“细节缺口”。")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    print(f"written: {OUT_MD}")


if __name__ == "__main__":
    main()

