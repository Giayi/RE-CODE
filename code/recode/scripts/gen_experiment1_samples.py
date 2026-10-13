#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path("/data/llm/jiayi/MetaGPT-main")
EVAL_ROOT = ROOT / "results/experiment1/experiment1_evaluate/evaluation_results"
OUT_PATH = ROOT / "results/experiment1/实验一_不同方法样例对照.md"

METHODS = ["LLM", "EGRCM", "metagpt"]
DATASETS = ["pure", "promise", "frnfr"]

METHOD_CN = {
    "LLM": "LLM（直接）",
    "EGRCM": "EGRCM（熵引导澄清）",
    "metagpt": "MetaGPT（PM→PRD）",
}


def load_eval(method: str, dataset: str) -> dict:
    p = EVAL_ROOT / method / f"evaluation_{dataset}.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_experiment_items(eval_json: dict) -> list[dict]:
    experiments = eval_json.get("experiments", {})
    if not experiments:
        return []
    exp = next(iter(experiments.values()))
    return exp.get("items", [])


def frnfr_key(original_requirement: str) -> tuple[str, str]:
    """
    FR-NFR 原始需求在评测里是字符串化 dict:
      {'requirement': '...', 'label': 'FR'/'NFR'}
    用 requirement 做对齐 key，同时返回 label 方便展示。
    """
    try:
        d = ast.literal_eval(original_requirement)
        if isinstance(d, dict):
            return str(d.get("requirement", original_requirement)), str(d.get("label", ""))
    except Exception:
        pass
    return original_requirement, ""


def trunc(text: str, max_chars: int = 700) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "\n...（截断）"


def build_index() -> dict[str, dict[str, dict[str, dict]]]:
    """
    idx[dataset][key][method] -> item
    """
    idx: dict[str, dict[str, dict[str, dict]]] = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        for m in METHODS:
            items = first_experiment_items(load_eval(m, ds))
            for it in items:
                orig = it.get("original_requirement", "")
                key = orig
                if ds == "frnfr":
                    key, _ = frnfr_key(orig)
                idx[ds].setdefault(key, {})[m] = it
    return idx


def select_examples(idx: dict, dataset: str, k: int = 3) -> list[str]:
    """
    选取“差异最大”的 k 条：优先挑 MetaGPT drift 明显更高、EGRCM drift 更低的样本。
    """
    cands: list[tuple[float, str]] = []
    for key, pack in idx[dataset].items():
        if not all(m in pack for m in METHODS):
            continue
        llm, eg, mg = pack["LLM"], pack["EGRCM"], pack["metagpt"]
        score = (mg["drift"] - eg["drift"]) * 3 + (llm["drift"] - eg["drift"]) * 1
        score += (eg["faithfulness"] - mg["faithfulness"]) + (eg["completeness"] - mg["completeness"])
        if (mg["drift"] - eg["drift"] >= 1) or (llm["drift"] - eg["drift"] >= 1):
            cands.append((score, key))
    cands.sort(reverse=True)
    if len(cands) >= k:
        return [key for _, key in cands[:k]]
    # fallback: 不足 k 时随便补齐（保证每个数据集都有展示）
    keys = [key for _, key in cands]
    for key, pack in idx[dataset].items():
        if key in keys:
            continue
        if all(m in pack for m in METHODS):
            keys.append(key)
        if len(keys) >= k:
            break
    return keys[:k]


def main() -> None:
    idx = build_index()
    selected = {ds: select_examples(idx, ds, k=3) for ds in DATASETS}

    out: list[str] = []
    out.append("# 实验1：不同方法样例对照（LLM vs EGRCM vs MetaGPT）")
    out.append("")
    out.append("结构：同一条原始需求下，展示三方法生成结果与（faithfulness / drift / completeness）评分。")
    out.append("")
    out.append("> 评分说明：faithfulness/completeness 越高越好；drift 越低越好。")

    for ds in DATASETS:
        out.append("")
        out.append(f"## 数据集：{ds.upper()}")
        for i, key in enumerate(selected[ds], 1):
            pack = idx[ds][key]
            orig_any = pack["LLM"].get("original_requirement", "")
            label = ""
            if ds == "frnfr":
                orig_text, label = frnfr_key(orig_any)
            else:
                orig_text = orig_any

            out.append("")
            out.append(f"### 样例 {i}")
            if label:
                out.append(f"- 标签：`{label}`")
            out.append(f"- 原始需求：{orig_text}")
            out.append("")

            for m in METHODS:
                it = pack[m]
                out.append(f"#### {METHOD_CN[m]}")
                out.append(
                    f"- 评分：faithfulness={it['faithfulness']} / drift={it['drift']} / completeness={it['completeness']}"
                )
                bc = it.get("brief_comment", "")
                if bc:
                    out.append(f"- 简评：{bc}")
                out.append("```text")
                out.append(trunc(it.get("generated_requirement", "")))
                out.append("```")

    out.append("")
    out.append("## 如何在论文中使用（建议写法）")
    out.append("")
    out.append("- 选 1–2 个“漂移差异最大”的样例放正文，其余放附录。")
    out.append("- 正文配合实验1的统计表强调：EGRCM 的多轮澄清能显著降低需求漂移（drift）。")
    out.append("- MetaGPT 样例重点展示：PRD 模板可能引入扩展内容导致 drift 上升（不代表其对开发任务不可用）。")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"written: {OUT_PATH}")


if __name__ == "__main__":
    main()

