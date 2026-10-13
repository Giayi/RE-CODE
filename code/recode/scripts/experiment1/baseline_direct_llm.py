#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline：直接使用 LLM 对原始需求做单次生成，无澄清、无多轮对话、无熵控制。

用途：与 EGRCM（需求熵模块）对比，用于证明直接生成可能带来的问题，例如：
  - 需求漂移（偏离原文）
  - 依赖缺失或虚构
  - 验收标准模糊

输入：results/datasets/requirements_clean.json（默认）
输出：results/experiments/baseline_direct/baseline_direct_results.json（默认）

运行（项目根目录）：
  python scripts/baseline_direct_llm.py --limit 10
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if _PROJECT_ROOT not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from metagpt.context import Context
from metagpt.llm import LLM

DIRECT_PROMPT = """You are a software requirement analyst.
Given the following requirement, produce a structured requirement specification including:
1. Requirement statement
2. Dependencies and constraints
3. Acceptance criteria

Requirement:
{raw_requirement}

Please output in Simplified Chinese (简体中文)."""


def load_requirements(input_path: str) -> list[str]:
    """从 JSON 文件加载需求列表（每项为字符串）。"""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 须为字符串数组")
    return [str(item).strip() for item in data if str(item).strip()]


def save_results(results: list[dict], meta: dict, output_path: str) -> None:
    """将实验结果写入 JSON 文件。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_meta": meta, "results": results}
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


async def run_single(raw: str, llm) -> dict:
    """对单条原始需求做一次直接 LLM 调用，返回结果记录。"""
    prompt = DIRECT_PROMPT.format(raw_requirement=raw)
    direct_output = await llm.aask(prompt)
    return {
        "original_requirement": raw,
        "direct_output": direct_output,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline：直接 LLM 生成结构化需求（无澄清、无多轮、无熵控制）"
    )
    parser.add_argument(
        "--input",
        default="results/datasets/requirements_clean.json",
        help="原始需求 JSON 文件路径（字符串数组）",
    )
    parser.add_argument(
        "--output",
        default="results/experiments/baseline_direct/baseline_direct_results.json",
        help="结果输出路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条需求（不指定则处理全部）",
    )
    args = parser.parse_args()

    requirements = load_requirements(args.input)
    if args.limit is not None:
        requirements = requirements[: args.limit]
    total = len(requirements)

    ctx = Context()
    llm = LLM(context=ctx)

    results = []
    for i, raw in enumerate(requirements, start=1):
        print(f"[{i}/{total}] 直接生成: {raw[:60]}...")
        try:
            record = await run_single(raw, llm)
            results.append(record)
        except Exception as e:
            print(f"  失败: {e}")
            results.append(
                {
                    "original_requirement": raw,
                    "direct_output": None,
                    "error": str(e),
                }
            )

    meta = {
        "method": "direct_llm",
        "description": "Single LLM call per requirement, no clarification, no multi-round, no entropy control",
        "source": args.input,
        "total_processed": len(results),
        "total_input": total,
        "timestamp": datetime.now().isoformat(),
    }
    save_results(results, meta, args.output)
    print(f"\nBaseline 完成，结果已保存至: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
