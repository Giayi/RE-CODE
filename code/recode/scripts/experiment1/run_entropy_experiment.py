#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
需求收敛实验脚本：将原始需求经 EGRCM（需求熵模块）多轮澄清后，产出收敛后的需求并保存。

实验流程：原始需求 → 需求熵模块 (EGRCM) → round refined requirement
输出：entropy_experiment_results.json（内容为简体中文）

运行方式（项目根目录执行，默认读写 results/ 下路径）：
  python scripts/run_entropy_experiment.py --limit 10
  # product 模式输出到 results/experiments/entropy_product/ 请加：
  #   --mode product --output results/experiments/entropy_product/entropy_experiment_results.json
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 保证从项目根目录运行，并让 metagpt 可被导入
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if _PROJECT_ROOT not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from metagpt.context import Context
from metagpt.llm import LLM
from metagpt.modules.entropy_dialogue_engine import (
    EntropyDialogueConfig,
    EntropyReductionDialogueEngine,
)
from metagpt.states.requirement_state import RequirementState


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


async def run_single(
    raw: str,
    engine: EntropyReductionDialogueEngine,
    llm,
) -> dict:
    """对单条原始需求跑 EGRCM，返回一条结果记录。"""
    state = RequirementState(raw_requirement=raw)

    async def ask_fn(question_text: str) -> str:
        return await llm.aask(question_text)

    refined_state = await engine.refine(
        state=state,
        ask_fn=ask_fn,
        llm=llm,
    )

    return {
        "original_requirement": refined_state.raw_requirement,
        "refined_requirement": refined_state.refined_requirement or refined_state.raw_requirement,
        "entropy_score": refined_state.entropy_score,
        "clarifications": [
            {
                "question_id": c.question_id,
                "dimension": c.dimension,
                "question": c.question,
                "answer": c.answer,
            }
            for c in refined_state.clarifications
        ],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="需求收敛实验：原始需求 → EGRCM → 收敛需求")
    parser.add_argument(
        "--input",
        default="results/datasets/requirements_clean.json",
        help="原始需求 JSON 文件路径（字符串数组）",
    )
    parser.add_argument(
        "--output",
        default="results/experiments/entropy_atomic/entropy_experiment_results_atomic.json",
        help="实验结果输出路径（atomic 默认此处；product 建议用 results/experiments/entropy_product/entropy_experiment_results.json）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条需求（不指定则处理全部）",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="EGRCM 最大澄清轮数",
    )
    parser.add_argument(
        "--mode",
        choices=["product", "atomic"],
        default="atomic",
        help="product=产品级需求澄清（产品名/用户/场景等）；atomic=原子级 SRS 单条澄清（约束类型/依赖/验收条件/术语等），适合 requirements_clean.json",
    )
    args = parser.parse_args()

    requirements = load_requirements(args.input)
    if args.limit is not None:
        requirements = requirements[: args.limit]
    total = len(requirements)

    ctx = Context()
    llm = LLM(context=ctx)
    cfg = EntropyDialogueConfig(
        enabled=True,
        use_human=False,
        max_rounds=args.max_rounds,
        requirement_type=args.mode,
    )
    engine = EntropyReductionDialogueEngine(cfg)

    results = []
    for i, raw in enumerate(requirements, start=1):
        print(f"[{i}/{total}] 处理: {raw[:60]}...")
        try:
            record = await run_single(raw, engine, llm)
            results.append(record)
        except Exception as e:
            print(f"  失败: {e}")
            results.append(
                {
                    "original_requirement": raw,
                    "refined_requirement": None,
                    "entropy_score": None,
                    "clarifications": [],
                    "error": str(e),
                }
            )

    meta = {
        "source": args.input,
        "total_processed": len(results),
        "total_input": total,
        "timestamp": datetime.now().isoformat(),
        "max_rounds": args.max_rounds,
        "requirement_type": args.mode,
    }
    save_results(results, meta, args.output)
    print(f"\n实验完成，结果已保存至: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
