#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline MetaGPT（无需求熵）：使用原 MetaGPT ProductManager，不启用需求熵模块。

对每条原始需求调用 ProductManager（entropy_config 不启用），走 UserRequirement → PrepareDocuments → WritePRD 一轮，
产出 PRD/总结内容，用于与「直接 LLM」和「EGRCM 原子级」对比。

输入：JSON 字符串数组（如 results/datasets/promise/nfr_requirements_clean.json）
输出：results/experiments/baseline_metagpt/baseline_metagpt_results.json

修复：每条需求使用独立临时 workspace，运行后清理，避免 context 累积导致超长（128K tokens 限制）。
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if _PROJECT_ROOT not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from metagpt.actions import UserRequirement
from metagpt.context import Context
from metagpt.roles import ProductManager
from metagpt.schema import Message


def load_requirements(input_path: str) -> list[str]:
    """从 JSON 文件加载需求列表（支持字符串数组或 [{"requirement": "...", "label": "..."}]）。"""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 须为数组")
    out = []
    for item in data:
        if isinstance(item, str):
            s = item.strip()
        elif isinstance(item, dict) and "requirement" in item:
            s = str(item["requirement"]).strip()
        else:
            s = str(item).strip()
        if s:
            out.append(s)
    return out


def save_results(results: list[dict], meta: dict, output_path: str) -> None:
    """将实验结果写入 JSON 文件。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_meta": meta, "results": results}
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


async def run_single(raw: str, pm: ProductManager) -> dict:
    """对单条原始需求调用 ProductManager（无需求熵），返回结果记录。"""
    msg = Message(content=raw, cause_by=UserRequirement)
    rsp = await pm.run(with_message=msg)
    if rsp is None:
        return {"original_requirement": raw, "metagpt_output": ""}
    # 输出以 rsp.content 为主；若有 PRD 文档则一并拼接便于对比
    metagpt_output = rsp.content or ""
    if getattr(rsp, "instruct_content", None) and getattr(rsp.instruct_content, "docs", None):
        docs = rsp.instruct_content.docs
        if docs:
            first_doc = list(docs.values())[0]
            doc_content = getattr(first_doc, "content", None) or str(first_doc)
            if doc_content and doc_content not in metagpt_output:
                metagpt_output = (metagpt_output + "\n\n" + doc_content).strip() if metagpt_output else doc_content
    return {
        "original_requirement": raw,
        "metagpt_output": metagpt_output,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline MetaGPT：无需求熵模块，ProductManager 单轮 PrepareDocuments → WritePRD"
    )
    parser.add_argument(
        "--input",
        default="results/datasets/promise/nfr_requirements_clean.json",
        help="原始需求 JSON 文件路径（字符串数组）",
    )
    parser.add_argument(
        "--output",
        default="results/experiments/baseline_metagpt/baseline_metagpt_results.json",
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

    # 无需求熵：不传 entropy_config 或 enabled=False，走 PrepareDocuments → WritePRD
    # 每条需求使用新的 Context、ProductManager 和独立临时 workspace，运行后清理，避免 context 超长
    results = []
    for i, raw in enumerate(requirements, start=1):
        print(f"[{i}/{total}] MetaGPT(无熵): {raw[:60]}...")
        # 为每条需求创建独立临时 workspace，避免 workspace 累积导致 context 超 128K tokens
        workspace_dir = Path(tempfile.mkdtemp(prefix="metagpt_baseline_"))
        try:
            ctx = Context()
            # 使用独立 workspace，运行后清理
            ctx.config.workspace.path = workspace_dir
            ctx.config.workspace.use_uid = False
            pm = ProductManager(context=ctx, use_fixed_sop=True)
            assert not pm.use_entropy_dialogue, "应关闭需求熵"
            try:
                record = await run_single(raw, pm)
                results.append(record)
            except Exception as e:
                print(f"  失败: {e}")
                results.append(
                    {
                        "original_requirement": raw,
                        "metagpt_output": None,
                        "error": str(e),
                    }
                )
        finally:
            # 清理临时 workspace，释放磁盘并避免后续 run 加载历史文件
            if workspace_dir.exists():
                try:
                    shutil.rmtree(workspace_dir, ignore_errors=True)
                except OSError:
                    pass

    meta = {
        "method": "baseline_metagpt",
        "description": "原 MetaGPT ProductManager，无需求熵模块，单轮 PrepareDocuments → WritePRD",
        "source": args.input,
        "total_processed": len(results),
        "total_input": total,
        "timestamp": datetime.now().isoformat(),
    }
    save_results(results, meta, args.output)
    print(f"\nBaseline MetaGPT 完成，结果已保存至: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
