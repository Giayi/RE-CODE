#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
需求生成質量評估腳本（使用 Claude Opus 4.5 作為評分 LLM）。

對每條 (original_requirement, generated_requirement) 用 Claude 打分。

維度：faithfulness（忠實度）、drift（漂移程度）、completeness（完整性）
- faithfulness: 1-5，生成需求與原文的忠實程度（5=完全忠實）
- drift: 1-5，相對原文的漂移/發散程度（1=無漂移，5=嚴重漂移，越低越好）
- completeness: 1-5，結構化需求的完整與清晰程度（5=完整且可驗證）

模型：claude-opus-4-5
Base URL：https://api.openai-proxy.org/anthropic

請設置環境變數 CLAUDE_EVAL_API_KEY 為您的 API Key。
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if _PROJECT_ROOT not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.llm import LLM

EVAL_PROMPT = """You are an expert in software requirements quality assessment.
Given an original requirement and a generated (refined) requirement specification, score the generated specification on three dimensions.

**Scoring scale: 1-5 (integers only)**
- **faithfulness** (1-5): How faithful is the generated requirement to the original? 5 = fully preserves intent and key terms; 1 = largely ignores or contradicts the original.
- **drift** (1-5): How much does the generated requirement drift or invent content not in the original? 1 = no drift, stays on topic; 5 = severe drift (e.g. invents products, users, or features unrelated to the original).
- **completeness** (1-5): How complete and verifiable is the structured output (requirement statement, dependencies/constraints, acceptance criteria)? 5 = complete, clear, and testable; 1 = missing or vague.

**Original requirement:**
{original_requirement}

**Generated requirement:**
{generated_requirement}

**IMPORTANT**: Reply with ONLY one line of valid JSON. No markdown, no code blocks (no ```), no explanation before or after.
Use exactly these four keys: faithfulness, drift, completeness, brief_comment. Values: three integers (1-5) and one string for brief_comment.
Example (copy this format, one line only):
{{"faithfulness": 4, "drift": 2, "completeness": 5, "brief_comment": "Faithful; minor elaboration; acceptance criteria clear."}}"""


# Claude 評分 LLM 預設配置
DEFAULT_MODEL = "claude-opus-4-5"
DEFAULT_BASE_URL = "https://api.openai-proxy.org/anthropic"


def load_result_file(path: str, generated_key: str) -> list[dict]:
    """加載實驗結果 JSON，返回列表 [{ original_requirement, generated_requirement }, ...]。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError(f"無法從 {path} 解析 results 列表")
    out = []
    for r in results:
        if not isinstance(r, dict):
            continue
        orig = r.get("original_requirement") or r.get("original")
        gen = r.get(generated_key)
        if orig is None or gen is None:
            continue
        out.append({"original_requirement": orig, "generated_requirement": gen})
    return out


def _strip_markdown_code_block(text: str) -> str:
    """去掉 markdown 代碼塊包裹，如 ```json ... ``` 或 ``` ... ```。"""
    text = (text or "").strip()
    if text.startswith("```"):
        first = text.find("\n")
        if first > 0:
            text = text[first + 1 :]
        else:
            text = text[3:]
        if text.startswith("json"):
            text = text[4:].lstrip("\n")
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


def _extract_brace_json(text: str) -> str | None:
    """從文本中提取第一段完整的 {...}，返回子串或 None。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_scores_by_regex(text: str) -> dict | None:
    """正則兜底：提取 faithfulness, drift, completeness 的整數值。"""
    patterns = [
        r'"faithfulness"\s*:\s*(\d+)',
        r"'faithfulness'\s*:\s*(\d+)",
        r"faithfulness\s*:\s*(\d+)",
        r'"drift"\s*:\s*(\d+)',
        r"'drift'\s*:\s*(\d+)",
        r"drift\s*:\s*(\d+)",
        r'"completeness"\s*:\s*(\d+)',
        r"'completeness'\s*:\s*(\d+)",
        r"completeness\s*:\s*(\d+)",
    ]
    keys = ["faithfulness", "drift", "completeness"]
    found = {}
    for i, key in enumerate(keys):
        for pat in [patterns[i * 3], patterns[i * 3 + 1], patterns[i * 3 + 2]]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                found[key] = min(5, max(1, int(m.group(1))))
                break
        if key not in found:
            return None
    cm = re.search(r'"brief_comment"\s*:\s*"([^"]*)"', text)
    if not cm:
        cm = re.search(r"'brief_comment'\s*:\s*'([^']*)'", text)
    comment = cm.group(1)[:500] if cm else ""
    return {
        "faithfulness": found["faithfulness"],
        "drift": found["drift"],
        "completeness": found["completeness"],
        "brief_comment": comment,
    }


def parse_score_reply(text: str) -> dict:
    """從 LLM 回覆中解析 JSON，提取 faithfulness, drift, completeness, brief_comment。"""
    raw = (text or "").strip()
    if not raw:
        return _fail_scores("(empty reply)")

    text = _strip_markdown_code_block(raw)

    try:
        obj = json.loads(text)
        return _norm_scores(obj)
    except (json.JSONDecodeError, TypeError):
        pass

    frag = _extract_brace_json(text)
    if frag:
        try:
            obj = json.loads(frag)
            return _norm_scores(obj)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            one_line = " ".join(frag.split())
            obj = json.loads(one_line)
            return _norm_scores(obj)
        except (json.JSONDecodeError, TypeError):
            pass

    parsed = _parse_scores_by_regex(raw)
    if parsed:
        return parsed

    return _fail_scores(raw[:200])


def _fail_scores(reason: str) -> dict:
    return {
        "faithfulness": 0,
        "drift": 0,
        "completeness": 0,
        "brief_comment": f"(parse failed: {reason})",
    }


def _norm_scores(obj: dict) -> dict:
    """將解析出的對象規範為 1-5 分與 brief_comment。"""

    def to_int(v):
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return 0

    def clamp(v):
        v = to_int(v)
        return min(5, max(1, v)) if v else 1

    return {
        "faithfulness": clamp(obj.get("faithfulness", 0)),
        "drift": clamp(obj.get("drift", 0)),
        "completeness": clamp(obj.get("completeness", 0)),
        "brief_comment": str(obj.get("brief_comment", ""))[:500],
    }


async def score_single(original: str, generated: str, llm) -> dict:
    """對單條 (original, generated) 調用 LLM 打分。"""
    prompt = EVAL_PROMPT.format(
        original_requirement=original,
        generated_requirement=generated,
    )
    reply = await llm.aask(prompt)
    scores = parse_score_reply(reply)
    return {
        "original_requirement": original,
        "generated_requirement": generated,
        **scores,
    }


async def evaluate_experiment(
    name: str,
    result_path: str,
    generated_key: str,
    llm,
    limit: int | None,
) -> dict:
    """評估一個實驗結果文件，返回 { source, items: [...], summary: {...} }。"""
    items_raw = load_result_file(result_path, generated_key)
    if limit is not None:
        items_raw = items_raw[:limit]
    items = []
    for i, row in enumerate(items_raw):
        print(f"  [{name}] 評分 {i+1}/{len(items_raw)}: {row['original_requirement'][:50]}...")
        try:
            ev = await score_single(
                row["original_requirement"],
                row["generated_requirement"],
                llm,
            )
            items.append(ev)
        except Exception as e:
            items.append({
                "original_requirement": row["original_requirement"],
                "generated_requirement": row["generated_requirement"],
                "faithfulness": 0,
                "drift": 0,
                "completeness": 0,
                "brief_comment": str(e),
            })
    n = len(items)
    if n == 0:
        summary = {"avg_faithfulness": 0, "avg_drift": 0, "avg_completeness": 0, "count": 0}
    else:
        summary = {
            "avg_faithfulness": round(sum(x["faithfulness"] for x in items) / n, 2),
            "avg_drift": round(sum(x["drift"] for x in items) / n, 2),
            "avg_completeness": round(sum(x["completeness"] for x in items) / n, 2),
            "count": n,
        }
    return {
        "source": result_path,
        "generated_key": generated_key,
        "items": items,
        "summary": summary,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="對需求生成結果進行 Claude 打分（faithfulness / drift / completeness）"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CLAUDE_EVAL_API_KEY", ""),
        help="Claude API Key（預設從環境變數 CLAUDE_EVAL_API_KEY 讀取）",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API Base URL（預設: {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名稱（預設: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--baseline",
        default="results/experiments/baseline_direct/baseline_direct_results.json",
        help="Baseline 直接 LLM 結果路徑（generated 字段為 direct_output）",
    )
    parser.add_argument(
        "--atomic",
        default="results/experiments/entropy_atomic/entropy_experiment_results_atomic.json",
        help="EGRCM 原子級結果路徑（generated 字段為 refined_requirement）",
    )
    parser.add_argument(
        "--output",
        default="results/evaluation_results/claude/evaluation_results.json",
        help="評估結果輸出路徑",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每個實驗只評估前 N 條（用於快速測試）",
    )
    parser.add_argument(
        "--only",
        choices=["baseline", "atomic"],
        default=None,
        help="只評估其中一個實驗",
    )
    parser.add_argument(
        "--generated-key",
        default=None,
        help="生成內容字段名（預設 baseline 用 direct_output，atomic 用 refined_requirement）",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("錯誤: 請設置 API Key（環境變數 CLAUDE_EVAL_API_KEY 或 --api-key）")
        sys.exit(1)

    llm_config = LLMConfig(
        api_key=args.api_key,
        api_type=LLMType.ANTHROPIC,
        base_url=args.base_url,
        model=args.model,
    )
    llm = LLM(llm_config=llm_config)

    print(f"使用評分模型: {args.model} @ {args.base_url}\n")

    experiments = {}

    baseline_key = args.generated_key or "direct_output"
    if args.only != "atomic" and Path(args.baseline).exists():
        print(f"評估 baseline: {args.baseline} (generated_key={baseline_key})")
        experiments["baseline_direct"] = await evaluate_experiment(
            "baseline_direct",
            args.baseline,
            baseline_key,
            llm,
            args.limit,
        )
    else:
        if args.only != "atomic" and not Path(args.baseline).exists():
            print(f"跳過 baseline（文件不存在）: {args.baseline}")

    atomic_key = args.generated_key or "refined_requirement"
    if args.only != "baseline" and Path(args.atomic).exists():
        print(f"評估 atomic: {args.atomic} (generated_key={atomic_key})")
        experiments["entropy_atomic"] = await evaluate_experiment(
            "entropy_atomic",
            args.atomic,
            atomic_key,
            llm,
            args.limit,
        )
    else:
        if args.only != "baseline" and not Path(args.atomic).exists():
            print(f"跳過 atomic（文件不存在）: {args.atomic}")

    payload = {
        "evaluation_meta": {
            "timestamp": datetime.now().isoformat(),
            "evaluator_model": args.model,
            "evaluator_base_url": args.base_url,
            "scale": "1-5 (faithfulness/completeness: higher better; drift: lower better)",
            "dimensions": ["faithfulness", "drift", "completeness"],
        },
        "experiments": experiments,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n評估完成，已保存至: {args.output}")
    for name, data in experiments.items():
        s = data.get("summary", {})
        print(f"  {name}: faithfulness={s.get('avg_faithfulness')}, drift={s.get('avg_drift')}, completeness={s.get('avg_completeness')} (n={s.get('count')})")


if __name__ == "__main__":
    asyncio.run(main())
