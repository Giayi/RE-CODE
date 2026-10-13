#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RE-CODE（MetaGPT+EGRCM+PRQE）的逐次呼叫記錄，歸結成環節與題目層級的 token 增加原因。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BUCKET_REASON = {
    "baseline_codegen": "對照分支 D 的第 1 次代碼生成，近似單次 MetaGPT 呼叫的成本。",
    "ensemble_repeat": "四分支集成且每支固定採樣 5 次，生成階段不提前停止，因此同一題被反覆生成。",
    "context_inflation": "相對分支 D 的提示詞變長。變長部分來自 EGRCM 精煉需求被寫進 system prompt，或 PRQE 固定約束被追加。",
    "egrcm_clarify": "低熵門控沒有跳過，澄清問答把完整需求再送進模型。",
    "egrcm_summary": "澄清結束後又做一次總結，原始需求與全部回答被重新拼接。",
    "repair": "該分支候選都沒通過測試，修復呼叫把原題、測試、失敗訊息和舊程式碼一併送回模型。",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_calls(output_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    samples = []
    for path in sorted(output_dir.glob("samples_mbpp_*.jsonl")):
        samples.extend(_read_jsonl(path))
    dedup: dict[str, dict] = {}
    for row in samples:
        if row.get("task_id"):
            dedup[row["task_id"]] = row

    calls = _read_jsonl(output_dir / "token_calls.jsonl")
    seen = {(c.get("task_id"), c.get("stage"), c.get("branch"), c.get("sample_index"), c.get("total_tokens")) for c in calls}
    for row in dedup.values():
        for call in row.get("token_calls") or []:
            key = (
                call.get("task_id"),
                call.get("stage"),
                call.get("branch"),
                call.get("sample_index"),
                call.get("total_tokens"),
            )
            if key not in seen:
                calls.append(call)
                seen.add(key)
    audits = {tid: (row.get("token_audit") or {}) for tid, row in dedup.items()}
    return calls, audits


def _baseline_prompt(codegen_calls: list[dict]) -> int:
    for call in codegen_calls:
        if call.get("branch") == "D_baseline" and call.get("sample_index") == 0:
            return int(call.get("prompt_tokens") or 0)
    if not codegen_calls:
        return 0
    return min(int(c.get("prompt_tokens") or 0) for c in codegen_calls)


def _is_reference_call(call: dict) -> bool:
    return call.get("stage") == "codegen" and call.get("branch") == "D_baseline" and call.get("sample_index") == 0


def _inflate_issue(call: dict) -> str:
    if int(call.get("extra_context_chars") or 0) > 0:
        return "egrcm_context_reinjected"
    if call.get("prqe_applied"):
        return "prqe_constraints_appended"
    return "prompt_longer_than_baseline"


def analyze_problem(task_id: str, calls: list[dict], audit: dict) -> dict:
    buckets: dict[str, dict[str, Any]] = {}

    def add(name: str, tokens: int, call: dict | None = None) -> None:
        bucket = buckets.setdefault(
            name,
            {"tokens": 0, "calls": 0, "reason": BUCKET_REASON.get(name, ""), "examples": []},
        )
        bucket["tokens"] += int(tokens)
        if call is not None:
            bucket["calls"] += 1
            if len(bucket["examples"]) < 2 and call.get("reason"):
                bucket["examples"].append(call["reason"])

    codegen = [c for c in calls if c.get("stage") == "codegen"]
    d0 = _baseline_prompt(codegen)
    reference_total = 0
    inflation_by_issue: dict[str, int] = defaultdict(int)

    for call in calls:
        stage = call.get("stage")
        total = int(call.get("total_tokens") or 0)
        prompt = int(call.get("prompt_tokens") or 0)
        if stage == "egrcm_clarify":
            add("egrcm_clarify", total, call)
        elif stage == "egrcm_summary":
            add("egrcm_summary", total, call)
        elif stage == "repair":
            add("repair", total, call)
        elif stage == "codegen":
            if _is_reference_call(call):
                reference_total = total
                add("baseline_codegen", total, call)
            else:
                inflation = max(0, prompt - d0) if d0 else 0
                repeated = total - inflation
                add("ensemble_repeat", repeated, call)
                if inflation:
                    add("context_inflation", inflation, call)
                    inflation_by_issue[_inflate_issue(call)] += inflation
        else:
            add(stage or "other", total, call)

    total_tokens = sum(int(c.get("total_tokens") or 0) for c in calls)
    increase = max(0, total_tokens - reference_total)
    ranked = sorted(
        ((name, info) for name, info in buckets.items() if name != "baseline_codegen"),
        key=lambda item: item[1]["tokens"],
        reverse=True,
    )
    primary_name, primary_info = ranked[0] if ranked else ("baseline_codegen", buckets.get("baseline_codegen", {}))
    egrcm = audit.get("egrcm") or {}
    primary = {
        "bucket": primary_name,
        "tokens": primary_info.get("tokens", 0),
        "reason": primary_info.get("reason", ""),
        "evidence": (primary_info.get("examples") or [""])[0],
        "gate_reason": egrcm.get("gate_reason", ""),
        "reject_reason": egrcm.get("reject_reason", ""),
    }
    return {
        "task_id": task_id,
        "llm_calls": len(calls),
        "total_tokens": total_tokens,
        "prompt_tokens": sum(int(c.get("prompt_tokens") or 0) for c in calls),
        "completion_tokens": sum(int(c.get("completion_tokens") or 0) for c in calls),
        "baseline_call_tokens": reference_total,
        "increase_over_single_codegen": increase,
        "buckets": {name: {"tokens": info["tokens"], "calls": info["calls"], "reason": info["reason"]} for name, info in buckets.items()},
        "inflation_by_issue": dict(inflation_by_issue),
        "primary_cause": primary,
        "egrcm_skipped": egrcm.get("skipped"),
        "egrcm_gate_reason": egrcm.get("gate_reason", ""),
        "egrcm_refinement_kept": egrcm.get("refinement_kept"),
        "egrcm_reject_reason": egrcm.get("reject_reason", ""),
        "prqe": audit.get("prqe") or {},
        "stages": _stage_totals(calls),
    }


def _stage_totals(calls: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for call in calls:
        key = call.get("stage") or "unknown"
        bucket = out.setdefault(key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        bucket["calls"] += 1
        bucket["prompt_tokens"] += int(call.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(call.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(call.get("total_tokens") or 0)
    return out


def write_analysis(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    calls, audits = _load_calls(output_dir)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for call in calls:
        grouped[call.get("task_id") or "unknown"].append(call)

    problems = [
        analyze_problem(task_id, task_calls, audits.get(task_id) or {})
        for task_id, task_calls in grouped.items()
    ]
    problems.sort(key=lambda row: row["total_tokens"], reverse=True)

    def sum_bucket(name: str) -> int:
        return sum(int((row["buckets"].get(name) or {}).get("tokens") or 0) for row in problems)

    bucket_names = [
        "baseline_codegen",
        "ensemble_repeat",
        "context_inflation",
        "egrcm_clarify",
        "egrcm_summary",
        "repair",
    ]
    bucket_totals = {name: sum_bucket(name) for name in bucket_names}
    total_tokens = sum(row["total_tokens"] for row in problems)
    prompt_tokens = sum(row["prompt_tokens"] for row in problems)
    completion_tokens = sum(row["completion_tokens"] for row in problems)
    n = len(problems)
    skipped = sum(1 for row in problems if row.get("egrcm_skipped"))
    rejected = sum(1 for row in problems if row.get("egrcm_reject_reason"))
    kept = sum(1 for row in problems if row.get("egrcm_refinement_kept"))
    inflation_issues: dict[str, int] = defaultdict(int)
    for row in problems:
        for key, value in (row.get("inflation_by_issue") or {}).items():
            inflation_issues[key] += int(value)

    stage_totals: dict[str, dict[str, int]] = {}
    for row in problems:
        for stage, info in row["stages"].items():
            dest = stage_totals.setdefault(
                stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            for key in dest:
                dest[key] += int(info.get(key) or 0)

    cost = prompt_tokens * 0.15 / 1e6 + completion_tokens * 0.60 / 1e6
    top = problems[:15]
    report = _render_report(
        n=n,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost,
        bucket_totals=bucket_totals,
        stage_totals=stage_totals,
        inflation_issues=dict(inflation_issues),
        skipped=skipped,
        rejected=rejected,
        kept=kept,
        top=top,
    )
    payload = {
        "method": "MetaGPT+EGRCM+PRQE",
        "display_name": "RE-CODE",
        "model": "gpt-4o-mini",
        "dataset": "MBPP",
        "n_problems": n,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_problem": round(total_tokens / n, 2) if n else 0,
        "est_cost_usd": round(cost, 6),
        "pricing_usd_per_million": {"input": 0.15, "output": 0.60},
        "how_increase_is_split": (
            "每題以分支 D 第 1 次代碼生成作為單次基線。"
            "其餘代碼生成裡，不超過基線提示詞的部分算作集成重複呼叫；超出的提示詞算作上下文膨脹。"
            "EGRCM 澄清、EGRCM 總結、測試失敗後的修復各自單獨成桶。各桶 token 之和等於總 token。"
        ),
        "bucket_totals": [
            {"bucket": name, "tokens": bucket_totals[name], "reason": BUCKET_REASON[name]}
            for name in bucket_names
            if bucket_totals[name]
        ],
        "stage_totals": stage_totals,
        "context_inflation_issues": dict(inflation_issues),
        "egrcm_gate": {
            "skipped_problems": skipped,
            "refinement_kept_problems": kept,
            "refinement_rejected_problems": rejected,
            "not_skipped_problems": n - skipped,
        },
        "top_problems_by_total_tokens": top,
        "problems": problems,
        "report_zh": report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage_token_analysis.json"
    md_path = output_dir / "stage_token_analysis.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(report, encoding="utf-8")
    parent = output_dir.parent
    if parent.name.replace(" ", "") and output_dir.name != parent.name:
        if "RE-CODE" in parent.name or "RE-CODE" in output_dir.name:
            target_dir = parent if "RE-CODE" in parent.name else output_dir
            if target_dir != output_dir:
                (target_dir / "stage_token_analysis.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (target_dir / "stage_token_analysis.md").write_text(report, encoding="utf-8")
    return payload


def _render_report(
    *,
    n: int,
    total_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    bucket_totals: dict[str, int],
    stage_totals: dict[str, dict[str, int]],
    inflation_issues: dict[str, int],
    skipped: int,
    rejected: int,
    kept: int,
    top: list[dict],
) -> str:
    lines = [
        "# RE-CODE 在 MBPP 上的 Token 環節分析",
        "",
        "方法為 MetaGPT+EGRCM+PRQE（RE-CODE），模型為 GPT-4o-mini。",
        "",
        f"- 已完成題數：{n}",
        f"- 總 token：{total_tokens}（輸入 {prompt_tokens}，輸出 {completion_tokens}）",
        f"- 平均每題：{round(total_tokens / n, 2) if n else 0}",
        f"- 估計費用：${cost:.4f}（輸入 $0.15 / 1M，輸出 $0.60 / 1M）",
        "",
        "## 增加量落在哪個環節",
        "",
        "對照成本是每題分支 D 的第 1 次代碼生成。下表其餘各項是相對這一次呼叫多出來、或另外發生的消耗，加總後等於全部 token。",
        "",
        "| 環節 | Token | 原因 |",
        "|------|------:|------|",
    ]
    for name, reason in BUCKET_REASON.items():
        tokens = bucket_totals.get(name, 0)
        if not tokens:
            continue
        share = (tokens / total_tokens * 100) if total_tokens else 0
        lines.append(f"| {name} | {tokens} ({share:.1f}%) | {reason} |")

    lines.extend(["", "## 原始呼叫階段", ""])
    lines.append("| 階段 | 呼叫次數 | 輸入 token | 輸出 token | 合計 |")
    lines.append("|------|--------:|----------:|----------:|-----:|")
    for stage, info in sorted(stage_totals.items(), key=lambda item: item[1]["total_tokens"], reverse=True):
        lines.append(
            f"| {stage} | {info['calls']} | {info['prompt_tokens']} | {info['completion_tokens']} | {info['total_tokens']} |"
        )

    lines.extend(
        [
            "",
            "## 為什麼這些題會變貴",
            "",
            f"- EGRCM 被門控跳過的題數：{skipped}。沒有跳過的題數：{n - skipped}。",
            f"- 精煉結果被注入代碼生成的題數：{kept}。精煉結果因詞重疊過低被丟棄、但澄清 token 已消耗的題數：{rejected}。",
            "- 門控看的是函數定義之前的說明。MBPP 的 `def` 與測試在後段，多數說明裡沒有 assert 或示例，所以 EGRCM 仍會跑滿澄清與總結。",
            "- 代碼生成固定走 A（EGRCM+PRQE）、B（PRQE）、C（EGRCM）、D（基線）四條分支，每條 5 次採樣，生成時不因已經通過測試而停止。",
        ]
    )
    if inflation_issues:
        lines.append("- 提示詞超出基線的部分來自：")
        labels = {
            "egrcm_context_reinjected": "EGRCM 精煉需求被寫進 system prompt，並在該分支的每次採樣中重發",
            "prqe_constraints_appended": "PRQE 固定約束被追加到使用者提示",
            "prompt_longer_than_baseline": "提示詞比基線長，但沒有帶上精煉需求或 PRQE 標記",
        }
        for key, value in sorted(inflation_issues.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {labels.get(key, key)}：{value} token")

    lines.extend(["", "## Token 最高的題目", ""])
    lines.append("| 題目 | 總 token | 比單次生成多出 | 主因 | 主因 token |")
    lines.append("|------|--------:|---------------:|------|----------:|")
    for row in top:
        cause = row["primary_cause"]
        lines.append(
            f"| {row['task_id']} | {row['total_tokens']} | {row['increase_over_single_codegen']} | {cause.get('bucket')} | {cause.get('tokens')} |"
        )
    if top:
        lines.extend(["", "### 最高題的具體原因", ""])
        for row in top[:8]:
            cause = row["primary_cause"]
            lines.append(f"**{row['task_id']}**：{cause.get('evidence') or cause.get('reason')}")
            if row.get("egrcm_reject_reason"):
                lines.append(f"  - {row['egrcm_reject_reason']}")
            elif row.get("egrcm_gate_reason"):
                lines.append(f"  - {row['egrcm_gate_reason']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = write_analysis(args.output_dir)
    print(
        f"problems={payload['n_problems']} total_tokens={payload['total_tokens']} "
        f"avg={payload['avg_tokens_per_problem']}"
    )
