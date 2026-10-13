#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MetaGPT + EGRCM：記錄每題需求熵與 Pass@1，便於分析「熵 ↔ 通過率」。

輸出目錄（預設 results/egrcm/）：
  humaneval/records_gpt-4o-mini.jsonl   # 逐題：熵 + pass + completion
  mbpp/records_gpt-4o-mini.jsonl
  humaneval/summary_gpt-4o-mini.json    # 分桶統計等
  mbpp/summary_gpt-4o-mini.json
  analysis_summary.md                   # 兩數據集匯總

用法（專案根目錄，metagpt_env）：
  python scripts/run_egrcm_entropy_pass1.py
  python scripts/run_egrcm_entropy_pass1.py --dataset humaneval --limit 3
  python scripts/run_egrcm_entropy_pass1.py --dataset both --model gpt-4o-mini --delay 1
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from metagpt.modules.entropy_dialogue_engine import (
    EntropyDialogueConfig,
    EntropyReductionDialogueEngine,
)
from metagpt.modules.gating import should_skip_egrcm
from metagpt.modules.postprocess import postprocess
from metagpt.modules.prompt_enhancer import inject_test_cases
from metagpt.modules.soft_constraints import inject_soft_constraints
from metagpt.states.requirement_state import RequirementState

DEFAULT_OUT = _PROJECT_ROOT / "results" / "egrcm"
HE_DATASET = _PROJECT_ROOT / "results" / "datasets" / "humaneval" / "humaneval_public_test.jsonl"
MB_DATASET = _PROJECT_ROOT / "results" / "datasets" / "mbpp" / "mbpp_public_test.jsonl"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_he():
    return _load_module(_SCRIPT_DIR / "run_experiment3_humaneval.py", "he_egrcm_entropy")


def _load_mb():
    return _load_module(_SCRIPT_DIR / "run_experiment3_mbpp.py", "mb_egrcm_entropy")


def _load_he_eval():
    return _load_module(_SCRIPT_DIR / "evaluate_humaneval_pass1.py", "he_eval_entropy")


def _load_mb_eval():
    return _load_module(_SCRIPT_DIR / "evaluate_mbpp_pass1.py", "mb_eval_entropy")


class SimpleChatLLM:
    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    async def aask(self, prompt: str) -> str:
        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_call)


async def run_egrcm_with_entropy(
    requirement: str,
    llm,
    max_rounds: int = 3,
) -> dict[str, Any]:
    """Run EGRCM and return refined text + entropy metadata."""
    entropy_before = 1.0
    if not requirement:
        return {
            "refined": "",
            "entropy_before": entropy_before,
            "entropy_after": entropy_before,
            "entropy_delta": 0.0,
            "egrcm_skipped": True,
            "skip_reason": "empty_requirement",
            "egrcm_rounds": 0,
            "n_clarifications": 0,
        }

    if should_skip_egrcm(requirement):
        # Gating says already low-entropy / fully specified.
        # Use threshold 0.3 so skipped items form the low-entropy group in analysis
        # (do NOT leave score at 1.0, which would look like high entropy).
        low = 0.3
        return {
            "refined": requirement,
            "entropy_before": low,
            "entropy_after": low,
            "entropy_delta": 0.0,
            "egrcm_skipped": True,
            "skip_reason": "gating_low_entropy_heuristic",
            "egrcm_rounds": 0,
            "n_clarifications": 0,
        }

    cfg = EntropyDialogueConfig(
        enabled=True,
        use_human=False,
        max_rounds=max_rounds,
        requirement_type="atomic",
    )
    engine = EntropyReductionDialogueEngine(cfg)
    state = RequirementState(raw_requirement=requirement)

    async def ask_fn(q: str) -> str:
        return await llm.aask(q)

    refined_state = await engine.refine(state=state, ask_fn=ask_fn, llm=llm)
    entropy_after = float(refined_state.entropy_score)
    n_clar = len(refined_state.clarifications)
    return {
        "refined": refined_state.refined_requirement or requirement,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy_delta": entropy_before - entropy_after,
        "egrcm_skipped": False,
        "skip_reason": "",
        "egrcm_rounds": n_clar,  # one clarification ≈ one round
        "n_clarifications": n_clar,
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj.get("task_id")
            if tid and "passed" in obj:
                done.add(tid)
    return done


def _entropy_bins(records: list[dict], n_bins: int = 5) -> list[dict]:
    """Bucket by entropy_after and compute Pass@1 per bucket."""
    vals = [float(r.get("entropy_after", 1.0)) for r in records]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if math.isclose(lo, hi):
        passed = sum(1 for r in records if r.get("passed"))
        return [
            {
                "bin": 0,
                "entropy_low": lo,
                "entropy_high": hi,
                "n": len(records),
                "passed": passed,
                "pass_at_1": passed / len(records),
            }
        ]

    # equal-width bins on [lo, hi]
    width = (hi - lo) / n_bins
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        e = float(r.get("entropy_after", 1.0))
        idx = min(n_bins - 1, int((e - lo) / width)) if width > 0 else 0
        buckets[idx].append(r)

    out = []
    for i in range(n_bins):
        rs = buckets.get(i, [])
        if not rs:
            continue
        passed = sum(1 for r in rs if r.get("passed"))
        out.append(
            {
                "bin": i,
                "entropy_low": lo + i * width,
                "entropy_high": lo + (i + 1) * width,
                "n": len(rs),
                "passed": passed,
                "pass_at_1": passed / len(rs),
            }
        )
    return out


def _summarize(records: list[dict], dataset: str, model: str) -> dict:
    n = len(records)
    passed = sum(1 for r in records if r.get("passed"))
    skipped = sum(1 for r in records if r.get("egrcm_skipped"))
    ran = n - skipped
    avg_after = sum(float(r.get("entropy_after", 0)) for r in records) / n if n else 0.0
    avg_delta = sum(float(r.get("entropy_delta", 0)) for r in records) / n if n else 0.0

    # skipped vs ran Pass@1
    def _rate(subset):
        if not subset:
            return None
        p = sum(1 for r in subset if r.get("passed"))
        return p / len(subset)

    skip_set = [r for r in records if r.get("egrcm_skipped")]
    run_set = [r for r in records if not r.get("egrcm_skipped")]

    return {
        "dataset": dataset,
        "method": "metagpt_egrcm",
        "model": model,
        "n_problems": n,
        "passed": passed,
        "pass_at_1": passed / n if n else 0.0,
        "egrcm_skipped": skipped,
        "egrcm_run": ran,
        "pass_at_1_skipped": _rate(skip_set),
        "pass_at_1_egrcm_run": _rate(run_set),
        "avg_entropy_after": round(avg_after, 6),
        "avg_entropy_delta": round(avg_delta, 6),
        "entropy_bins": _entropy_bins(records),
        "note": (
            "entropy_score starts at 1.0 and multiplies by 0.7 per clarification round; "
            "skipped problems keep entropy_after=1.0 (gating heuristic)."
        ),
    }


async def run_humaneval(
    *,
    client: OpenAI,
    model: str,
    out_dir: Path,
    limit: Optional[int],
    delay: float,
    max_rounds: int,
) -> Path:
    he = _load_he()
    he_eval = _load_he_eval()
    llm = SimpleChatLLM(client, model)
    problems = he.load_problems(HE_DATASET, limit)
    model_safe = model.replace(":", "-")
    records_path = out_dir / "humaneval" / f"records_{model_safe}.jsonl"
    done = _load_done_ids(records_path)
    if done:
        print(f"[HumanEval] resume: skip {len(done)} done")

    dataset_lookup = he_eval.load_dataset(HE_DATASET)

    for problem in tqdm(problems, desc="HumanEval metagpt_egrcm"):
        task_id = problem.get("task_id") or problem.get("problem_id") or "unknown"
        if task_id in done:
            continue

        docstring = he.extract_docstring(problem.get("prompt") or "")
        try:
            ent = await run_egrcm_with_entropy(docstring, llm, max_rounds=max_rounds)
            prompt = he.build_prompt(problem, inject_tests=False)
            extra = (
                ent["refined"]
                if ent["refined"] and ent["refined"] != docstring and not ent["egrcm_skipped"]
                else (ent["refined"] if ent["refined"] and ent["refined"] != docstring else None)
            )
            # match method_metagpt_egrcm: only pass extra_context when refined differs
            if ent["egrcm_skipped"]:
                extra = None
            elif ent["refined"] and ent["refined"] != docstring:
                extra = ent["refined"]
            else:
                extra = None

            raw = he.generate_code_sync(client, model, prompt, extra_context=extra)
            completion = postprocess(raw, entry_point=problem.get("entry_point"))
        except Exception as e:  # noqa: BLE001
            ent = {
                "refined": "",
                "entropy_before": 1.0,
                "entropy_after": 1.0,
                "entropy_delta": 0.0,
                "egrcm_skipped": True,
                "skip_reason": f"error:{e}",
                "egrcm_rounds": 0,
                "n_clarifications": 0,
            }
            completion = f"Error: {e}"

        meta = dataset_lookup.get(task_id) or {}
        status, message = he_eval.check_solution(
            completion,
            meta.get("test") or "",
            meta.get("entry_point") or problem.get("entry_point") or "",
            timeout=15,
        )
        passed = status == "PASS"
        row = {
            "task_id": task_id,
            "dataset": "HumanEval",
            "method": "metagpt_egrcm",
            "model": model,
            "entropy_before": ent["entropy_before"],
            "entropy_after": ent["entropy_after"],
            "entropy_delta": ent["entropy_delta"],
            "egrcm_skipped": ent["egrcm_skipped"],
            "skip_reason": ent.get("skip_reason", ""),
            "egrcm_rounds": ent["egrcm_rounds"],
            "n_clarifications": ent["n_clarifications"],
            "passed": passed,
            "pass_at_1": 1.0 if passed else 0.0,
            "status": status,
            "message": message,
            "completion": completion,
            "refined_requirement": ent.get("refined", ""),
        }
        _append_jsonl(records_path, row)
        if delay > 0:
            await asyncio.sleep(delay)

    # rewrite summary from full file
    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    # dedup keep last
    dedup = {r["task_id"]: r for r in records}
    records = list(dedup.values())
    with open(records_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = _summarize(records, "HumanEval", model)
    summary_path = out_dir / "humaneval" / f"summary_{model_safe}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HumanEval] Pass@1={summary['pass_at_1']:.4f} ({summary['passed']}/{summary['n_problems']})")
    print(f"[HumanEval] records -> {records_path}")
    return records_path


async def run_mbpp(
    *,
    client: OpenAI,
    model: str,
    out_dir: Path,
    limit: Optional[int],
    delay: float,
    max_rounds: int,
) -> Path:
    mb = _load_mb()
    mb_eval = _load_mb_eval()
    llm = SimpleChatLLM(client, model)
    problems = mb.load_problems(MB_DATASET, limit)
    model_safe = model.replace(":", "-")
    records_path = out_dir / "mbpp" / f"records_{model_safe}.jsonl"
    done = _load_done_ids(records_path)
    if done:
        print(f"[MBPP] resume: skip {len(done)} done")

    dataset_lookup = mb_eval.load_mbpp_public(MB_DATASET)

    for problem in tqdm(problems, desc="MBPP metagpt_egrcm"):
        task_id = problem.get("task_id", "unknown")
        if task_id in done:
            continue

        req = mb.extract_requirement(problem.get("prompt") or "")
        try:
            ent = await run_egrcm_with_entropy(req, llm, max_rounds=max_rounds)
            refined = ent["refined"]
            # safety gate same as method_metagpt_egrcm
            if (
                req
                and refined
                and refined != req
                and not ent["egrcm_skipped"]
                and hasattr(mb, "_is_egrcm_refinement_safe")
                and not mb._is_egrcm_refinement_safe(req, refined)
            ):
                refined = req
                # treat as no effective refinement; entropy already recorded from run
            prompt = inject_test_cases(problem["prompt"], problem.get("test"))
            prompt = inject_soft_constraints(prompt)
            extra = refined if refined and refined != req and not ent["egrcm_skipped"] else None
            if ent["egrcm_skipped"]:
                extra = None
            elif refined and refined != req:
                extra = refined
            else:
                extra = None

            raw = mb.generate_code_sync(client, model, prompt, extra_context=extra, temperature=0.0)
            if hasattr(mb, "clean_code"):
                raw = mb.clean_code(raw)
            completion = postprocess(raw, entry_point=problem.get("entry_point"))
        except Exception as e:  # noqa: BLE001
            ent = {
                "refined": "",
                "entropy_before": 1.0,
                "entropy_after": 1.0,
                "entropy_delta": 0.0,
                "egrcm_skipped": True,
                "skip_reason": f"error:{e}",
                "egrcm_rounds": 0,
                "n_clarifications": 0,
            }
            completion = f"Error: {e}"

        meta = dataset_lookup.get(task_id) or {
            "entry_point": problem.get("entry_point"),
            "test": problem.get("test"),
        }
        # mbpp evaluate helper
        status, message = "FAIL", "no evaluator"
        try:
            # reuse evaluate_mbpp_pass1 check if available
            if hasattr(mb_eval, "check_solution"):
                status, message = mb_eval.check_solution(
                    completion,
                    meta.get("test") or "",
                    meta.get("entry_point") or "",
                    timeout=15,
                )
            else:
                # fallback: local check via mb._candidate_passes_mbpp
                ok = mb._candidate_passes_mbpp(problem, completion, timeout=15)
                status, message = ("PASS", "passed") if ok else ("FAIL", "failed public tests")
        except Exception as e:  # noqa: BLE001
            status, message = "FAIL", str(e)

        passed = status == "PASS"
        row = {
            "task_id": task_id,
            "dataset": "MBPP",
            "method": "metagpt_egrcm",
            "model": model,
            "entropy_before": ent["entropy_before"],
            "entropy_after": ent["entropy_after"],
            "entropy_delta": ent["entropy_delta"],
            "egrcm_skipped": ent["egrcm_skipped"],
            "skip_reason": ent.get("skip_reason", ""),
            "egrcm_rounds": ent["egrcm_rounds"],
            "n_clarifications": ent["n_clarifications"],
            "passed": passed,
            "pass_at_1": 1.0 if passed else 0.0,
            "status": status,
            "message": message,
            "completion": completion,
            "refined_requirement": ent.get("refined", ""),
        }
        _append_jsonl(records_path, row)
        if delay > 0:
            await asyncio.sleep(delay)

    records = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    dedup = {r["task_id"]: r for r in records}
    records = list(dedup.values())
    with open(records_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = _summarize(records, "MBPP", model)
    summary_path = out_dir / "mbpp" / f"summary_{model_safe}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[MBPP] Pass@1={summary['pass_at_1']:.4f} ({summary['passed']}/{summary['n_problems']})")
    print(f"[MBPP] records -> {records_path}")
    return records_path


def _write_markdown(out_dir: Path, model: str) -> None:
    model_safe = model.replace(":", "-")
    parts = ["# MetaGPT + EGRCM：需求熵 × Pass@1\n", f"模型：`{model}`\n"]
    for ds, sub in [("HumanEval", "humaneval"), ("MBPP", "mbpp")]:
        sp = out_dir / sub / f"summary_{model_safe}.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        parts.append(f"## {ds}\n")
        parts.append(
            f"- Pass@1: **{s['pass_at_1']:.4%}** ({s['passed']}/{s['n_problems']})\n"
            f"- EGRCM skipped / run: {s['egrcm_skipped']} / {s['egrcm_run']}\n"
            f"- Pass@1 (skipped): {s['pass_at_1_skipped']}\n"
            f"- Pass@1 (EGRCM run): {s['pass_at_1_egrcm_run']}\n"
            f"- avg entropy_after: {s['avg_entropy_after']}\n"
            f"- avg entropy_delta: {s['avg_entropy_delta']}\n"
        )
        if s.get("entropy_bins"):
            parts.append("\n| bin | entropy range | n | passed | Pass@1 |\n|-----|---------------|--:|-------:|-------:|\n")
            for b in s["entropy_bins"]:
                parts.append(
                    f"| {b['bin']} | [{b['entropy_low']:.3f}, {b['entropy_high']:.3f}) | "
                    f"{b['n']} | {b['passed']} | {b['pass_at_1']:.4f} |\n"
                )
        parts.append(
            f"\n逐題記錄：`results/egrcm/{sub}/records_{model_safe}.jsonl`\n"
        )
    parts.append(
        "\n## 欄位說明\n"
        "- `entropy_before` / `entropy_after`：EGRCM 狀態熵（初始 1.0，每輪澄清 ×0.7）\n"
        "- `egrcm_skipped`：gating 判定低熵而跳過\n"
        "- `passed` / `pass_at_1`：該題是否通過（0/1）\n"
    )
    (out_dir / "analysis_summary.md").write_text("".join(parts), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    he = _load_he()
    cfg = he._load_llm_config(args.model)
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    model = cfg["model"]
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    if args.dataset in ("humaneval", "both"):
        datasets.append("humaneval")
    if args.dataset in ("mbpp", "both"):
        datasets.append("mbpp")

    print(f"model={model}, datasets={datasets}, out={out_dir}")
    if "humaneval" in datasets:
        await run_humaneval(
            client=client,
            model=model,
            out_dir=out_dir,
            limit=args.limit,
            delay=args.delay,
            max_rounds=args.max_rounds,
        )
    if "mbpp" in datasets:
        await run_mbpp(
            client=client,
            model=model,
            out_dir=out_dir,
            limit=args.limit,
            delay=args.delay,
            max_rounds=args.max_rounds,
        )
    _write_markdown(out_dir, model)
    print(f"Done. See {out_dir / 'analysis_summary.md'}")


def main():
    parser = argparse.ArgumentParser(description="MetaGPT+EGRCM with entropy & Pass@1 logging")
    parser.add_argument(
        "--dataset",
        choices=["humaneval", "mbpp", "both"],
        default="both",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None, help="only first N problems")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUT,
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
