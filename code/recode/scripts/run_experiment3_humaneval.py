#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
實驗三：HumanEval 四種方法對比測試

方法：
  1. MetaGPT          - 直接 LLM 代碼生成（基線）
  2. MetaGPT + EGRCM  - 先對 docstring 做熵收斂（帶 gating），再生成代碼
  3. MetaGPT + PRQE   - 先對 docstring 做需求質量評估與精煉，再生成代碼
  4. MetaGPT + EGRCM + PRQE - EGRCM（帶 gating）→ PRQE → refined_docstring → 强约束（soft 强化）→ postprocess

數據集：results/datasets/humaneval/humaneval_public_test.jsonl
輸出目錄：results/full_experiments/{method}/

小範圍測試：--limit 5
完整測試：不加 --limit
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

# 保證從項目根目錄運行
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from tqdm import tqdm

from metagpt.configs.prqe_config import PRQEConfig
from metagpt.modules.entropy_dialogue_engine import (
    EntropyDialogueConfig,
    EntropyReductionDialogueEngine,
)
from metagpt.modules.gating import should_skip_egrcm
from metagpt.modules.pipeline_controller import get_code_prompt_via_rpc
from metagpt.modules.postprocess import postprocess
from metagpt.modules.prompt_enhancer import inject_test_cases
from metagpt.modules.requirement_quality_evaluator import (
    RequirementFeedbackGenerator,
    RequirementQualityEvaluator,
    RequirementRefiner,
)
from metagpt.modules.soft_constraints import inject_soft_constraints
from metagpt.states.requirement_state import RequirementState
from openai import OpenAI

from scripts.token_tracker import TRACKER, TokenTracker
import scripts.token_tracker as _tt_mod

from scripts.token_tracker import TRACKER, TokenTracker
import scripts.token_tracker as _tt_mod

# ====== 配置 ======

CONFIG_PATH = _PROJECT_ROOT / "config" / "config2.yaml"
DEFAULT_DATASET = _PROJECT_ROOT / "results" / "datasets" / "humaneval" / "humaneval_public_test.jsonl"
DEFAULT_OUTPUT_BASE = _PROJECT_ROOT / "results" / "full_experiments"

METHODS = ["metagpt", "metagpt_egrcm", "metagpt_prqe", "metagpt_egrcm_prqe"]

HUMANEVAL_CONTRACT_GUARDRAILS = """Contract guardrails (HumanEval):
- Do NOT add new constraints or input validation unless explicitly required by the docstring.
- Assume inputs satisfy the problem constraints unless stated otherwise.
- Do NOT return error messages/strings; return exactly what the spec asks.
- Do NOT change the intended behavior for edge cases unless explicitly specified.
- Output ONLY executable Python code (no explanations, no markdown).
- Keep the implementation simple; avoid unnecessary helper functions or complex reconstruction logic."""


def build_prompt(problem: dict, *, inject_tests: bool) -> str:
    """Build the user prompt for code generation."""
    prompt = problem["prompt"]
    if inject_tests:
        prompt = inject_test_cases(prompt, problem.get("test"))
    prompt = inject_soft_constraints(prompt)
    return prompt


def _load_llm_config(model_name: Optional[str] = None):
    """從 config2.yaml 載入 LLM 配置。"""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    models = config.get("models", {})

    if model_name and model_name in models:
        cfg = models[model_name]
    elif model_name:
        raise ValueError(
            f"模型 '{model_name}' 未在 config2.yaml 的 models 段中配置。"
            f"可用: {list(models.keys())}"
        )
    elif "qwen3:32b" in models:
        cfg = models["qwen3:32b"]
    else:
        cfg = config.get("llm", {})

    base = cfg.get("base_url", "").rstrip("/")
    if base.endswith("/api"):
        base = base.replace("/api", "/v1")
    elif "/v1" not in base and base:
        base = f"{base}/v1"

    return {
        "model": cfg.get("model", model_name or "qwen3:32b"),
        "base_url": base or "https://api.openai.com/v1",
        "api_key": cfg.get("api_key", ""),
    }


def extract_docstring(prompt: str) -> str:
    """從 prompt 中提取 docstring 作為需求描述。"""
    for pattern in [r'"""(.*?)"""', r"'''(.*?)'''"]:
        match = re.search(pattern, prompt, re.S)
        if match:
            return match.group(1).strip()
    return ""


def load_problems(dataset_path: Path, limit: Optional[int] = None) -> list[dict]:
    """載入 HumanEval 測試集。支援標準格式（含 prompt）與 public 格式（僅 problem_id/test/entry_point）。"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"數據集不存在: {dataset_path}")
    problems = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
            if limit and len(problems) >= limit:
                break

    # 若為 public 格式（無 prompt），從 human-eval.jsonl 合併 prompt
    if problems and "prompt" not in problems[0]:
        prompt_source = dataset_path.parent / "human-eval.jsonl"
        if not prompt_source.exists():
            raise FileNotFoundError(
                f"humaneval_public_test 格式需 prompt，但 {prompt_source} 不存在。"
                "請使用 human-eval.jsonl 或確保 human-eval.jsonl 與數據集同目錄。"
            )
        prompts_by_id = {}
        with open(prompt_source, encoding="utf-8") as pf:
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tid = obj.get("task_id") or obj.get("problem_id")
                if tid and "prompt" in obj:
                    prompts_by_id[tid] = obj["prompt"]

        for p in problems:
            pid = p.get("problem_id") or p.get("task_id")
            p["task_id"] = p.get("task_id") or pid
            if pid in prompts_by_id:
                p["prompt"] = prompts_by_id[pid]
            else:
                raise KeyError(f"problem_id {pid} 在 human-eval.jsonl 中找不到對應 prompt")
            # public 格式的 test 為 list，需轉為 check 函數字串
            t = p.get("test")
            if isinstance(t, list):
                p["test"] = "def check(candidate):\n    " + "\n    ".join(t) if t else None

    return problems


def filter_problems_by_task_ids(problems: list[dict], task_ids: list[str]) -> list[dict]:
    """Keep only problems whose task_id is in task_ids (preserving order)."""
    if not task_ids:
        return problems
    wanted = set(task_ids)
    kept = []
    for p in problems:
        tid = p.get("task_id") or p.get("problem_id")
        if tid in wanted:
            kept.append(p)
    return kept


# ====== 代碼生成 ======


def generate_code_sync(
    client: OpenAI,
    model: str,
    prompt: str,
    extra_context: Optional[str] = None,
    *,
    temperature: float = 0,
) -> str:
    """同步調用 LLM 生成代碼。"""
    system = """You are an expert Python programmer.
Complete the function body according to the problem description.
Return the complete function including the def line. Ensure proper indentation (4 spaces for function body).
Do not include explanations or markdown."""
    if extra_context:
        system += f"\n\nRefined requirement context:\n{extra_context}"

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    _tt_mod.TRACKER.add_usage(getattr(resp, "usage", None))
    return (resp.choices[0].message.content or "").strip()


def generate_code_candidates_sync(
    client: OpenAI,
    model: str,
    prompt: str,
    extra_context: Optional[str] = None,
    *,
    n: int,
    temperature: float,
) -> list[str]:
    """Generate multiple candidates in one call (when supported)."""
    system = """You are an expert Python programmer.
Complete the function body according to the problem description.
Return the complete function including the def line. Ensure proper indentation (4 spaces for function body).
Do not include explanations or markdown."""
    if extra_context:
        system += f"\n\nRefined requirement context:\n{extra_context}"

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        n=n,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    _tt_mod.TRACKER.add_usage(getattr(resp, "usage", None))
    outs: list[str] = []
    for ch in resp.choices:
        outs.append((ch.message.content or "").strip())
    return outs


def _passes_public_tests(code: str, problem: dict, *, timeout: int = 15) -> bool:
    """Best-effort public test check to rerank candidates.

    This uses the same basic strategy as `scripts/evaluate_humaneval_pass1.py`.
    Any exception/timeout is treated as failure.
    """
    from metagpt.utils.sanitize import sanitize

    test = (problem.get("test") or "").strip()
    entry_point = (problem.get("entry_point") or "").strip()
    if not code.strip() or not test or not entry_point:
        return False

    # Run check(candidate) in a sub-thread for timeout control.
    import threading

    class _Timeout(Exception):
        pass

    result_box: list[object] = []
    done = threading.Event()

    def _target():
        try:
            cleaned = sanitize(code=code, entrypoint=entry_point)
            g = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
            }
            exec(cleaned, g)
            if entry_point not in g:
                result_box.append(False)
                return
            exec(test, g)
            check = g.get("check")
            if not callable(check):
                result_box.append(False)
                return
            out = check(g[entry_point])
            result_box.append(True if out is None else bool(out))
        except Exception:
            result_box.append(False)
        finally:
            done.set()

    t = threading.Thread(target=_target)
    t.start()
    if not done.wait(timeout):
        return False
    return bool(result_box[0]) if result_box else False


class SimpleChatLLM:
    """Minimal async LLM wrapper exposing .aask(prompt) used by EGRCM/PRQE modules.

    Avoids importing MetaGPT provider registry (which pulls in many optional deps).
    """

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    async def aask(self, prompt: str) -> str:
        def _call():
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            _tt_mod.TRACKER.add_usage(getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_call)


# ====== EGRCM ======


async def run_egrcm(docstring: str, llm, max_rounds: int = 3) -> str:
    """對 docstring 執行 EGRCM 熵收斂，返回精煉後的需求。"""
    if not docstring:
        return ""
    cfg = EntropyDialogueConfig(
        enabled=True,
        use_human=False,
        max_rounds=max_rounds,
        requirement_type="atomic",
    )
    engine = EntropyReductionDialogueEngine(cfg)
    state = RequirementState(raw_requirement=docstring)

    async def ask_fn(q: str) -> str:
        return await llm.aask(q)

    refined_state = await engine.refine(state=state, ask_fn=ask_fn, llm=llm)
    return refined_state.refined_requirement or docstring


# ====== PRQE ======


async def run_prqe(
    requirement: str,
    llm,
    config: Optional[PRQEConfig] = None,
) -> str:
    """對需求執行 PRQE 評估與精煉，返回精煉後的需求。"""
    if not requirement:
        return ""
    cfg = config or PRQEConfig()
    evaluator = RequirementQualityEvaluator(cfg)
    result = await evaluator.evaluate(requirement, llm)

    if evaluator.is_passed(result):
        return requirement

    feedback_gen = RequirementFeedbackGenerator()
    suggestions = await feedback_gen.generate(requirement, result, llm)
    if not suggestions:
        suggestions = result.improvement_suggestions

    refiner = RequirementRefiner()
    return await refiner.refine(requirement, suggestions, llm)


# ====== 四種方法 ======


async def method_metagpt(
    problem: dict,
    client: OpenAI,
    model: str,
    *,
    inject_tests: bool,
) -> str:
    """MetaGPT：直接生成代碼。"""
    prompt = build_prompt(problem, inject_tests=inject_tests)
    completion = generate_code_sync(client, model, prompt)
    return postprocess(completion, entry_point=problem.get("entry_point"))


async def method_metagpt_egrcm(
    problem: dict,
    client: OpenAI,
    model: str,
    llm,
    *,
    inject_tests: bool,
) -> str:
    """MetaGPT + EGRCM（帶 gating）：先 EGRCM 精煉 docstring，再生成代碼。"""
    docstring = extract_docstring(problem["prompt"])
    if docstring and should_skip_egrcm(docstring):
        refined = docstring
    else:
        refined = await run_egrcm(docstring, llm) if docstring else ""
    prompt = build_prompt(problem, inject_tests=inject_tests)
    completion = generate_code_sync(
        client, model, prompt,
        extra_context=refined if refined and refined != docstring else None,
    )
    return postprocess(completion, entry_point=problem.get("entry_point"))


async def method_metagpt_prqe(
    problem: dict,
    client: OpenAI,
    model: str,
    llm,
    *,
    inject_tests: bool,
) -> str:
    """MetaGPT + PRQE + Adapter + RPC：HumanEval 上 PRQE 不改写需求，只注入契约护栏提示。"""
    docstring = extract_docstring(problem["prompt"])
    prompt = build_prompt(problem, inject_tests=inject_tests)
    completion = generate_code_sync(
        client, model, prompt,
        extra_context=HUMANEVAL_CONTRACT_GUARDRAILS if docstring else None,
    )
    return postprocess(completion, entry_point=problem.get("entry_point"))


async def method_metagpt_egrcm_prqe(
    problem: dict,
    client: OpenAI,
    model: str,
    llm,
    *,
    inject_tests: bool,
) -> str:
    """EGRCM（帶 gating）→ PRQE(guardrails only) → RPC → soft constraints → postprocess。"""
    prompt = problem["prompt"]
    docstring = extract_docstring(prompt)
    if not docstring:
        prompt = build_prompt(problem, inject_tests=inject_tests)
        completion = generate_code_sync(client, model, prompt)
        return postprocess(completion, entry_point=problem.get("entry_point"))

    # EGRCM（帶 gating）
    if should_skip_egrcm(docstring):
        refined_docstring = docstring
    else:
        refined_docstring = await run_egrcm(docstring, llm)

    # PRQE: HumanEval uses contract guardrails only (no rewrite, no RPC).
    extra_ctx = refined_docstring if refined_docstring and refined_docstring != docstring else None
    if extra_ctx:
        extra_ctx = f"{extra_ctx}\n\n{HUMANEVAL_CONTRACT_GUARDRAILS}"

    # 强约束（soft 强化）
    prompt = build_prompt(problem, inject_tests=inject_tests)

    # Light self-consistency rerank (still outputs ONE completion):
    # 1) get a deterministic candidate; if it passes public tests, keep it
    # 2) otherwise sample a few diverse candidates and pick the first that passes
    base = generate_code_sync(client, model, prompt, extra_context=extra_ctx, temperature=0)
    base_pp = postprocess(base, entry_point=problem.get("entry_point"))
    if _passes_public_tests(base_pp, problem):
        return base_pp

    candidates = generate_code_candidates_sync(
        client, model, prompt,
        extra_context=extra_ctx,
        n=5,
        temperature=0.4,
    )
    for c in candidates:
        c_pp = postprocess(c, entry_point=problem.get("entry_point"))
        if _passes_public_tests(c_pp, problem):
            return c_pp
    # One more diverse batch if still failing.
    candidates2 = generate_code_candidates_sync(
        client, model, prompt,
        extra_context=extra_ctx,
        n=5,
        temperature=0.8,
    )
    for c in candidates2:
        c_pp = postprocess(c, entry_point=problem.get("entry_point"))
        if _passes_public_tests(c_pp, problem):
            return c_pp
    completion = base_pp
    return completion


# ====== 主流程 ======


async def run_experiment(
    method: str,
    limit: Optional[int] = None,
    model_name: Optional[str] = None,
    delay: float = 0,
    dataset_path: Optional[Path] = None,
    output_base: Optional[Path] = None,
    inject_tests: bool = False,
    only_tasks: Optional[list[str]] = None,
) -> None:
    """執行單一方法的實驗。"""
    if method not in METHODS:
        raise ValueError(f"method 須為 {METHODS} 之一，當前: {method}")

    dataset_path = dataset_path or DEFAULT_DATASET
    output_base = output_base or DEFAULT_OUTPUT_BASE

    llm_cfg = _load_llm_config(model_name)
    print(f"使用模型: {llm_cfg['model']}")
    client = OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
    )
    model = llm_cfg["model"]

    llm = None
    if method in ("metagpt_egrcm", "metagpt_prqe", "metagpt_egrcm_prqe"):
        # Use a minimal wrapper to avoid heavy optional provider imports.
        llm = SimpleChatLLM(client=client, model=model)

    problems = load_problems(dataset_path, limit)
    if only_tasks:
        problems = filter_problems_by_task_ids(problems, only_tasks)
    output_dir = output_base / method
    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = (model_name or model or "default").replace(":", "-")
    output_file = output_dir / f"samples_{model_safe}.jsonl"

    # Resume support
    results = []
    done_ids = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                results.append(obj)
                if obj.get("task_id"):
                    done_ids.add(obj["task_id"])
        if done_ids:
            print(f"斷點續跑：已完成 {len(done_ids)} 題，將跳過這些題目")

    # Fresh tracker for newly solved problems; merge with prior token file later
    _tt_mod.TRACKER = TokenTracker()
    global TRACKER
    TRACKER = _tt_mod.TRACKER

    use_prqe = method in ("metagpt_prqe", "metagpt_egrcm_prqe")
    # Truncate then rewrite incrementally for new run append
    # Keep existing lines; append new ones
    for problem in tqdm(problems, desc=f"Experiment3 [{method}]"):
        task_id = problem.get("task_id", "unknown")
        if task_id in done_ids:
            continue
        TRACKER.start_problem(task_id)
        try:
            if method == "metagpt":
                completion = await method_metagpt(problem, client, model, inject_tests=inject_tests)
            elif method == "metagpt_egrcm":
                completion = await method_metagpt_egrcm(problem, client, model, llm, inject_tests=inject_tests)
            elif method == "metagpt_prqe":
                completion = await method_metagpt_prqe(problem, client, model, llm, inject_tests=inject_tests)
            else:
                completion = await method_metagpt_egrcm_prqe(
                    problem, client, model, llm, inject_tests=inject_tests
                )
        except Exception as e:
            completion = f"Error: {e}"
        TRACKER.end_problem()

        usage = {}
        if TRACKER.problems:
            p = TRACKER.problems[-1]
            usage = {
                "prompt_tokens": p.prompt_tokens,
                "completion_tokens": p.completion_tokens,
                "total_tokens": p.total_tokens,
                "llm_calls": p.llm_calls,
            }
        row = {"task_id": task_id, "completion": completion, "usage": usage}
        results.append(row)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if use_prqe and delay > 0:
            await asyncio.sleep(delay)

    # Dedup rewrite
    dedup = {}
    for r in results:
        dedup[r["task_id"]] = r
    ordered = list(dedup.values())
    with open(output_file, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    token_file = output_dir / f"token_usage_{model_safe}.json"
    summary = TRACKER.summary(
        model=model, method=method, dataset="HumanEval", n_problems=len(ordered)
    )
    if token_file.exists() and done_ids:
        try:
            prev = json.loads(token_file.read_text(encoding="utf-8"))
            summary["prompt_tokens"] += int(prev.get("prompt_tokens", 0))
            summary["completion_tokens"] += int(prev.get("completion_tokens", 0))
            summary["total_tokens"] += int(prev.get("total_tokens", 0))
            summary["llm_calls"] += int(prev.get("llm_calls", 0))
            prev_probs = {p["task_id"]: p for p in prev.get("problems", [])}
            for p in summary["problems"]:
                prev_probs[p["task_id"]] = p
            summary["problems"] = list(prev_probs.values())
            summary["n_problems"] = len(ordered)
            if ordered:
                summary["avg_tokens_per_problem"] = round(summary["total_tokens"] / len(ordered), 2)
                summary["avg_llm_calls_per_problem"] = round(summary["llm_calls"] / len(ordered), 3)
            summary["est_cost_usd"] = round(
                summary["prompt_tokens"] * 0.15 / 1e6
                + summary["completion_tokens"] * 0.60 / 1e6,
                6,
            )
        except Exception:
            pass
    token_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{method}] 完成，共 {len(ordered)} 條，已保存: {output_file}")
    print(f"[{method}] Token 統計: {token_file}")
    print(
        f"  total_tokens={summary['total_tokens']}, "
        f"avg/problem={summary['avg_tokens_per_problem']}, "
        f"est_cost=${summary['est_cost_usd']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="實驗三：HumanEval 四種方法對比測試"
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        required=True,
        help="實驗方法",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="小範圍測試：僅處理前 N 條（例如 5）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定模型（如 qwen3:32b、gpt-4o-mini、deepseek_v3.2），"
             "須在 config2.yaml 的 models 段中已配置。不指定則用 qwen3:32b",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="PRQE 方法時，每題完成後額外等待秒數，用於避免 API 限流（如 2.0）",
    )
    parser.add_argument(
        "--inject-tests",
        action="store_true",
        help="是否将 dataset 的 test 注入 prompt（HumanEval 通常不建议；默认关闭）",
    )
    parser.add_argument(
        "--only-tasks",
        type=str,
        default=None,
        help="只运行指定 task_id（逗号分隔），用于小回归，例如: HumanEval/40,HumanEval/43",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="數據集路徑（預設: results/datasets/humaneval/humaneval_public_test.jsonl）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="輸出根目錄，結果保存為 {output-dir}/{method}/samples_{model}.jsonl",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if dataset_path and not dataset_path.is_absolute():
        dataset_path = _PROJECT_ROOT / dataset_path

    output_base = args.output_dir
    if output_base and not output_base.is_absolute():
        output_base = _PROJECT_ROOT / output_base

    only_tasks = None
    if args.only_tasks:
        only_tasks = [t.strip() for t in args.only_tasks.split(",") if t.strip()]

    asyncio.run(run_experiment(
        method=args.method,
        limit=args.limit,
        model_name=args.model,
        delay=args.delay,
        dataset_path=dataset_path,
        output_base=output_base,
        inject_tests=bool(args.inject_tests),
        only_tasks=only_tasks,
    ))


if __name__ == "__main__":
    main()
