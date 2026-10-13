#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import asyncio
import ast
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from metagpt.utils.sanitize import sanitize

from scripts.token_tracker import TRACKER, TokenTracker
import scripts.token_tracker as _tt_mod

# ====== 配置 ======

CONFIG_PATH = _PROJECT_ROOT / "config" / "config2.yaml"
DEFAULT_DATASET = _PROJECT_ROOT / "results" / "datasets" / "mbpp" / "mbpp_public_test.jsonl"
DEFAULT_OUTPUT_BASE = _PROJECT_ROOT / "results" / "full_experiments"

METHODS = ["metagpt", "metagpt_egrcm", "metagpt_prqe", "metagpt_egrcm_prqe"]

N_SAMPLES = 5
TEMPERATURE = 0.8
PRQE_APPLY_MAX_PROMPT_LEN = 300
EGRCM_MIN_TOKEN_OVERLAP = 0.35


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


def extract_requirement(prompt: str) -> str:
    """從 MBPP prompt 提取需求描述（instruction 部分，def 之前）。"""
    lines = prompt.strip().split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            req = "\n".join(lines[:i]).strip()
            return req if req else prompt
    return prompt


def load_problems(dataset_path: Optional[Path] = None, limit: Optional[int] = None) -> list[dict]:
    """載入 MBPP 測試集。支援標準格式（含 prompt）與 public 格式（僅 entry_point/test）。"""
    dataset_path = dataset_path or DEFAULT_DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"數據集不存在: {dataset_path}")
    problems = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["task_id"] = f"MBPP/{obj.get('task_id', len(problems))}"
            problems.append(obj)
            if limit and len(problems) >= limit:
                break

    # 若為 public 格式（無 prompt），從 mbpp_test.jsonl 合併 prompt（按 entry_point 匹配）
    if problems and "prompt" not in problems[0]:
        prompt_source = dataset_path.parent / "mbpp_test.jsonl"
        if not prompt_source.exists():
            raise FileNotFoundError(
                f"mbpp_public_test 格式需 prompt，但 {prompt_source} 不存在。"
                "請使用 mbpp_test.jsonl 或確保 mbpp_test.jsonl 與數據集同目錄。"
            )
        def _normalize_entry(name: Optional[str]) -> str:
            return re.sub(r"[^0-9a-zA-Z_]", "", (name or "")).strip().lower()

        def _xor_or_variants(name: Optional[str]) -> list[str]:
            """生成常见命名变体，用于处理数据集内函数名不一致的情况。

            例如：pair_xor_Sum <-> pair_OR_Sum（mbpp_test.jsonl 中曾出现 OR 实际表示 XOR）。
            """
            if not name:
                return []
            variants: list[str] = []
            s = name
            variants.append(s)
            variants.append(s.lower())
            # token 级别替换：xor <-> or
            variants.append(re.sub(r"(?i)(^|_)xor(_|$)", r"\1or\2", s))
            variants.append(re.sub(r"(?i)(^|_)or(_|$)", r"\1xor\2", s))
            # 兼容 mbpp_test 常见形式：OR 大写
            variants.append(re.sub(r"(?i)(^|_)xor(_|$)", r"\1OR\2", s))
            variants.append(re.sub(r"(?i)(^|_)or(_|$)", r"\1OR\2", s))
            # 去重并保序
            out: list[str] = []
            seen: set[str] = set()
            for v in variants:
                if v and v not in seen:
                    out.append(v)
                    seen.add(v)
            return out

        def _build_prompt_mapping_from_mbpp_test() -> dict[str, str]:
            """從 mbpp_test.jsonl 建立 entry_point -> prompt 的更鲁棒映射。

            注意：mbpp_test.jsonl 的 entry_point 有時不是目標函數（test 可能調用其他 def），
            因此除了 entry_point 外，還會從 code 中提取所有 def 的簽名，為每個 def 生成 prompt。
            """
            mapping: dict[str, str] = {}
            with open(prompt_source, encoding="utf-8") as pf:
                for raw in pf:
                    raw = raw.strip()
                    if not raw:
                        continue
                    o = json.loads(raw)
                    base_prompt = o.get("prompt") or ""
                    if not base_prompt:
                        continue

                    # 盡量提取 “instruction” 部分（def 之前），用於重建 prompt
                    instruction = base_prompt
                    if "\n\ndef " in base_prompt:
                        instruction = base_prompt.split("\n\ndef ", 1)[0].strip()
                    instr_l = instruction.lower()

                    # 1) 先用原始 prompt 覆蓋 entry_point（若存在）
                    ep = o.get("entry_point")
                    if ep and ep not in mapping:
                        mapping[ep] = base_prompt

                    # 2) 再从 code 里提取所有 def 的签名，为每个函数名生成 prompt
                    code = o.get("code") or ""
                    if code:
                        # 捕获形如 "def foo(bar, baz):" 的整行签名（不含换行）
                        for m in re.finditer(
                            r"(?m)^\s*(def\s+([A-Za-z_]\w*)\s*\([^)]*\))\s*:",
                            code,
                        ):
                            def_sig, fn = m.group(1).strip(), m.group(2).strip()
                            if not fn:
                                continue
                            if fn in mapping:
                                continue
                            rebuilt = f"{instruction}\n\n{def_sig}"
                            mapping[fn] = rebuilt
                            # 2.1) 若 instruction 明确提到 xor，但函数名用 OR（历史命名问题），补一个 xor 别名
                            if "xor" in instr_l and re.search(r"(?i)(^|_)or(_|$)", fn):
                                alias = re.sub(r"(?i)(^|_)or(_|$)", r"\1xor\2", fn)
                                if alias and alias not in mapping:
                                    mapping[alias] = rebuilt
            return mapping

        prompts_by_entry = _build_prompt_mapping_from_mbpp_test()
        prompts_by_norm = {_normalize_entry(k): v for k, v in prompts_by_entry.items()}

        def _infer_arity_from_tests(tests: Optional[list[str]]) -> Optional[int]:
            """从 public 测试用例里推断 candidate(...) 的参数个数（仅用于兜底生成 prompt）。"""
            if not tests:
                return None
            for s in tests:
                if "candidate(" not in s:
                    continue
                m = re.search(r"candidate\s*\((.*)\)", s)
                if not m:
                    continue
                inside = m.group(1).strip()
                if inside == "":
                    return 0
                # 计数顶层逗号（忽略括号/方括号/花括号/字符串内逗号）
                depth = 0
                in_str: Optional[str] = None
                escaped = False
                commas = 0
                for ch in inside:
                    if in_str:
                        if escaped:
                            escaped = False
                        elif ch == "\\":
                            escaped = True
                        elif ch == in_str:
                            in_str = None
                        continue
                    if ch in ("'", '"'):
                        in_str = ch
                        continue
                    if ch in "([{":
                        depth += 1
                        continue
                    if ch in ")]}":
                        depth = max(0, depth - 1)
                        continue
                    if ch == "," and depth == 0:
                        commas += 1
                return commas + 1
            return None

        def _synthetic_prompt(entry: str, tests: Optional[list[str]]) -> str:
            """当 mbpp_test.jsonl 找不到 prompt 时，为 public 数据生成最小 prompt。"""
            arity = _infer_arity_from_tests(tests)
            if arity is None:
                params = "*args"
            else:
                params = ", ".join([f"p{i}" for i in range(arity)])
            test_block = "\n".join(tests or [])
            return (
                "Write a Python function that passes the given tests.\n\n"
                f"Tests:\n{test_block}\n\n"
                f"def {entry}({params})"
            )

        for i, p in enumerate(problems):
            ep = p.get("entry_point")
            prompt = None
            # 先尝试精确匹配/规范化匹配，再尝试 xor<->or 等常见变体
            for cand in _xor_or_variants(ep):
                if cand in prompts_by_entry:
                    prompt = prompts_by_entry[cand]
                    break
                prompt = prompts_by_norm.get(_normalize_entry(cand))
                if prompt:
                    break
            if prompt:
                p["prompt"] = prompt
            else:
                # mbpp_public_test 可能包含 mbpp_test.jsonl 不覆盖的题：兜底生成 prompt，避免整批实验中断
                p["prompt"] = _synthetic_prompt(ep or "candidate", p.get("test"))
                print(
                    f"[WARN] entry_point '{ep}' 未在 mbpp_test.jsonl 中找到 prompt，已使用 synthetic prompt 兜底。",
                    file=sys.stderr,
                )
            # public 格式的 test 為 list，需轉為 check 函數字串
            t = p.get("test")
            if isinstance(t, list):
                p["test"] = "def check(candidate):\n    " + "\n    ".join(t) if t else None

    return problems


def _chat_create(client: OpenAI, **kwargs):
    """呼叫模型；對限流與暫時性錯誤做退避重試。"""
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = any(
                s in msg
                for s in ("rate", "429", "timeout", "timed out", "503", "502", "500", "overloaded")
            )
            if not retryable or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise last_err  # pragma: no cover


def _egrcm_gate_detail(requirement: str) -> dict:
    """說明此題為什麼會、或不會進入 EGRCM。"""
    text = (requirement or "").strip()
    if not text:
        return {
            "skipped": False,
            "reason": "需求文本為空，不呼叫 EGRCM。",
        }
    hits: list[str] = []
    if re.search(r"\bassert\s+", text):
        hits.append("含有 assert")
    if re.search(r"\bdef\s+\w+\s*\(", text):
        hits.append("含有函數簽名")
    if (
        ">>>" in text
        or re.search(r"assert\s+.*==\s*", text)
        or re.search(r"(?:example|e\.g\.|for example)\s*:?\s*\w", text, re.I)
    ):
        hits.append("含有輸入輸出示例")
    if hits:
        return {
            "skipped": True,
            "reason": "低熵門控跳過 EGRCM，因為抽取後的需求" + "、".join(hits) + "。此題沒有澄清與總結呼叫。",
        }
    return {
        "skipped": False,
        "reason": (
            "抽取需求時只保留函數定義之前的說明，這段文字沒有 assert、函數簽名或輸入輸出示例，"
            "低熵門控因此判定為高熵並執行 EGRCM。MBPP 的測試與 def 行在後段，沒有進入這次判斷，"
            "所以大多數題目仍會多出最多 3 輪澄清和 1 次總結。"
        ),
    }


def _question_id_from_prompt(prompt: str) -> str:
    text = prompt or ""
    # 總結提示本身含有「驗收條件」等字，必須先於具體問題關鍵詞判斷。
    if any(
        s in text
        for s in (
            "最终需求描述",
            "最終需求描述",
            "结构化、可验证",
            "結構化、可驗證",
            "可验证的最终需求",
        )
    ):
        return "summary"
    rules = (
        ("用户角色或使用场景", "target_user"),
        ("用戶角色或使用場景", "target_user"),
        ("技术约束、环境限制", "system_constraints"),
        ("技術約束、環境限制", "system_constraints"),
        ("哪一类", "constraint_type"),
        ("哪一類", "constraint_type"),
        ("依赖特定系统", "dependency_system"),
        ("依賴特定系統", "dependency_system"),
        ("验收条件或测试要点", "acceptance_criteria"),
        ("驗收條件或測試要點", "acceptance_criteria"),
        ("专有名词", "terminology"),
        ("專有名詞", "terminology"),
        ("哪个部分或模块", "applicable_subsystem"),
        ("哪個部分或模組", "applicable_subsystem"),
        ("优先级或可选性", "priority_optionality"),
        ("優先級或可選性", "priority_optionality"),
    )
    for needle, qid in rules:
        if needle in text:
            return qid
    return "unknown_question"


def _overlap_ratio(original_req: str, refined_req: str) -> float:
    original_tokens = _tokenize_for_overlap(original_req)
    refined_tokens = _tokenize_for_overlap(refined_req)
    if not original_tokens or not refined_tokens:
        return 0.0
    return len(original_tokens & refined_tokens) / max(len(original_tokens), 1)


def _compose_codegen_reason(meta: dict, sample_index: int, n_samples: int, temperature: float) -> str:
    branch = meta.get("branch") or "unknown"
    parts = [
        f"環節=代碼生成/{branch}，第 {sample_index + 1}/{n_samples} 次採樣（temperature={temperature}）"
    ]
    if sample_index > 0:
        parts.append(
            "該分支固定生成 5 個候選，生成階段不會因為前面的候選已經通過測試而停止，相同提示詞被重複發送"
        )
    extra_chars = int(meta.get("extra_context_chars") or 0)
    if extra_chars > 0:
        parts.append(
            f"EGRCM 精煉需求（{extra_chars} 字）寫入 system prompt，每一次採樣都會把這段上下文再發一次"
        )
    if meta.get("prqe_applied"):
        parts.append(meta.get("prqe_reason") or "已追加 PRQE 固定約束，提示詞變長")
    elif meta.get("prqe_reason"):
        parts.append(meta["prqe_reason"])
    if meta.get("egrcm_note"):
        parts.append(meta["egrcm_note"])
    return "；".join(parts)


def generate_code_sync(
    client: OpenAI,
    model: str,
    prompt: str,
    extra_context: Optional[str] = None,
    *,
    temperature: float = 0.0,
    call_meta: Optional[dict] = None,
) -> str:
    """同步調用 LLM 生成代碼。"""
    system = """You are an expert Python programmer.
Complete the function body according to the problem description.
Return the complete function including the def line. Ensure proper indentation (4 spaces for function body).
Do not include explanations or markdown."""
    if extra_context:
        system += f"\n\nRefined requirement context:\n{extra_context}"

    resp = _chat_create(
        client,
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    if call_meta is None:
        _tt_mod.TRACKER.add_usage(getattr(resp, "usage", None))
    else:
        meta = dict(call_meta)
        meta["system_chars"] = len(system)
        meta["user_chars"] = len(prompt)
        meta["extra_context_chars"] = len(extra_context or "")
        meta["temperature"] = temperature
        _tt_mod.TRACKER.add_usage(getattr(resp, "usage", None), meta=meta)
    return (resp.choices[0].message.content or "").strip()


def _run_with_timeout(func, args: tuple, timeout: int = 15):
    """在子线程中执行函数；超时则抛出 TimeoutError。"""
    result = []
    stop_event = threading.Event()

    def target():
        try:
            result.append(func(*args))
        except Exception as e:
            result.append(e)
        finally:
            stop_event.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    is_timeout = not stop_event.wait(timeout)
    if is_timeout:
        raise TimeoutError("Function execution timed out")
    if not result:
        return None
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]


def _candidate_passes_mbpp(problem: dict, completion_code: str, timeout: int = 15) -> bool:
    """执行 MBPP public 测试（problem["test"] 已经是 def check(candidate): ...）。"""
    solution = (completion_code or "").strip()
    if not solution or solution.startswith("Error:"):
        return False

    entry_point = problem.get("entry_point")
    test = problem.get("test")
    if not entry_point or not test:
        return False

    try:
        solution = sanitize(code=solution, entrypoint=entry_point)
    except Exception:
        return False

    global_dict: dict[str, Any] = {
        "__builtins__": __builtins__,
        "math": __import__("math"),
        "hashlib": __import__("hashlib"),
        "re": __import__("re"),
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        "Optional": Optional,
        "Any": Any,
    }

    try:
        exec(solution, global_dict)
        if entry_point not in global_dict:
            return False
        exec(test, global_dict)
        check = global_dict.get("check")
        if not callable(check):
            return False
        _run_with_timeout(check, (global_dict[entry_point],), timeout=timeout)
        return True
    except Exception:
        return False


def _signature_matches_problem(problem: dict, completion_code: str) -> bool:
    """Hard filter for signature drift: function name/arity should match problem tests."""
    entry_point = problem.get("entry_point")
    if not entry_point or not completion_code:
        return False
    try:
        tree = ast.parse(completion_code)
    except Exception:
        return False
    func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            func_node = node
            break
    if func_node is None:
        return False
    # Infer expected arity from tests: candidate(...)
    test = problem.get("test") or ""
    m = re.search(r"candidate\s*\((.*)\)", test)
    if not m:
        return True
    inside = m.group(1).strip()
    expected_arity = 0 if inside == "" else inside.count(",") + 1
    actual_arity = len(func_node.args.args)
    return actual_arity == expected_arity


def _candidate_eval_mbpp(
    problem: dict,
    completion_code: str,
    timeout: int = 15,
) -> tuple[bool, bool, bool, str]:
    """Return (passed_all_tests, compiles, signature_ok, err_msg)."""
    solution = (completion_code or "").strip()
    if not solution or solution.startswith("Error:"):
        return (False, False, False, "empty_or_error")
    entry_point = problem.get("entry_point")
    test = problem.get("test")
    if not entry_point or not test:
        return (False, False, False, "missing_problem_fields")

    signature_ok = _signature_matches_problem(problem, solution)
    try:
        solution = sanitize(code=solution, entrypoint=entry_point)
    except Exception as e:
        return (False, False, signature_ok, f"sanitize_error: {e}")

    global_dict: dict[str, Any] = {
        "__builtins__": __builtins__,
        "math": __import__("math"),
        "hashlib": __import__("hashlib"),
        "re": __import__("re"),
        "itertools": __import__("itertools"),
        "collections": __import__("collections"),
        "functools": __import__("functools"),
        "statistics": __import__("statistics"),
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        "Optional": Optional,
        "Any": Any,
    }

    try:
        exec(solution, global_dict)
        if entry_point not in global_dict:
            return (False, True, signature_ok, "entry_point_not_defined")
        fn = global_dict[entry_point]
        exec(test, global_dict)
        check = global_dict.get("check")
        if not callable(check):
            return (False, True, signature_ok, "check_not_callable")
        _run_with_timeout(check, (fn,), timeout=timeout)
        return (True, True, signature_ok, "")
    except Exception as e:
        return (False, True, signature_ok, str(e))


def _repair_candidate_once(
    client: OpenAI,
    model: str,
    problem: dict,
    bad_code: str,
    err_msg: str,
    *,
    temperature: float,
    call_meta: Optional[dict] = None,
) -> str:
    """One-shot repair: keep signature unchanged, fix runtime/assert errors only."""
    prompt = problem.get("prompt") or ""
    entry_point = problem.get("entry_point") or "candidate"
    test = problem.get("test") or ""
    repair_prompt = (
        f"{prompt}\n\n"
        "Your previous solution failed. Fix it with minimal edits.\n"
        "Hard requirements:\n"
        f"- Keep function name exactly: {entry_point}\n"
        "- Keep parameter count and order unchanged\n"
        "- Do not add explanation, only return valid Python code\n\n"
        f"Failure signal:\n{err_msg}\n\n"
        f"Tests:\n{test}\n\n"
        f"Previous code:\n{bad_code}"
    )
    failure = (err_msg or "assertion/runtime failure").replace("\n", " ")[:180]
    meta = dict(call_meta or {})
    branch = meta.get("branch") or "unknown"
    meta.update(
        {
            "stage": "repair",
            "issue": "repair_after_test_failure",
            "failure": failure,
            "prompt_chars": len(repair_prompt),
            "reason": (
                f"環節=失敗修復/{branch}；該分支的候選都沒有通過 MBPP 測試（{failure}），"
                "於是把原題、失敗訊息、測試程式與上一版程式碼重新送給模型，提示詞明顯變長"
            ),
            "drivers": list(meta.get("drivers") or [])
            + ["測試失敗觸發修復", "修復提示詞含原題、測試與失敗程式碼"],
        }
    )
    fixed = generate_code_sync(
        client,
        model,
        repair_prompt,
        temperature=temperature,
        call_meta=meta if call_meta is not None else None,
    )
    fixed = clean_code(fixed)
    return postprocess(fixed, entry_point=entry_point)


def should_apply_prqe(problem: dict) -> bool:
    """PRQE gating：仅当 prompt 足够短时才追加 PRQE 指导，避免系统性负迁移。"""
    prompt = problem.get("prompt") or ""
    if len(prompt) < PRQE_APPLY_MAX_PROMPT_LEN:
        return True
    return False


def should_apply_prqe_strict(problem: dict) -> bool:
    """Stricter PRQE gating for metagpt_prqe to reduce over-application."""
    prompt = (problem.get("prompt") or "").lower()
    if not prompt:
        return False
    if len(prompt) >= 220:
        return False
    # Heuristic: only enable when requirement text appears constraint-heavy.
    req = extract_requirement(prompt)
    signals = [
        " if ",
        " when ",
        " exactly ",
        " should ",
        " must ",
        " not ",
        " without ",
        " at least ",
        " return ",
    ]
    hit = sum(1 for s in signals if s in req)
    return hit >= 2


def _tokenize_for_overlap(text: str) -> set[str]:
    """Simple lexical tokens for requirement-overlap gating."""
    return set(re.findall(r"[a-zA-Z_]{2,}", (text or "").lower()))


def _is_egrcm_refinement_safe(original_req: str, refined_req: str) -> bool:
    """Guard against EGRCM over-rewrite: keep only semantically close refinements.

    We use a conservative lexical-overlap heuristic. If overlap is too small,
    refined requirement is treated as risky and will be ignored.
    """
    if not original_req or not refined_req:
        return False
    o_tokens = _tokenize_for_overlap(original_req)
    r_tokens = _tokenize_for_overlap(refined_req)
    if not o_tokens or not r_tokens:
        return False
    overlap = len(o_tokens & r_tokens) / max(len(o_tokens), 1)
    return overlap >= EGRCM_MIN_TOKEN_OVERLAP


def clean_code(code: str) -> str:
    """Mandatory cleaning: strip markdown fences and keep only the code body."""
    code = (code or "").strip()
    if not code:
        return code

    # Remove leading/trailing markdown fences if present.
    # NOTE: regexes must be valid; any re.error here will break generation/eval.
    code = re.sub(r"^```(?:python|py)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"```\s*$", "", code, flags=re.IGNORECASE)

    # Keep from the first "real code line" (def/import/from/class).
    m = re.search(
        r"(?m)^\s*(?:from\s+\w+|import\s+[\w\.]+|def\s+\w+\s*\(|class\s+\w+)",
        code,
    )
    if m:
        return code[m.start() :].strip()
    return code


# 固定短约束：仅 4 条，不接 PRQE 长文本。用于 EGRCM+PRQE 时减轻过度约束。
PRQE_FIXED_CONSTRAINTS_ONLY = (
    "### Additional Constraints:\n"
    "- Follow the exact function signature\n"
    "- Do not use undefined variables\n"
    "- Ensure return type matches problem requirement\n"
    "- Prefer simple and direct implementation\n"
)


def _format_prqe_guidance(prqe_refined: str, fixed_only: bool = False) -> str:
    """Format PRQE output as auxiliary constraints (do NOT overwrite original prompt).

    When fixed_only=True, only append the 4 fixed bullets (no prqe_refined body).
    Used by metagpt_egrcm_prqe to avoid double-refinement over-constraining.
    """
    if fixed_only:
        return PRQE_FIXED_CONSTRAINTS_ONLY
    prqe_refined = (prqe_refined or "").strip()
    if not prqe_refined:
        return ""
    return (
        "### Additional Constraints:\n"
        "- Follow the exact function signature\n"
        "- Do not use undefined variables\n"
        "- Ensure return type matches problem requirement\n"
        "- Prefer simple and direct implementation\n\n"
        f"{prqe_refined}"
    )


def _candidate_compiles(code: str) -> bool:
    """Syntax-level fallback: if no candidate passes tests, pick the first compilable one."""
    try:
        compile(code, "<mbpp_candidate>", "exec")
        return True
    except Exception:
        return False


def _generate_candidates_and_select_mbpp(
    problem: dict,
    client: OpenAI,
    model: str,
    final_prompt: str,
    *,
    extra_context: Optional[str],
    candidate_n: int,
    temperature: float,
    timeout: int = 15,
    call_meta: Optional[dict] = None,
) -> str:
    """Generate candidates and rerank with one optional repair round."""
    entry_point = problem.get("entry_point")
    candidates: list[str] = []
    temp_schedule = [0.0, 0.2, 0.5, 0.8, temperature]
    temps = temp_schedule[:candidate_n]
    while len(temps) < candidate_n:
        temps.append(temperature)

    for sample_index, t in enumerate(temps):
        meta = None
        if call_meta is not None:
            meta = dict(call_meta)
            meta["stage"] = "codegen"
            meta["sample_index"] = sample_index
            meta["issue"] = "extra_sample" if sample_index else meta.get("issue", "codegen")
            meta["extra_context_chars"] = len(extra_context or "")
            meta["user_chars"] = len(final_prompt)
            meta["reason"] = _compose_codegen_reason(meta, sample_index, len(temps), t)
            drivers = list(meta.get("drivers") or [])
            if sample_index > 0:
                drivers.append("固定 5 次採樣且不提前停止")
            if extra_context:
                drivers.append("system prompt 重複攜帶 EGRCM 精煉需求")
            if meta.get("prqe_applied"):
                drivers.append("提示詞追加 PRQE 固定約束")
            meta["drivers"] = drivers
        raw_completion = generate_code_sync(
            client,
            model,
            final_prompt,
            extra_context=extra_context,
            temperature=t,
            call_meta=meta,
        )
        raw_completion = clean_code(raw_completion)
        completion_pp = postprocess(raw_completion, entry_point=entry_point)
        candidates.append(completion_pp)

    # Light-weight rerank: full pass > signature OK > compile.
    scored: list[tuple[str, bool, bool, bool, str]] = []
    for c in candidates:
        passed_all, compiles, signature_ok, err = _candidate_eval_mbpp(problem, c, timeout=timeout)
        scored.append((c, passed_all, compiles, signature_ok, err))
        if passed_all and signature_ok:
            return c

    if not scored:
        return ""

    def _rank_key(item: tuple[str, bool, bool, bool, str]) -> tuple[int, int, int]:
        _, passed_all, compiles, signature_ok, _ = item
        return (1 if passed_all else 0, 1 if signature_ok else 0, 1 if compiles else 0)

    scored.sort(key=_rank_key, reverse=True)
    best_code, best_passed_all, _, best_sig_ok, best_err = scored[0]

    # One repair round if top candidate still fails.
    if not best_passed_all:
        repaired = _repair_candidate_once(
            client,
            model,
            problem,
            best_code,
            best_err or "assertion/runtime failure",
            temperature=0.2,
            call_meta=None if call_meta is None else {**call_meta, "sample_index": len(temps)},
        )
        r_passed_all, _, r_sig_ok, _ = _candidate_eval_mbpp(problem, repaired, timeout=timeout)
        if r_passed_all and r_sig_ok:
            return repaired

    # Fallback: best scored candidate, then first compilable, then first.
    if best_sig_ok:
        return best_code
    for c, _, compiles, _, _ in scored:
        if compiles:
            return c
    return candidates[0]


def _temperature_list(n: int) -> list[float]:
    """候选生成温度序列：N 越大，候选多样性越强。"""
    if n <= 1:
        return [0.0]
    # 0.0 作为保底；其余温度逐步增加
    base = [0.0, 0.2, 0.4, 0.6, 0.8]
    return base[:n]


def _pick_first_passing_completion(
    problem: dict,
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    extra_context: Optional[str],
    candidate_n: int,
    timeout: int = 15,
) -> str:
    """生成多个候选，并按 MBPP 测试选择第一条能通过的候选。"""
    entry_point = problem.get("entry_point")
    temps = _temperature_list(candidate_n)
    first_completion_pp: Optional[str] = None

    for t in temps:
        completion = generate_code_sync(
            client,
            model,
            prompt,
            extra_context=extra_context,
            temperature=t,
        )
        completion_pp = postprocess(completion, entry_point=entry_point)
        if first_completion_pp is None:
            first_completion_pp = completion_pp
        if _candidate_passes_mbpp(problem, completion_pp, timeout=timeout):
            return completion_pp

    return first_completion_pp or ""


async def run_egrcm(requirement: str, llm, max_rounds: int = 3) -> str:
    """對需求執行 EGRCM 熵收斂。"""
    if not requirement:
        return ""
    cfg = EntropyDialogueConfig(
        enabled=True,
        use_human=False,
        max_rounds=max_rounds,
        requirement_type="atomic",
    )
    engine = EntropyReductionDialogueEngine(cfg)
    state = RequirementState(raw_requirement=requirement)
    clarify_index = {"n": 0}

    class _StageLLM:
        async def aask(self, prompt: str) -> str:
            question_id = _question_id_from_prompt(prompt)
            if question_id == "summary":
                meta = {
                    "stage": "egrcm_summary",
                    "branch": "egrcm",
                    "issue": "egrcm_summary_rewrites_requirement",
                    "question_id": "summary",
                    "prompt_chars": len(prompt),
                    "reason": (
                        "環節=EGRCM 需求總結；澄清結束後，原始需求與每一輪回答被重新拼進總結提示，"
                        f"再生成將注入後續代碼生成的精煉需求（本次提示 {len(prompt)} 字）"
                    ),
                    "drivers": [
                        "總結提示包含原始需求與全部澄清回答",
                        "精煉結果之後還會在代碼生成分支中重複發送",
                    ],
                }
            else:
                round_no = clarify_index["n"] + 1
                clarify_index["n"] += 1
                meta = {
                    "stage": "egrcm_clarify",
                    "branch": "egrcm",
                    "issue": "egrcm_gate_not_skipped",
                    "question_id": question_id,
                    "sample_index": round_no - 1,
                    "prompt_chars": len(prompt),
                    "reason": (
                        f"環節=EGRCM 澄清問答，第 {round_no} 輪（問題 {question_id}）；"
                        "抽取後的需求沒有被低熵門控跳過，完整需求被寫進提問提示。"
                        f"本輪提示 {len(prompt)} 字"
                    ),
                    "drivers": [
                        "低熵門控未跳過",
                        "每一輪提問都重複攜帶原始需求",
                    ],
                }
            return await llm.aask(prompt, call_meta=meta)

    proxy = _StageLLM()

    async def ask_fn(q: str) -> str:
        return await proxy.aask(q)

    refined_state = await engine.refine(state=state, ask_fn=ask_fn, llm=proxy)
    return refined_state.refined_requirement or requirement


async def run_prqe(requirement: str, llm, config: Optional[PRQEConfig] = None) -> str:
    """對需求執行 PRQE 評估與精煉。"""
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


async def method_metagpt(problem: dict, client: OpenAI, model: str) -> str:
    """MetaGPT：直接生成代碼。"""
    prompt = inject_test_cases(problem["prompt"], problem.get("test"))
    prompt = inject_soft_constraints(prompt)
    completion = generate_code_sync(client, model, prompt)
    completion = clean_code(completion)
    return postprocess(completion, entry_point=problem.get("entry_point"))


async def method_metagpt_egrcm(
    problem: dict, client: OpenAI, model: str, llm
) -> str:
    """MetaGPT + EGRCM（帶 gating）：先 EGRCM 精煉需求，再生成代碼。"""
    req = extract_requirement(problem["prompt"])
    if req and should_skip_egrcm(req):
        refined = req
    else:
        refined = await run_egrcm(req, llm) if req else ""
    # EGRCM confidence gate: avoid over-rewrite drift.
    if req and refined and refined != req and not _is_egrcm_refinement_safe(req, refined):
        refined = req
    prompt = inject_test_cases(problem["prompt"], problem.get("test"))
    prompt = inject_soft_constraints(prompt)
    extra_context = refined if refined and refined != req else None

    completion = generate_code_sync(
        client,
        model,
        prompt,
        extra_context=extra_context,
        temperature=0.0,
    )
    completion = clean_code(completion)
    return postprocess(completion, entry_point=problem.get("entry_point"))


async def method_metagpt_prqe(
    problem: dict, client: OpenAI, model: str, llm
) -> str:
    """MetaGPT + PRQE（auxiliary guidance）+ rerank.

    CRITICAL: do NOT overwrite/replace the original instruction; only append PRQE guidance.
    """
    # Use strict gating + fixed short constraints to avoid PRQE over-steering.
    apply = should_apply_prqe_strict(problem)
    prqe_guidance = _format_prqe_guidance("", fixed_only=True) if apply else ""

    # Build from original prompt, append short guidance only when strictly needed.
    prompt = inject_test_cases(problem["prompt"], problem.get("test"))
    prompt = inject_soft_constraints(prompt)
    if prqe_guidance:
        prompt = prompt + "\n\n" + prqe_guidance

    return _generate_candidates_and_select_mbpp(
        problem,
        client,
        model,
        prompt,
        extra_context=None,
        candidate_n=N_SAMPLES,
        temperature=TEMPERATURE,
        timeout=15,
    )


async def method_metagpt_egrcm_prqe(
    problem: dict, client: OpenAI, model: str, llm
) -> str:
    """Four-branch ensemble for egrcm+prqe:
    A) egrcm+prqe, B) prqe-only, C) egrcm-only, D) baseline.
    Final selection uses local MBPP eval, tie-breaker A > C > B > D.
    """
    prompt_base = problem["prompt"]
    req = extract_requirement(prompt_base)
    task_id = problem.get("task_id", "unknown")
    gate = _egrcm_gate_detail(req)
    audit: dict[str, Any] = {
        "task_id": task_id,
        "requirement_chars": len(req or ""),
        "prompt_chars": len(prompt_base or ""),
        "egrcm": {
            "skipped": gate["skipped"],
            "gate_reason": gate["reason"],
        },
    }
    _tt_mod.TRACKER.problem_audits[task_id] = audit

    # 1) Unified EGRCM refinement for A/C branches.
    if req and should_skip_egrcm(req):
        refined_egrcm = req
    else:
        refined_egrcm = await run_egrcm(req, llm) if req else ""
    overlap = _overlap_ratio(req, refined_egrcm) if req and refined_egrcm else 0.0
    rejected = bool(
        req
        and refined_egrcm
        and refined_egrcm != req
        and not _is_egrcm_refinement_safe(req, refined_egrcm)
    )
    if rejected:
        refined_egrcm = req
    egrcm_note = ""
    if rejected:
        egrcm_note = (
            f"EGRCM 輸出與原需求的詞重疊只有 {overlap:.2f}，低於 {EGRCM_MIN_TOKEN_OVERLAP}，"
            "精煉結果被丟棄、不會注入代碼生成，但澄清與總結的 token 已經花掉"
        )
    elif gate["skipped"]:
        egrcm_note = gate["reason"]
    elif refined_egrcm and refined_egrcm != req:
        egrcm_note = (
            f"EGRCM 精煉結果被保留（詞重疊 {overlap:.2f}，{len(refined_egrcm)} 字），"
            "並會注入分支 A 與分支 C 的每一次代碼生成"
        )
    audit["egrcm"].update(
        {
            "overlap": round(overlap, 4),
            "refinement_kept": bool(refined_egrcm and refined_egrcm != req and not rejected),
            "refined_chars": len(refined_egrcm or "") if not rejected else 0,
            "reject_reason": egrcm_note if rejected else "",
        }
    )

    egrcm_ctx = refined_egrcm if refined_egrcm and refined_egrcm != req else None

    # 2) Base prompt shared by all branches.
    base = inject_test_cases(prompt_base, problem.get("test"))
    base = inject_soft_constraints(base)

    # 3) Branch A: egrcm + prqe (current default strategy).
    apply_a = should_apply_prqe(problem)
    prqe_a = _format_prqe_guidance("", fixed_only=True) if apply_a else ""
    prompt_a = base + ("\n\n" + prqe_a if prqe_a else "")
    prqe_a_reason = (
        f"分支 A：原 prompt 長度 {len(prompt_base)} < {PRQE_APPLY_MAX_PROMPT_LEN}，追加 4 條固定 PRQE 約束"
        if apply_a
        else f"分支 A：原 prompt 長度 {len(prompt_base)} ≥ {PRQE_APPLY_MAX_PROMPT_LEN}，不追加 PRQE，避免長提示再變長"
    )
    cand_a = _generate_candidates_and_select_mbpp(
        problem,
        client,
        model,
        prompt_a,
        extra_context=egrcm_ctx,
        candidate_n=N_SAMPLES,
        temperature=TEMPERATURE,
        timeout=15,
        call_meta={
            "branch": "A_egrcm_prqe",
            "issue": "branch_a_egrcm_and_prqe",
            "prqe_applied": apply_a,
            "prqe_reason": prqe_a_reason,
            "egrcm_note": egrcm_note,
            "drivers": ["四分支集成中的 EGRCM+PRQE 分支"],
        },
    )

    # 4) Branch B: prqe-only with stricter gate.
    apply_b = should_apply_prqe_strict(problem)
    prqe_b = _format_prqe_guidance("", fixed_only=True) if apply_b else ""
    prompt_b = base + ("\n\n" + prqe_b if prqe_b else "")
    prqe_b_reason = (
        "分支 B：需求較短且含多個約束詞，嚴格門控允許追加固定 PRQE 約束"
        if apply_b
        else "分支 B：未達到嚴格 PRQE 門控（過長或約束詞不足），不追加 PRQE"
    )
    cand_b = _generate_candidates_and_select_mbpp(
        problem,
        client,
        model,
        prompt_b,
        extra_context=None,
        candidate_n=N_SAMPLES,
        temperature=TEMPERATURE,
        timeout=15,
        call_meta={
            "branch": "B_prqe",
            "issue": "branch_b_prqe_only",
            "prqe_applied": apply_b,
            "prqe_reason": prqe_b_reason,
            "drivers": ["四分支集成中的 PRQE-only 分支", "與基線分開再生成一整組候選"],
        },
    )

    # 5) Branch C: egrcm-only.
    cand_c = _generate_candidates_and_select_mbpp(
        problem,
        client,
        model,
        base,
        extra_context=egrcm_ctx,
        candidate_n=N_SAMPLES,
        temperature=TEMPERATURE,
        timeout=15,
        call_meta={
            "branch": "C_egrcm",
            "issue": "branch_c_egrcm_only",
            "prqe_applied": False,
            "prqe_reason": "分支 C 不追加 PRQE",
            "egrcm_note": egrcm_note,
            "drivers": ["四分支集成中的 EGRCM-only 分支"],
        },
    )

    # 6) Branch D: baseline (no egrcm, no prqe).
    cand_d = _generate_candidates_and_select_mbpp(
        problem,
        client,
        model,
        base,
        extra_context=None,
        candidate_n=N_SAMPLES,
        temperature=TEMPERATURE,
        timeout=15,
        call_meta={
            "branch": "D_baseline",
            "issue": "branch_d_baseline",
            "prqe_applied": False,
            "prqe_reason": "分支 D 是不帶 EGRCM、不帶 PRQE 的對照，仍固定採樣 5 次",
            "drivers": ["四分支集成中的基線分支", "固定 5 次採樣"],
        },
    )
    audit["prqe"] = {
        "branch_a_applied": apply_a,
        "branch_a_reason": prqe_a_reason,
        "branch_b_applied": apply_b,
        "branch_b_reason": prqe_b_reason,
    }
    audit["ensemble"] = {
        "branches": 4,
        "samples_per_branch": N_SAMPLES,
        "early_stop_during_generation": False,
        "reason": "A/B/C/D 四個分支各自固定生成 5 個候選後才做本地測試篩選，生成呼叫次數不會因提前通過而減少",
    }

    # 7) Final local selection with deterministic tie-breaker: A > C > B > D.
    ranked_candidates: list[tuple[int, str]] = [
        (0, cand_a),
        (1, cand_c),
        (2, cand_b),
        (3, cand_d),
    ]
    best_score: Optional[tuple[int, int, int, int]] = None
    best_code: str = cand_a

    for tie_rank, code in ranked_candidates:
        passed_all, compiles, signature_ok, _ = _candidate_eval_mbpp(problem, code, timeout=15)
        score = (
            1 if passed_all else 0,
            1 if signature_ok else 0,
            1 if compiles else 0,
            -tie_rank,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_code = code

    by_branch: dict[str, dict[str, int]] = {}
    for call in _tt_mod.TRACKER.call_log:
        if call.get("task_id") != task_id:
            continue
        branch = call.get("branch") or call.get("stage") or "unknown"
        bucket = by_branch.setdefault(
            branch,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "repairs": 0},
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += int(call.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(call.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(call.get("total_tokens") or 0)
        if call.get("stage") == "repair":
            bucket["repairs"] += 1
    audit["branches"] = by_branch
    return best_code


async def run_experiment(
    method: str,
    limit: Optional[int] = None,
    model_name: Optional[str] = None,
    delay: float = 0,
    dataset_path: Optional[Path] = None,
    output_base: Optional[Path] = None,
) -> None:
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

    # PRQE/EGRCM 只需要 llm 实例提供 async aask(prompt)。
    # 直接依赖 create_llm_instance() 会触发加载所有 provider（例如 Gemini），
    # 在部分环境缺少可选依赖时会导致 ModuleNotFoundError。
    class SimpleChatLLM:
        def __init__(self, client: OpenAI, model: str):
            self._client = client
            self._model = model

        async def aask(self, prompt: str, call_meta: Optional[dict] = None) -> str:
            def _call() -> str:
                resp = _chat_create(
                    self._client,
                    model=self._model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                if call_meta is None:
                    _tt_mod.TRACKER.add_usage(getattr(resp, "usage", None))
                else:
                    meta = dict(call_meta)
                    meta.setdefault("prompt_chars", len(prompt))
                    _tt_mod.TRACKER.add_usage(getattr(resp, "usage", None), meta=meta)
                return (resp.choices[0].message.content or "").strip()

            return await asyncio.to_thread(_call)

    llm = None
    if method in ("metagpt_egrcm", "metagpt_prqe", "metagpt_egrcm_prqe"):
        llm = SimpleChatLLM(client=client, model=model)

    problems = load_problems(dataset_path, limit)
    if os.environ.get("RECODE_TOKEN_FLAT") == "1":
        output_dir = output_base
    else:
        output_dir = output_base / method
    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = (model_name or model or "default").replace(":", "-")
    output_file = output_dir / f"samples_mbpp_{model_safe}.jsonl"

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

    _tt_mod.TRACKER = TokenTracker()
    global TRACKER
    TRACKER = _tt_mod.TRACKER

    use_prqe = method in ("metagpt_prqe", "metagpt_egrcm_prqe")
    for problem in tqdm(problems, desc=f"Experiment3-MBPP [{method}]"):
        task_id = problem.get("task_id", "unknown")
        if task_id in done_ids:
            continue
        TRACKER.start_problem(task_id)
        try:
            if method == "metagpt":
                completion = await method_metagpt(problem, client, model)
            elif method == "metagpt_egrcm":
                completion = await method_metagpt_egrcm(problem, client, model, llm)
            elif method == "metagpt_prqe":
                completion = await method_metagpt_prqe(problem, client, model, llm)
            else:
                completion = await method_metagpt_egrcm_prqe(
                    problem, client, model, llm
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
        new_calls = [c for c in TRACKER.call_log if c.get("task_id") == task_id]
        audit = TRACKER.problem_audits.get(task_id)
        row = {
            "task_id": task_id,
            "completion": completion,
            "usage": usage,
        }
        if audit:
            row["token_audit"] = audit
        if new_calls:
            row["token_calls"] = new_calls
        results.append(row)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        calls_file = output_dir / "token_calls.jsonl"
        if new_calls:
            with open(calls_file, "a", encoding="utf-8") as f:
                for call in new_calls:
                    f.write(json.dumps(call, ensure_ascii=False) + "\n")
        if os.environ.get("RECODE_TOKEN_FLAT") == "1":
            try:
                from scripts.analyze_recode_token_stages import write_analysis

                write_analysis(output_dir)
            except Exception as exc:
                print(f"階段 token 分析更新失敗: {exc}")

        if use_prqe and delay > 0:
            await asyncio.sleep(delay)

    dedup = {}
    for r in results:
        dedup[r["task_id"]] = r
    ordered = list(dedup.values())
    with open(output_file, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    token_file = output_dir / f"token_usage_mbpp_{model_safe}.json"
    summary = TRACKER.summary(
        model=model, method=method, dataset="MBPP", n_problems=len(ordered)
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
    if os.environ.get("RECODE_TOKEN_FLAT") == "1" or (output_dir / "token_calls.jsonl").exists():
        try:
            from scripts.analyze_recode_token_stages import write_analysis

            analysis = write_analysis(output_dir)
            print(
                f"[{method}] 階段分析: {output_dir / 'stage_token_analysis.json'} "
                f"(buckets={ [b['bucket'] for b in analysis['bucket_totals']] })"
            )
        except Exception as exc:
            print(f"階段 token 分析失敗: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="實驗三（MBPP）：四種方法對比測試"
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定模型（如 qwen3:32b、gpt-4o-mini、deepseek_v3.2）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="PRQE 方法時，每題完成後額外等待秒數，用於避免 API 限流（如 2.0）",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="數據集路徑（預設: results/datasets/mbpp/mbpp_public_test.jsonl）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="輸出根目錄（預設: results/full_experiments）",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if dataset_path and not dataset_path.is_absolute():
        dataset_path = _PROJECT_ROOT / dataset_path
    output_base = args.output_dir
    if output_base and not output_base.is_absolute():
        output_base = _PROJECT_ROOT / output_base

    asyncio.run(run_experiment(
        method=args.method,
        limit=args.limit,
        model_name=args.model,
        delay=args.delay,
        dataset_path=dataset_path,
        output_base=output_base,
    ))


if __name__ == "__main__":
    main()
