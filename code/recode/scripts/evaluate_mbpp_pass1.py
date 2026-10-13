#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBPP Pass@1 评估脚本

评估 scripts/run_experiment3_mbpp.py 生成的 samples_mbpp_*.jsonl：
  - 每行: {"task_id": "MBPP/<idx>", "completion": "<python code>"}
并使用 results/datasets/mbpp/mbpp_public_test.jsonl 中的 (entry_point, test) 来执行断言。

用法（项目根目录）：
  python3 scripts/evaluate_mbpp_pass1.py \
    -i results/full_experiments/metagpt/samples_mbpp_gpt-4o-mini.jsonl \
    -o results/experiment3_evaluate/metagpt/pass1_mbpp_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from metagpt.utils.sanitize import sanitize


DEFAULT_DATASET = _PROJECT_ROOT / "results" / "datasets" / "mbpp" / "mbpp_public_test.jsonl"


class TimeoutError(Exception):
    pass


def run_with_timeout(func, args: tuple, timeout: int = 15):
    """在子线程中执行函数，超时则抛出 TimeoutError。"""
    result: list[Any] = []
    done = threading.Event()

    def target():
        try:
            result.append(func(*args))
        except Exception as e:  # noqa: BLE001 - 评测需要捕获所有异常
            result.append(e)
        finally:
            done.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    if not done.wait(timeout):
        raise TimeoutError("Function execution timed out")

    if not result:
        return None
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]


def _to_check_fn(test_field: Any) -> str:
    """将 mbpp_public_test.jsonl 的 test 字段转换为 `def check(candidate): ...` 字符串。"""
    if test_field is None:
        return ""
    if isinstance(test_field, str):
        return test_field.strip()
    if isinstance(test_field, list):
        return "def check(candidate):\n    " + "\n    ".join(test_field) if test_field else ""
    return ""


def load_mbpp_public(dataset_path: Path) -> Dict[str, dict]:
    """加载 mbpp_public_test.jsonl，返回 task_id -> {entry_point, test} 的映射。

    注意：public 文件通常没有 task_id，本脚本按行号生成 task_id: MBPP/<idx>。
    这与 scripts/run_experiment3_mbpp.py 的默认行为一致。
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    lookup: Dict[str, dict] = {}
    with open(dataset_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entry_point = obj.get("entry_point")
            test_str = _to_check_fn(obj.get("test"))
            if not entry_point or not test_str:
                continue
            lookup[f"MBPP/{idx}"] = {"entry_point": entry_point, "test": test_str}
    return lookup


def load_samples(samples_path: Path) -> List[dict]:
    if not samples_path.exists():
        raise FileNotFoundError(f"样本文件不存在: {samples_path}")
    out: List[dict] = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def check_solution(
    solution: str,
    test: str,
    entry_point: str,
    timeout: int = 15,
) -> Tuple[str, str]:
    """执行 MBPP `check(candidate)` 测试。"""
    if not (solution or "").strip():
        return ("FAIL", "Empty solution")
    if solution.strip().startswith("Error:"):
        return ("FAIL", solution.strip())

    try:
        solution = sanitize(code=solution, entrypoint=entry_point)
    except Exception as e:  # noqa: BLE001
        return ("FAIL", f"Sanitize error: {e}")

    try:
        g: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "re": __import__("re"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "collections": __import__("collections"),
            "statistics": __import__("statistics"),
        }
        exec(solution, g)
        if entry_point not in g:
            return ("FAIL", f"Function {entry_point} is not defined in the solution.")

        exec(test, g)
        check = g.get("check")
        if not callable(check):
            return ("FAIL", "Test does not define callable check(candidate).")

        run_with_timeout(check, (g[entry_point],), timeout=timeout)
        return ("PASS", "The solution passed all test cases.")
    except TimeoutError:
        return ("FAIL", "Execution timed out.")
    except Exception as e:  # noqa: BLE001
        return ("FAIL", str(e))


def main() -> None:
    p = argparse.ArgumentParser(description="MBPP Pass@1 evaluation (samples_mbpp_*.jsonl)")
    p.add_argument("-i", "--input", required=True, type=Path, help="samples_mbpp_*.jsonl 路径")
    p.add_argument("-o", "--output", required=True, type=Path, help="输出 JSON 路径")
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="mbpp_public_test.jsonl 路径（默认: results/datasets/mbpp/mbpp_public_test.jsonl）",
    )
    p.add_argument("--timeout", type=int, default=15, help="每题超时时间（秒）")
    args = p.parse_args()

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = _PROJECT_ROOT / dataset_path
    samples_path = args.input
    if not samples_path.is_absolute():
        samples_path = _PROJECT_ROOT / samples_path
    output_path = args.output
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path

    lookup = load_mbpp_public(dataset_path)
    samples = load_samples(samples_path)

    results: List[dict] = []
    passed = 0
    start = time.time()

    for s in samples:
        task_id = s.get("task_id")
        completion = s.get("completion", "")
        meta = lookup.get(task_id)
        if not meta:
            results.append(
                {
                    "task_id": task_id,
                    "status": "FAIL",
                    "message": f"Task {task_id} not found in dataset mapping.",
                }
            )
            continue

        ep = meta["entry_point"]
        test_str = meta["test"]
        status, msg = check_solution(completion, test_str, ep, timeout=args.timeout)
        if status == "PASS":
            passed += 1
        results.append(
            {
                "task_id": task_id,
                "entry_point": ep,
                "status": status,
                "message": msg,
            }
        )

    total = len(results)
    pass_at_1 = passed / total if total else 0.0
    summary = {
        "dataset": str(dataset_path),
        "samples": str(samples_path),
        "total": total,
        "passed": passed,
        "pass@1": pass_at_1,
        "elapsed_s": time.time() - start,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

