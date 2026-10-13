#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HumanEval 实验三 Pass@1 评估脚本

对 run_experiment3_humaneval 生成的 samples_*.jsonl 进行评估，计算 Pass@1。
参考 metagpt/ext/aflow/benchmark/humaneval.py 的 check_solution 逻辑。

用法（项目根目录执行，需在 metagpt 环境中）：
  python scripts/evaluate_humaneval_pass1.py \
    -i results/full_experiments/metagpt/samples_gpt-4o-mini.jsonl \
    -o results/experiment3_evaluate/metagpt/pass1_result.json

  # 批量评估多种方法
  for m in metagpt metagpt_egrcm metagpt_prqe metagpt_egrcm_prqe; do
    python scripts/evaluate_humaneval_pass1.py \
      -i results/full_experiments/$m/samples_gpt-4o-mini.jsonl \
      -o results/experiment3_evaluate/$m/pass1_result.json
  done
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 保证从项目根目录运行
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple

from metagpt.utils.sanitize import sanitize


class TimeoutError(Exception):
    pass


def run_with_timeout(func, args: tuple, timeout: int = 15):
    """在子线程中执行函数，超时则抛出 TimeoutError。"""
    result = []
    stop_event = threading.Event()

    def target():
        try:
            result.append(func(*args))
        except Exception as e:
            result.append(e)
        finally:
            stop_event.set()

    thread = threading.Thread(target=target)
    thread.start()
    is_timeout = not stop_event.wait(timeout)

    if is_timeout:
        raise TimeoutError("Function execution timed out")

    if not result:
        return None
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]


def check_solution(
    solution: str,
    test: str,
    entry_point: str,
    timeout: int = 15,
) -> Tuple[str, str]:
    """
    检查生成的代码是否通过测试。
    参考 HumanEvalBenchmark.check_solution。

    Returns:
        (status, message) - status 为 "PASS" 或 "FAIL"
    """
    if not (solution or "").strip():
        return ("FAIL", "Empty solution")
    if solution.strip().startswith("Error:"):
        return ("FAIL", solution.strip())

    try:
        solution = sanitize(code=solution, entrypoint=entry_point)
    except Exception as e:
        return ("FAIL", f"Sanitize error: {e}")

    try:
        global_dict = {
            "math": __import__("math"),
            "hashlib": __import__("hashlib"),
            "re": __import__("re"),
            "List": List,
            "Dict": Dict,
            "Tuple": Tuple,
            "Optional": Optional,
            "Any": Any,
        }

        # HumanEval 特殊 case（与 humaneval.py 一致）
        if entry_point == "decode_cyclic":
            solution = (
                '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    # split string to groups. Each of length 3.\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    # cycle elements in each group. Unless group has fewer elements than 3.\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)'
                + "\n\n"
                + solution
            )
        elif entry_point == "decode_shift":
            solution = (
                '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                + solution
            )
        elif entry_point == "find_zero":
            solution = (
                "\n\ndef poly(xs: list, x: float):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n"
                + solution
            )

        exec(solution, global_dict)

        if entry_point not in global_dict:
            return ("FAIL", f"Function {entry_point} is not defined in the solution.")

        exec(test, global_dict)
        check = global_dict["check"]

        result = run_with_timeout(check, (global_dict[entry_point],), timeout)

        if result is None:
            result = ("PASS", "The solution passed all test cases.")
        return result

    except TimeoutError:
        return (
            "FAIL",
            "Execution timed out. Check for infinite loops or time-consuming operations.",
        )
    except Exception as e:
        return ("FAIL", str(e))


def load_dataset(dataset_path: Path) -> Dict[str, dict]:
    """加载 HumanEval 数据集，返回 task_id -> {test, entry_point} 的映射。"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    problems = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))

    # public 格式（无 prompt）需从 human-eval.jsonl 合并 test
    if problems and "test" not in problems[0]:
        raise ValueError("数据集缺少 test 字段")

    lookup = {}
    for p in problems:
        pid = p.get("problem_id") or p.get("task_id")
        if not pid:
            continue

        t = p.get("test")
        if isinstance(t, list):
            test_str = "def check(candidate):\n    " + "\n    ".join(t) if t else ""
        else:
            test_str = (t or "").strip()

        ep = p.get("entry_point")
        if not ep or not test_str:
            continue

        lookup[pid] = {"test": test_str, "entry_point": ep}

    return lookup


def load_samples(samples_path: Path) -> List[dict]:
    """加载生成的 samples jsonl。"""
    if not samples_path.exists():
        raise FileNotFoundError(f"样本文件不存在: {samples_path}")

    samples = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def evaluate(
    samples_path: Path,
    dataset_path: Path,
    output_path: Optional[Path] = None,
    timeout: int = 15,
) -> dict:
    """
    执行 Pass@1 评估。

    Returns:
        {
            "pass1": float,           # Pass@1 百分比
            "total": int,
            "passed": int,
            "results": [{"task_id": ..., "status": "PASS"|"FAIL", "message": ...}, ...]
        }
    """
    dataset = load_dataset(dataset_path)
    samples = load_samples(samples_path)

    results = []
    for s in samples:
        task_id = s.get("task_id", "unknown")
        completion = s.get("completion", "")

        if task_id not in dataset:
            results.append({
                "task_id": task_id,
                "status": "FAIL",
                "message": f"task_id {task_id} not found in dataset",
            })
            continue

        test = dataset[task_id]["test"]
        entry_point = dataset[task_id]["entry_point"]

        status, message = check_solution(completion, test, entry_point, timeout=timeout)

        results.append({
            "task_id": task_id,
            "status": status,
            "message": message[:200] if len(str(message)) > 200 else message,
        })

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    pass1 = (passed / total * 100) if total > 0 else 0.0

    out = {
        "pass1": round(pass1, 2),
        "total": total,
        "passed": passed,
        "samples_file": str(samples_path),
        "dataset_file": str(dataset_path),
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {output_path}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description="HumanEval 实验三 Pass@1 评估"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="samples 文件路径，如 results/full_experiments/metagpt/samples_gpt-4o-mini.jsonl",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="评估结果 JSON 输出路径",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_PROJECT_ROOT / "results" / "datasets" / "humaneval" / "humaneval_public_test.jsonl",
        help="HumanEval 数据集路径（需含 test、entry_point；public 格式会与 human-eval.jsonl 合并）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="单题执行超时秒数",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = _PROJECT_ROOT / dataset_path

    # public 格式需合并 human-eval.jsonl 获取 test（若 public 本身有 test 则直接用）
    # load_dataset 已支持 public 的 test 为 list 的情况
    # 但 humaneval_public_test 只有 problem_id, test(list), entry_point，无需合并
    if dataset_path.name == "humaneval_public_test.jsonl":
        # 直接加载，public 已有 test 和 entry_point
        pass
    elif dataset_path.name == "human-eval.jsonl":
        # human-eval 有 task_id, test(string), entry_point
        pass

    out = evaluate(
        samples_path=args.input,
        dataset_path=dataset_path,
        output_path=args.output,
        timeout=args.timeout,
    )

    print(f"\nPass@1: {out['pass1']:.2f}% ({out['passed']}/{out['total']})")


if __name__ == "__main__":
    main()
