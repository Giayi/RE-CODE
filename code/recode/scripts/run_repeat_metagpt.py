#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Repeat baseline MetaGPT on HumanEval and MBPP, then summarize pass@1.

Five independent rounds (default). Each round records Pass@1, token usage, and
wall-clock time. Interrupted rounds resume from the existing samples jsonl.
Completions that start with "Error:" are dropped and regenerated.

Output (default): results/repeat_metagpt/
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
METHOD = "metagpt"
MODEL = "gpt-4o-mini"

DATASETS = {
    "humaneval": {
        "label": "HumanEval",
        "n": 159,
        "script": "scripts/run_experiment3_humaneval.py",
        "eval_script": "scripts/evaluate_humaneval_pass1.py",
        "dataset": "results/datasets/humaneval/humaneval_public_test.jsonl",
        "samples_name": "samples_{model}.jsonl",
        "token_name": "token_usage_{model}.json",
    },
    "mbpp": {
        "label": "MBPP",
        "n": 427,
        "script": "scripts/run_experiment3_mbpp.py",
        "eval_script": "scripts/evaluate_mbpp_pass1.py",
        "dataset": "results/datasets/mbpp/mbpp_public_test.jsonl",
        "samples_name": "samples_mbpp_{model}.jsonl",
        "token_name": "token_usage_mbpp_{model}.json",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def model_safe(model: str) -> str:
    return model.replace(":", "-")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_samples(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def is_error_completion(obj: dict) -> bool:
    return str(obj.get("completion") or "").startswith("Error:")


def count_samples(path: Path) -> tuple[int, int]:
    """Return (unique non-error task_ids, error completions)."""
    ids: set[str] = set()
    errors = 0
    for obj in iter_samples(path) or ():
        if is_error_completion(obj):
            errors += 1
            continue
        tid = obj.get("task_id")
        if tid:
            ids.add(str(tid))
    return len(ids), errors


def recompute_token_file(token_path: Path, drop_ids: set[str]) -> None:
    data = read_json(token_path)
    if not data:
        return
    problems = [p for p in data.get("problems", []) if str(p.get("task_id")) not in drop_ids]
    prompt = sum(int(p.get("prompt_tokens") or 0) for p in problems)
    completion = sum(int(p.get("completion_tokens") or 0) for p in problems)
    total = sum(int(p.get("total_tokens") or 0) for p in problems)
    calls = sum(int(p.get("llm_calls") or 0) for p in problems)
    n = len(problems)
    data["problems"] = problems
    data["n_problems"] = n
    data["prompt_tokens"] = prompt
    data["completion_tokens"] = completion
    data["total_tokens"] = total
    data["llm_calls"] = calls
    data["avg_tokens_per_problem"] = round(total / n, 2) if n else 0.0
    data["avg_llm_calls_per_problem"] = round(calls / n, 3) if n else 0.0
    data["est_cost_usd"] = round(prompt * 0.15 / 1e6 + completion * 0.60 / 1e6, 6)
    write_json(token_path, data)


def strip_error_samples(paths: dict[str, Path]) -> int:
    """Drop API-failure rows so the generator will retry those task ids."""
    samples = paths["samples"]
    kept: list[str] = []
    dropped: set[str] = set()
    for obj in iter_samples(samples) or ():
        if is_error_completion(obj):
            tid = obj.get("task_id")
            if tid:
                dropped.add(str(tid))
            continue
        kept.append(json.dumps(obj, ensure_ascii=False))
    if not dropped:
        return 0
    text = ("\n".join(kept) + "\n") if kept else ""
    samples.write_text(text, encoding="utf-8")
    recompute_token_file(paths["tokens"], dropped)
    return len(dropped)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小時{m}分{s}秒"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


def fmt_int(n: int | float | None) -> str:
    if n is None:
        return "—"
    return f"{int(n):,}"


def fmt_pct(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p:.4f}%"


def fmt_mean_std(mean: float | None, std: float | None, suffix: str = "", digits: int = 4) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.{digits}f}{suffix}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}{suffix}"


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = float(statistics.mean(values))
    if len(values) < 2:
        return mean, None
    return mean, float(statistics.stdev(values))


def log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {message}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def run_logged(cmd: list[str], log_path: Path, cwd: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {now_iso()} =====\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return proc.returncode, time.perf_counter() - start


def round_paths(root: Path, dataset: str, round_idx: int, model: str) -> dict[str, Path]:
    safe = model_safe(model)
    spec = DATASETS[dataset]
    rd = root / dataset / f"round_{round_idx:02d}"
    method_dir = rd / METHOD
    return {
        "round_dir": rd,
        "method_dir": method_dir,
        "samples": method_dir / spec["samples_name"].format(model=safe),
        "tokens": method_dir / spec["token_name"].format(model=safe),
        "eval": rd / "eval_pass1.json",
        "meta": rd / "meta.json",
        "gen_log": rd / "generation.log",
        "eval_log": rd / "evaluation.log",
    }


def parse_eval(dataset: str, data: dict) -> tuple[int, int, float]:
    if dataset == "humaneval":
        passed = int(data["passed"])
        total = int(data["total"])
    else:
        summary = data["summary"]
        passed = int(summary["passed"])
        total = int(summary["total"])
    rate = (passed / total) if total else 0.0
    return passed, total, rate


def token_brief(token_path: Path) -> dict | None:
    data = read_json(token_path)
    if not data:
        return None
    keys = [
        "n_problems",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_calls",
        "avg_tokens_per_problem",
        "avg_llm_calls_per_problem",
        "est_cost_usd",
    ]
    brief = {k: data.get(k) for k in keys}
    brief["pricing_usd_per_million"] = data.get("pricing_usd_per_million")
    return brief


def generation_complete(paths: dict[str, Path], n_expected: int) -> bool:
    n, errors = count_samples(paths["samples"])
    return errors == 0 and n >= n_expected


def ensure_generation(
    *,
    root: Path,
    dataset: str,
    round_idx: int,
    model: str,
    delay: float,
    py: str,
    runner_log: Path,
    max_retries: int,
) -> dict:
    spec = DATASETS[dataset]
    paths = round_paths(root, dataset, round_idx, model)
    meta = read_json(paths["meta"]) or {
        "round": round_idx,
        "dataset": dataset,
        "dataset_label": spec["label"],
        "method": METHOD,
        "model": model,
        "temperature": 0,
        "n_expected": spec["n"],
        "status": "pending",
        "generation": {"elapsed_s": 0.0, "attempts": []},
        "evaluation": None,
    }
    meta["generation"] = meta.get("generation") or {"elapsed_s": 0.0, "attempts": []}
    meta["generation"].setdefault("elapsed_s", 0.0)
    meta["generation"].setdefault("attempts", [])

    dropped = strip_error_samples(paths)
    if dropped:
        log_line(runner_log, f"{spec['label']} round {round_idx}: dropped {dropped} error completions for retry")

    if generation_complete(paths, spec["n"]):
        n, errors = count_samples(paths["samples"])
        meta["n_samples"] = n
        meta["n_error_completions"] = errors
        meta["tokens"] = token_brief(paths["tokens"])
        write_json(paths["meta"], meta)
        log_line(runner_log, f"{spec['label']} round {round_idx}: generation already complete ({n}/{spec['n']})")
        return meta

    cmd = [
        py,
        "-u",
        spec["script"],
        "--method",
        METHOD,
        "--model",
        model,
        "--delay",
        str(delay),
        "--dataset",
        spec["dataset"],
        "-o",
        str(paths["round_dir"].relative_to(_PROJECT_ROOT)),
    ]
    for attempt in range(1, max_retries + 1):
        strip_error_samples(paths)
        if generation_complete(paths, spec["n"]):
            break
        log_line(
            runner_log,
            f"{spec['label']} round {round_idx}: generation attempt {attempt}/{max_retries}",
        )
        meta["status"] = "generating"
        write_json(paths["meta"], meta)
        started = now_iso()
        rc, elapsed = run_logged(cmd, paths["gen_log"], _PROJECT_ROOT)
        finished = now_iso()
        dropped = strip_error_samples(paths)
        meta["generation"]["attempts"].append(
            {
                "attempt": attempt,
                "started_at": started,
                "finished_at": finished,
                "elapsed_s": round(elapsed, 3),
                "returncode": rc,
                "error_completions_dropped": dropped,
            }
        )
        meta["generation"]["elapsed_s"] = round(
            float(meta["generation"]["elapsed_s"]) + elapsed, 3
        )
        n, errors = count_samples(paths["samples"])
        meta["n_samples"] = n
        meta["n_error_completions"] = errors
        meta["tokens"] = token_brief(paths["tokens"])
        write_json(paths["meta"], meta)
        if rc == 0 and n >= spec["n"] and errors == 0:
            break
        log_line(
            runner_log,
            f"{spec['label']} round {round_idx}: generation exit={rc}, samples={n}/{spec['n']}, errors_dropped={dropped}",
        )
        if attempt < max_retries:
            time.sleep(min(120, 15 * attempt))

    n, errors = count_samples(paths["samples"])
    meta["n_samples"] = n
    meta["n_error_completions"] = errors
    meta["tokens"] = token_brief(paths["tokens"])
    if n < spec["n"] or errors:
        meta["status"] = "generation_incomplete"
    write_json(paths["meta"], meta)
    return meta


def ensure_evaluation(
    *,
    root: Path,
    dataset: str,
    round_idx: int,
    model: str,
    py: str,
    runner_log: Path,
    meta: dict,
) -> dict:
    spec = DATASETS[dataset]
    paths = round_paths(root, dataset, round_idx, model)
    if meta.get("status") == "completed" and paths["eval"].exists() and meta.get("pass_at_1") is not None:
        log_line(runner_log, f"{spec['label']} round {round_idx}: evaluation already complete")
        return meta
    if not generation_complete(paths, spec["n"]):
        meta["status"] = "generation_incomplete"
        write_json(paths["meta"], meta)
        return meta

    cmd = [
        py,
        "-u",
        spec["eval_script"],
        "-i",
        str(paths["samples"]),
        "-o",
        str(paths["eval"]),
        "--dataset",
        spec["dataset"],
        "--timeout",
        "15",
    ]
    log_line(runner_log, f"{spec['label']} round {round_idx}: evaluating pass@1")
    meta["status"] = "evaluating"
    write_json(paths["meta"], meta)
    started = now_iso()
    rc, elapsed = run_logged(cmd, paths["eval_log"], _PROJECT_ROOT)
    finished = now_iso()
    meta["evaluation"] = {
        "started_at": started,
        "finished_at": finished,
        "elapsed_s": round(elapsed, 3),
        "returncode": rc,
    }
    if rc != 0 or not paths["eval"].exists():
        meta["status"] = "evaluation_failed"
        write_json(paths["meta"], meta)
        log_line(runner_log, f"{spec['label']} round {round_idx}: evaluation failed rc={rc}")
        return meta

    passed, total, rate = parse_eval(dataset, read_json(paths["eval"]) or {})
    gen_s = float((meta.get("generation") or {}).get("elapsed_s") or 0.0)
    meta.update(
        {
            "status": "completed",
            "passed": passed,
            "total": total,
            "pass_at_1": rate,
            "pass_at_1_percent": round(rate * 100, 6),
            "total_elapsed_s": round(gen_s + elapsed, 3),
            "files": {
                "samples": str(paths["samples"].relative_to(_PROJECT_ROOT)),
                "token_usage": str(paths["tokens"].relative_to(_PROJECT_ROOT)),
                "eval": str(paths["eval"].relative_to(_PROJECT_ROOT)),
                "generation_log": str(paths["gen_log"].relative_to(_PROJECT_ROOT)),
                "evaluation_log": str(paths["eval_log"].relative_to(_PROJECT_ROOT)),
            },
        }
    )
    write_json(paths["meta"], meta)
    log_line(
        runner_log,
        f"{spec['label']} round {round_idx}: pass@1={meta['pass_at_1_percent']:.4f}% "
        f"({passed}/{total}), tokens={((meta.get('tokens') or {}).get('total_tokens'))}, "
        f"gen_time={fmt_duration(gen_s)}",
    )
    return meta


def collect_rounds(root: Path, dataset: str, rounds: int, model: str) -> list[dict]:
    rows = []
    for i in range(1, rounds + 1):
        meta = read_json(round_paths(root, dataset, i, model)["meta"])
        if meta:
            rows.append(meta)
    return rows


def summarize_dataset(rows: list[dict], rounds: int) -> dict:
    done = [r for r in rows if r.get("status") == "completed"]
    pass_pct = [float(r["pass_at_1_percent"]) for r in done]
    tokens = [float((r.get("tokens") or {}).get("total_tokens") or 0) for r in done]
    prompt = [float((r.get("tokens") or {}).get("prompt_tokens") or 0) for r in done]
    completion = [float((r.get("tokens") or {}).get("completion_tokens") or 0) for r in done]
    calls = [float((r.get("tokens") or {}).get("llm_calls") or 0) for r in done]
    cost = [float((r.get("tokens") or {}).get("est_cost_usd") or 0) for r in done]
    gen_s = [float((r.get("generation") or {}).get("elapsed_s") or 0) for r in done]
    eval_s = [float((r.get("evaluation") or {}).get("elapsed_s") or 0) for r in done]
    p_mean, p_std = mean_std(pass_pct)
    t_mean, t_std = mean_std(tokens)
    g_mean, g_std = mean_std(gen_s)
    return {
        "rounds_requested": rounds,
        "rounds_completed": len(done),
        "pass_at_1_percent_mean": None if p_mean is None else round(p_mean, 6),
        "pass_at_1_percent_std": None if p_std is None else round(p_std, 6),
        "pass_at_1_percent_values": pass_pct,
        "std_kind": "sample (ddof=1)" if len(done) >= 2 else None,
        "total_tokens_mean": t_mean,
        "total_tokens_std": t_std,
        "prompt_tokens_sum": int(sum(prompt)) if done else 0,
        "completion_tokens_sum": int(sum(completion)) if done else 0,
        "total_tokens_sum": int(sum(tokens)) if done else 0,
        "llm_calls_sum": int(sum(calls)) if done else 0,
        "est_cost_usd_sum": round(sum(cost), 6) if done else 0.0,
        "generation_elapsed_s_mean": g_mean,
        "generation_elapsed_s_std": g_std,
        "generation_elapsed_s_sum": round(sum(gen_s), 3) if done else 0.0,
        "evaluation_elapsed_s_sum": round(sum(eval_s), 3) if done else 0.0,
        "rounds": rows,
    }


def render_summary_md(root: Path, payload: dict) -> str:
    lines = [
        "# MetaGPT 基線重複實驗",
        "",
        f"- 方法：`{METHOD}`（直接生成程式碼，不含 EGRCM / PRQE）",
        f"- 模型：`{payload['model']}`",
        "- 解碼：temperature = 0（與既有 MetaGPT 基線腳本一致）",
        f"- 輪數：{payload['rounds']}",
        "- 資料集：HumanEval `results/datasets/humaneval/humaneval_public_test.jsonl`（159 題）、MBPP `results/datasets/mbpp/mbpp_public_test.jsonl`（427 題）",
        "- Pass@1 標準差：樣本標準差（ddof=1）",
        "- Token：生成過程中 `response.usage` 的實測值",
        "- 時間：每一輪生成的牆鐘時間（評估時間另列，不含在生成時間內）",
        "- 費用估計：輸入 $0.15 / 1M tokens，輸出 $0.60 / 1M tokens",
        "",
        "## Pass@1（mean ± std）",
        "",
        "| 資料集 | 完成輪數 | Pass@1 mean ± std |",
        "|--------|--------:|------------------:|",
    ]
    for key in ("humaneval", "mbpp"):
        block = payload["datasets"][key]
        label = DATASETS[key]["label"]
        lines.append(
            f"| {label} | {block['rounds_completed']}/{block['rounds_requested']} | "
            f"{fmt_mean_std(block['pass_at_1_percent_mean'], block['pass_at_1_percent_std'], '%')} |"
        )

    for key in ("humaneval", "mbpp"):
        block = payload["datasets"][key]
        label = DATASETS[key]["label"]
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- Pass@1：{fmt_mean_std(block['pass_at_1_percent_mean'], block['pass_at_1_percent_std'], '%')}",
                f"- 生成時間：{fmt_mean_std(block['generation_elapsed_s_mean'], block['generation_elapsed_s_std'], ' 秒', 1)}",
                f"- Token 合計（五輪加總，完成輪）：{fmt_int(block['total_tokens_sum'])}（估計費用 ${block['est_cost_usd_sum']:.4f}）",
                "",
                "| 輪次 | 狀態 | Passed | Total | Pass@1 | Prompt tokens | Completion tokens | Total tokens | LLM calls | 估計費用 (USD) | 生成時間 | 評估時間 |",
                "|-----:|------|-------:|------:|-------:|--------------:|------------------:|-------------:|----------:|---------------:|---------:|---------:|",
            ]
        )
        by_round = {int(r["round"]): r for r in block["rounds"]}
        for i in range(1, payload["rounds"] + 1):
            r = by_round.get(i)
            if not r:
                lines.append(f"| {i} | 未開始 | — | — | — | — | — | — | — | — | — | — |")
                continue
            tok = r.get("tokens") or {}
            gen = (r.get("generation") or {}).get("elapsed_s")
            ev = (r.get("evaluation") or {}).get("elapsed_s") if r.get("evaluation") else None
            lines.append(
                "| {round} | {status} | {passed} | {total} | {pct} | {pt} | {ct} | {tt} | {calls} | {cost} | {gen} | {ev} |".format(
                    round=i,
                    status=r.get("status") or "—",
                    passed=r.get("passed", "—"),
                    total=r.get("total", "—"),
                    pct=fmt_pct(r.get("pass_at_1_percent")),
                    pt=fmt_int(tok.get("prompt_tokens")),
                    ct=fmt_int(tok.get("completion_tokens")),
                    tt=fmt_int(tok.get("total_tokens")),
                    calls=fmt_int(tok.get("llm_calls")),
                    cost=f"{tok['est_cost_usd']:.6f}" if tok.get("est_cost_usd") is not None else "—",
                    gen=fmt_duration(gen),
                    ev=fmt_duration(ev),
                )
            )
        lines.extend(["", "逐輪原始檔：", ""])
        for i in range(1, payload["rounds"] + 1):
            rel = f"{key}/round_{i:02d}"
            lines.append(
                f"- 第 {i} 輪：`{rel}/meta.json`、`{rel}/eval_pass1.json`、`{rel}/{METHOD}/`"
            )

    lines.extend(
        [
            "",
            "## 目錄",
            "",
            "```",
            "results/repeat_metagpt/",
            "  summary.md          總表（Pass@1 mean±std、逐輪 token 與時間）",
            "  summary.json",
            "  humaneval/",
            "    summary.json",
            "    round_01 … round_05/",
            "      meta.json       該輪 Pass@1、token、時間",
            "      eval_pass1.json 逐題對錯",
            "      metagpt/        samples 與 token_usage",
            "  mbpp/",
            "    summary.json",
            "    round_01 … round_05/",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_summaries(root: Path, rounds: int, model: str, delay: float) -> dict:
    payload = {
        "method": METHOD,
        "model": model,
        "temperature": 0,
        "rounds": rounds,
        "delay_s": delay,
        "std_kind": "sample (ddof=1)",
        "updated_at": now_iso(),
        "datasets": {},
    }
    for key in ("humaneval", "mbpp"):
        rows = collect_rounds(root, key, rounds, model)
        block = summarize_dataset(rows, rounds)
        payload["datasets"][key] = block
        write_json(root / key / "summary.json", block)
    write_json(root / "summary.json", payload)
    (root / "summary.md").write_text(render_summary_md(root, payload), encoding="utf-8")
    return payload


def acquire_lock(root: Path) -> None:
    pid_path = root / "runner.pid"
    if pid_path.exists():
        try:
            old = int(pid_path.read_text().strip())
        except ValueError:
            old = None
        if old and Path(f"/proc/{old}").exists():
            raise SystemExit(f"runner already active pid={old}")
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat baseline MetaGPT and summarize pass@1")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "results" / "repeat_metagpt")
    parser.add_argument("--datasets", default="humaneval,mbpp")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    root = args.output if args.output.is_absolute() else _PROJECT_ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    acquire_lock(root)
    runner_log = root / "runner.log"
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset: {d}")

    log_line(
        runner_log,
        f"START method={METHOD} model={args.model} rounds={args.rounds} datasets={datasets} delay={args.delay}",
    )
    try:
        for dataset in datasets:
            for i in range(1, args.rounds + 1):
                meta = ensure_generation(
                    root=root,
                    dataset=dataset,
                    round_idx=i,
                    model=args.model,
                    delay=args.delay,
                    py=args.python,
                    runner_log=runner_log,
                    max_retries=args.max_retries,
                )
                ensure_evaluation(
                    root=root,
                    dataset=dataset,
                    round_idx=i,
                    model=args.model,
                    py=args.python,
                    runner_log=runner_log,
                    meta=meta,
                )
                write_summaries(root, args.rounds, args.model, args.delay)
        log_line(runner_log, "ALL DONE")
    finally:
        pid_path = root / "runner.pid"
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


if __name__ == "__main__":
    main()
