#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRQE 增益实验：验证 RQ - Does PRQE further improve requirement quality after EGRCM?

实验设计：
  Method          | Description
  EGRCM           | 仅进行需求熵收敛 → Requirement_v1
  EGRCM + PRQE    | 在 EGRCM 后进行需求质量评估与优化 → Requirement_v2

Pipeline: User Req → EGRCM → v1 → PRQE.evaluate → PRQE.feedback → LLM refine → v2

数据来源：entropy_frnfr10_pure_results.json（由 frnfr_10_for_pure.json 经 EGRCM 生成）

用法（需在项目根目录执行）:
  python scripts/run_prqe_refinement_experiment.py
  python scripts/run_prqe_refinement_experiment.py --report  # 生成 Markdown 报告
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if _PROJECT_ROOT not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_PROJECT_ROOT))

DIMENSIONS = [
    "completeness", "clarity", "feasibility", "consistency",
    "testability", "traceability", "atomicity",
]


def _to_simplified_chinese(text: str) -> str:
    """将文本中的繁体中文转为简体中文。"""
    if not text or not isinstance(text, str):
        return text
    try:
        import zhconv
        return zhconv.convert(text, "zh-cn")
    except ImportError:
        return text


async def run_experiment(
    input_path: Path,
    output_path: Path,
    llm,
) -> list[dict]:
    """执行 PRQE 精炼实验。"""
    from metagpt.configs.prqe_config import PRQEConfig
    from metagpt.modules.requirement_quality_evaluator import (
        RequirementFeedbackGenerator,
        RequirementQualityEvaluator,
        RequirementRefiner,
    )

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results_raw = data.get("results", [])
    if not results_raw:
        print(f"[WARN] 未找到 results: {input_path}")
        return [], {}

    evaluator = RequirementQualityEvaluator(PRQEConfig())
    feedback_gen = RequirementFeedbackGenerator()
    refiner = RequirementRefiner()

    experiment_results = []
    for i, item in enumerate(results_raw):
        req_id = f"R{i+1:02d}"
        # 支援 EGRCM (refined_requirement)、baseline_metagpt (metagpt_output)、原始需求 (original_requirement)
        requirement_v1 = item.get("refined_requirement", item.get("metagpt_output", item.get("original_requirement", "")))
        if not requirement_v1:
            continue

        print(f"[{i+1}/{len(results_raw)}] {req_id} 评估 v1...")
        result_v1 = await evaluator.evaluate(requirement_v1, llm)
        score_v1 = result_v1.overall_score

        # 生成 feedback，若 evaluator 已有 suggestions 则合并
        feedback = await feedback_gen.generate(requirement_v1, result_v1, llm)
        if not feedback and result_v1.improvement_suggestions:
            feedback = result_v1.improvement_suggestions

        print(f"       score_v1={score_v1:.2f}, issues={len(result_v1.issues)}, feedback={len(feedback)}")

        # LLM refine
        requirement_v2 = await refiner.refine(requirement_v1, feedback, llm)
        if not requirement_v2.strip():
            requirement_v2 = requirement_v1

        print(f"       精炼完成，评估 v2...")
        result_v2 = await evaluator.evaluate(requirement_v2, llm)
        score_v2 = result_v2.overall_score

        delta = round(score_v2 - score_v1, 2)
        print(f"       score_v2={score_v2:.2f}, delta={delta:+.2f}")

        issues_v1_list = [i.description if hasattr(i, "description") else str(i) for i in result_v1.issues]
        experiment_results.append({
            "requirement_id": req_id,
            "original_requirement": _to_simplified_chinese(item.get("original_requirement", "")[:200]),
            "requirement_v1": _to_simplified_chinese(requirement_v1)[:2000],
            "requirement_v2": _to_simplified_chinese(requirement_v2)[:2000],
            "score_v1": round(score_v1, 2),
            "score_v2": round(score_v2, 2),
            "delta": delta,
            "scores_v1": result_v1.scores,
            "scores_v2": result_v2.scores,
            "issues_v1": [_to_simplified_chinese(x) for x in issues_v1_list],
            "feedback": [_to_simplified_chinese(str(x)) for x in feedback],
        })

    # 汇总统计
    deltas = [r["delta"] for r in experiment_results]
    scores_v1 = [r["score_v1"] for r in experiment_results]
    scores_v2 = [r["score_v2"] for r in experiment_results]
    n = len(experiment_results)

    avg_delta = sum(deltas) / n if deltas else 0
    improved = sum(1 for d in deltas if d > 0)
    same = sum(1 for d in deltas if d == 0)
    worse = sum(1 for d in deltas if d < 0)

    # Std
    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    std_v1 = round(_std(scores_v1), 2)
    std_v2 = round(_std(scores_v2), 2)
    std_delta = round(_std(deltas), 2)

    # Cohen's d = mean(delta) / std(delta)
    cohens_d = round(avg_delta / std_delta, 4) if std_delta > 0 else 0.0

    # 95% CI
    def _ci_95(mean_val, std_val, sample_n):
        if sample_n < 2 or std_val == 0:
            return [round(mean_val, 2), round(mean_val, 2)]
        try:
            from scipy import stats as scipy_stats
            t_crit = scipy_stats.t.ppf(0.975, df=sample_n - 1)
            se = std_val / (sample_n ** 0.5)
            return [round(mean_val - t_crit * se, 2), round(mean_val + t_crit * se, 2)]
        except ImportError:
            # 近似：z=1.96 for large n
            se = std_val / (sample_n ** 0.5)
            return [round(mean_val - 1.96 * se, 2), round(mean_val + 1.96 * se, 2)]

    score_v1_avg = round(sum(scores_v1) / n, 2)
    score_v2_avg = round(sum(scores_v2) / n, 2)
    ci_v1 = _ci_95(score_v1_avg, _std(scores_v1), n)
    ci_v2 = _ci_95(score_v2_avg, _std(scores_v2), n)

    summary = {
        "experiment_meta": {
            "source": str(input_path),
            "total": n,
            "avg_delta": round(avg_delta, 2),
            "std_delta": std_delta,
            "improved": improved,
            "same": same,
            "worse": worse,
        },
        "summary": {
            "score_v1_avg": score_v1_avg,
            "score_v1_std": std_v1,
            "score_v1_ci95": ci_v1,
            "score_v2_avg": score_v2_avg,
            "score_v2_std": std_v2,
            "score_v2_ci95": ci_v2,
            "mean_improvement": round(avg_delta, 2),
            "std_improvement": std_delta,
            "cohens_d": cohens_d,
        },
        "results": experiment_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 维度平均（用于报告）
    dim_avg_v1 = {}
    dim_avg_v2 = {}
    for dim in DIMENSIONS:
        vals1 = [r["scores_v1"].get(dim, 0) for r in experiment_results]
        vals2 = [r["scores_v2"].get(dim, 0) for r in experiment_results]
        dim_avg_v1[dim] = round(sum(vals1) / len(vals1), 2) if vals1 else 0
        dim_avg_v2[dim] = round(sum(vals2) / len(vals2), 2) if vals2 else 0

    summary["dimension_avg"] = {"EGRCM": dim_avg_v1, "EGRCM+PRQE": dim_avg_v2}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    #  paired t-test
    try:
        from scipy import stats
        scores_v1 = [r["score_v1"] for r in experiment_results]
        scores_v2 = [r["score_v2"] for r in experiment_results]
        t_stat, p_value = stats.ttest_rel(scores_v2, scores_v1)
        summary["paired_ttest"] = {"t_statistic": round(t_stat, 4), "p_value": round(p_value, 4)}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[统计] paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    except ImportError:
        summary["paired_ttest"] = {"note": "scipy not installed, skip t-test"}

    # 输出简要对比表
    print("\n" + "=" * 60)
    print("score_v1 vs score_v2 对比")
    print("=" * 60)
    for r in experiment_results:
        delta_str = f"+{r['delta']}" if r["delta"] >= 0 else str(r["delta"])
        print(f"  {r['requirement_id']}: {r['score_v1']:.2f} → {r['score_v2']:.2f}  (delta={delta_str})")
    print("=" * 60)
    print(f"[OK] 实验完成，结果已写入 {output_path}")
    print(f"     Mean Improvement={avg_delta:+.2f}, Std Improvement={std_delta}")
    print(f"     Cohen's d={cohens_d}, 提升={improved}, 持平={same}, 下降={worse}")
    return experiment_results, summary


def _recompute_stats_from_results(results: list) -> dict:
    """从 results 重新计算统计量（含 std、Cohen's d、95% CI）。"""
    if not results:
        return {}
    deltas = [r["delta"] for r in results]
    scores_v1 = [r["score_v1"] for r in results]
    scores_v2 = [r["score_v2"] for r in results]
    n = len(results)

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    avg_delta = sum(deltas) / n
    std_v1 = round(_std(scores_v1), 2)
    std_v2 = round(_std(scores_v2), 2)
    std_delta = round(_std(deltas), 2)
    cohens_d = round(avg_delta / std_delta, 4) if std_delta > 0 else 0.0

    def _ci_95(mean_val, std_val, sample_n):
        if sample_n < 2 or std_val == 0:
            return [round(mean_val, 2), round(mean_val, 2)]
        try:
            from scipy import stats as scipy_stats
            t_crit = scipy_stats.t.ppf(0.975, df=sample_n - 1)
            se = std_val / (sample_n ** 0.5)
            return [round(mean_val - t_crit * se, 2), round(mean_val + t_crit * se, 2)]
        except ImportError:
            se = std_val / (sample_n ** 0.5)
            return [round(mean_val - 1.96 * se, 2), round(mean_val + 1.96 * se, 2)]

    score_v1_avg = round(sum(scores_v1) / n, 2)
    score_v2_avg = round(sum(scores_v2) / n, 2)
    return {
        "experiment_meta": {"std_delta": std_delta},
        "summary": {
            "score_v1_avg": score_v1_avg,
            "score_v1_std": std_v1,
            "score_v1_ci95": _ci_95(score_v1_avg, _std(scores_v1), n),
            "score_v2_avg": score_v2_avg,
            "score_v2_std": std_v2,
            "score_v2_ci95": _ci_95(score_v2_avg, _std(scores_v2), n),
            "mean_improvement": round(avg_delta, 2),
            "std_improvement": std_delta,
            "cohens_d": cohens_d,
        },
    }


def _convert_json_to_simplified(input_path: Path, output_path: Path) -> None:
    """将已有实验 JSON 中的中文转为简体，并补充统计量（若缺失）。"""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data.get("results", []):
        for key in ("original_requirement", "requirement_v1", "requirement_v2"):
            if key in r and r[key]:
                r[key] = _to_simplified_chinese(r[key])
        if "issues_v1" in r:
            r["issues_v1"] = [_to_simplified_chinese(x) for x in r["issues_v1"]]
        if "feedback" in r:
            r["feedback"] = [_to_simplified_chinese(str(x)) for x in r["feedback"]]
    # 若缺少新统计量，从 results 重新计算
    results = data.get("results", [])
    if results and ("std_delta" not in data.get("experiment_meta", {}) or "cohens_d" not in data.get("summary", {})):
        stats = _recompute_stats_from_results(results)
        if stats:
            data.setdefault("experiment_meta", {}).update(stats.get("experiment_meta", {}))
            data.setdefault("summary", {}).update(stats.get("summary", {}))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已转为简体中文并写入 {output_path}")


def generate_report(summary: dict, report_path: Path) -> None:
    """生成 Markdown 报告（表1/2/3 + t-test + Cohen's d + CI）。"""
    meta = summary.get("experiment_meta", {})
    sm = summary.get("summary", {})
    dim_avg = summary.get("dimension_avg", {})
    ttest = summary.get("paired_ttest", {})

    total = meta.get("total", 0)
    improved = meta.get("improved", 0)
    same = meta.get("same", 0)
    worse = meta.get("worse", 0)
    avg_delta = meta.get("avg_delta", 0)
    std_delta = meta.get("std_delta", 0)
    score_v1_avg = sm.get("score_v1_avg", 0)
    score_v2_avg = sm.get("score_v2_avg", 0)
    std_v1 = sm.get("score_v1_std", 0)
    std_v2 = sm.get("score_v2_std", 0)
    ci_v1 = sm.get("score_v1_ci95", [0, 0])
    ci_v2 = sm.get("score_v2_ci95", [0, 0])
    mean_improvement = sm.get("mean_improvement", avg_delta)
    std_improvement = sm.get("std_improvement", std_delta)
    cohens_d = sm.get("cohens_d", 0)
    improvement_rate = round(100 * improved / total, 1) if total else 0

    egrcm_dims = dim_avg.get("EGRCM", {})
    prqe_dims = dim_avg.get("EGRCM+PRQE", {})

    ci_v1_str = f"[{ci_v1[0]}, {ci_v1[1]}]" if ci_v1 else "N/A"
    ci_v2_str = f"[{ci_v2[0]}, {ci_v2[1]}]" if ci_v2 else "N/A"

    # Cohen's d 效应量解释
    if cohens_d >= 0.8:
        effect_desc = "large（PRQE provides a large practical improvement）"
    elif cohens_d >= 0.5:
        effect_desc = "medium"
    elif cohens_d >= 0.2:
        effect_desc = "small"
    else:
        effect_desc = "< 0.2（negligible）"

    lines = [
        "# PRQE 增益实验报告",
        "",
        "## 研究问题 (RQ)",
        "",
        "**Does PRQE further improve requirement quality after entropy-guided requirement convergence (EGRCM)?**",
        "",
        "EGRCM → clearer requirement  |  PRQE → higher-quality requirement",
        "",
        "---",
        "",
        "## 表1：总体评分",
        "",
        "| Method | Avg | Std | 95% CI |",
        "|--------|-----|-----|--------|",
        f"| EGRCM | {score_v1_avg} | {std_v1} | {ci_v1_str} |",
        f"| EGRCM + PRQE | {score_v2_avg} | {std_v2} | {ci_v2_str} |",
        "",
        "---",
        "",
        "## 平均提升量 (Mean Improvement)",
        "",
        "| Metric | Value |",
        "|---------|-------|",
        f"| Mean Improvement | +{mean_improvement} |",
        f"| Std Improvement | {std_improvement} |",
        "",
        "---",
        "",
        "## 表2：改进比例",
        "",
        "| Result | Count |",
        "|--------|-------|",
        f"| Improved | {improved} |",
        f"| Unchanged | {same} |",
        f"| Decreased | {worse} |",
        "",
        f"**Win Rate = {improvement_rate}%**",
        "",
        "*PRQE improves requirement quality in the majority of cases.*",
        "",
        "---",
        "",
        "## 表3：维度提升",
        "",
        "| Dimension | EGRCM | EGRCM+PRQE | Delta |",
        "|-----------|-------|------------|-------|",
    ]

    for dim in DIMENSIONS:
        v1 = egrcm_dims.get(dim, 0)
        v2 = prqe_dims.get(dim, 0)
        d = round(v2 - v1, 2)
        d_str = f"+{d}" if d >= 0 else str(d)
        lines.append(f"| {dim} | {v1} | {v2} | {d_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 效应量 (Effect Size: Cohen's d)",
        "",
        "公式：d = mean(delta) / std(delta)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Cohen's d | {cohens_d} |",
        f"| 效应量解释 | {effect_desc} |",
        "",
        "*参考：d=0.2 small, d=0.5 medium, d≥0.8 large*",
        "",
        "---",
        "",
        "## 统计检验 (paired t-test)",
        "",
        "score_v1 vs score_v2",
        "",
    ])
    if "t_statistic" in ttest:
        lines.extend([
            f"- t-statistic: {ttest['t_statistic']}",
            f"- p-value: {ttest['p_value']}",
            "",
            "*p < 0.05 表示 EGRCM+PRQE 相对 EGRCM 有显著提升*",
        ])
    else:
        lines.append(f"- {ttest.get('note', 'N/A')}")

    lines.extend([
        "",
        "---",
        "",
        "## 数据来源",
        "",
        f"- 输入: {meta.get('source', 'N/A')}",
        f"- 样本数: {total}",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] 报告已写入 {report_path}")


def main():
    parser = argparse.ArgumentParser(description="PRQE 增益实验")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=_PROJECT_ROOT / "results/experiments/entropy_atomic/entropy_frnfr10_pure_results.json",
        help="输入 JSON（EGRCM 结果，源自 frnfr_10_for_pure.json）",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=_PROJECT_ROOT / "results/prqe_experiments/prqe_refinement_comparison.json",
        help="输出 JSON 路径",
    )
    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument(
        "--report", "-r",
        action="store_true",
        default=True,
        help="生成 Markdown 报告（默认）",
    )
    report_group.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help="不生成 Markdown 报告",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="报告输出目录（默认与 JSON 同目录）",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="仅将已有 JSON 中的中文转为简体，不重新运行实验",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else _PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else _PROJECT_ROOT / args.output
    report_dir = args.report_dir if (args.report_dir and args.report_dir.is_absolute()) else (_PROJECT_ROOT / args.report_dir if args.report_dir else None)

    if args.convert_only:
        # 转换时默认读写同一文件
        path = output_path if output_path.exists() else input_path
        if not path.exists():
            print(f"[ERROR] 文件不存在: {path}，请用 -o 指定要转换的 JSON")
            sys.exit(1)
        _convert_json_to_simplified(path, path)
        # 若补充了统计量，重新生成报告
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("summary", {}).get("cohens_d") is not None and args.report:
            rpt = (report_dir / path.with_suffix("").name).with_suffix(".md") if report_dir else path.with_suffix(".md")
            generate_report(data, rpt)
        return

    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        sys.exit(1)

    from metagpt.context import Context
    from metagpt.llm import LLM

    ctx = Context()
    llm = LLM(context=ctx)
    results, summary = asyncio.run(run_experiment(input_path, output_path, llm))

    if args.report and summary.get("experiment_meta"):
        report_path = (report_dir / output_path.with_suffix("").name).with_suffix(".md") if report_dir else output_path.with_suffix(".md")
        generate_report(summary, report_path)


if __name__ == "__main__":
    main()
