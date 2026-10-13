#!/bin/bash
# Run MetaGPT+EGRCM+PRQE on HumanEval & MBPP with gpt-4o-mini, then evaluate.
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate metagpt_env
cd /workspace

OUT=results/output
METHOD=metagpt_egrcm_prqe
MODEL=gpt-4o-mini
LOGDIR=$OUT/logs
mkdir -p "$OUT/$METHOD" "$LOGDIR" "$OUT/evaluate"

echo "[$(date)] START HumanEval"
python scripts/run_experiment3_humaneval.py \
  --method "$METHOD" \
  --model "$MODEL" \
  --delay 2 \
  -o "$OUT" \
  2>&1 | tee "$LOGDIR/humaneval_gen.log"

echo "[$(date)] EVAL HumanEval"
python scripts/evaluate_humaneval_pass1.py \
  -i "$OUT/$METHOD/samples_${MODEL}.jsonl" \
  -o "$OUT/evaluate/humaneval_pass1_result.json" \
  2>&1 | tee "$LOGDIR/humaneval_eval.log"

echo "[$(date)] START MBPP"
python scripts/run_experiment3_mbpp.py \
  --method "$METHOD" \
  --model "$MODEL" \
  --delay 2 \
  -o "$OUT" \
  2>&1 | tee "$LOGDIR/mbpp_gen.log"

echo "[$(date)] EVAL MBPP"
python scripts/evaluate_mbpp_pass1.py \
  -i "$OUT/$METHOD/samples_mbpp_${MODEL}.jsonl" \
  -o "$OUT/evaluate/mbpp_pass1_result.json" \
  2>&1 | tee "$LOGDIR/mbpp_eval.log"

echo "[$(date)] WRITE SUMMARY"
python - <<'PY'
import json
from pathlib import Path
out = Path("results/output")
he_tok = json.loads((out / "metagpt_egrcm_prqe/token_usage_gpt-4o-mini.json").read_text())
mb_tok = json.loads((out / "metagpt_egrcm_prqe/token_usage_mbpp_gpt-4o-mini.json").read_text())
he_ev = json.loads((out / "evaluate/humaneval_pass1_result.json").read_text())
mb_ev = json.loads((out / "evaluate/mbpp_pass1_result.json").read_text())

def pass_rate(ev):
    # tolerate different schema keys
    if "pass_at_1" in ev:
        return ev["pass_at_1"], ev.get("passed"), ev.get("total")
    if "pass@1" in ev:
        return ev["pass@1"], ev.get("passed"), ev.get("total")
    if "summary" in ev:
        s = ev["summary"]
        return s.get("pass_at_1") or s.get("pass@1"), s.get("passed"), s.get("total")
    # fallback scan
    passed = ev.get("num_passed") or ev.get("passed")
    total = ev.get("num_total") or ev.get("total")
    rate = ev.get("pass_rate")
    return rate, passed, total

he_rate, he_p, he_t = pass_rate(he_ev)
mb_rate, mb_p, mb_t = pass_rate(mb_ev)

md = f"""# MetaGPT + EGRCM + PRQE 实验结果

模型：`gpt-4o-mini`  
方法：`metagpt_egrcm_prqe`  
输出目录：`results/output/`

## Pass@1

| Dataset | Passed | Total | Pass@1 |
|---------|-------:|------:|-------:|
| HumanEval | {he_p} | {he_t} | {he_rate} |
| MBPP | {mb_p} | {mb_t} | {mb_rate} |

## Token 消耗（live response.usage）

| Dataset | Problems | Total tokens | Avg/problem | Prompt | Completion | LLM calls | Est. cost (USD) |
|---------|----------:|-------------:|------------:|-------:|-----------:|----------:|----------------:|
| HumanEval | {he_tok['n_problems']} | {he_tok['total_tokens']:,} | {he_tok['avg_tokens_per_problem']} | {he_tok['prompt_tokens']:,} | {he_tok['completion_tokens']:,} | {he_tok['llm_calls']} | ${he_tok['est_cost_usd']} |
| MBPP | {mb_tok['n_problems']} | {mb_tok['total_tokens']:,} | {mb_tok['avg_tokens_per_problem']} | {mb_tok['prompt_tokens']:,} | {mb_tok['completion_tokens']:,} | {mb_tok['llm_calls']} | ${mb_tok['est_cost_usd']} |

## 文件

- HumanEval samples: `results/output/metagpt_egrcm_prqe/samples_gpt-4o-mini.jsonl`
- HumanEval tokens: `results/output/metagpt_egrcm_prqe/token_usage_gpt-4o-mini.json`
- HumanEval eval: `results/output/evaluate/humaneval_pass1_result.json`
- MBPP samples: `results/output/metagpt_egrcm_prqe/samples_mbpp_gpt-4o-mini.jsonl`
- MBPP tokens: `results/output/metagpt_egrcm_prqe/token_usage_mbpp_gpt-4o-mini.json`
- MBPP eval: `results/output/evaluate/mbpp_pass1_result.json`
"""
(out / "summary.md").write_text(md, encoding="utf-8")
print(md)
PY

echo "[$(date)] ALL DONE"
