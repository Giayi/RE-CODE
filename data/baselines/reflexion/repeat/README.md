# Output/repeat — GPT-4o-mini 五輪重複實驗

## 結構

```
Output/repeat/
  README.md
  summary/
    summary.json
    summary.md
    all_runs.json
  humaneval/
    direct/run_{1..5}/
      gen/...jsonl              # 生成結果
      public_eval.jsonl         # public test 逐題評測
      public_eval_summary.json  # Pass@1 摘要
      metrics.json              # 本輪 pass@1 / tokens / wall time
      generation.log
      evaluation.log
    reflexion/run_{1..5}/
  mbpp/
    direct/run_{1..5}/
    reflexion/run_{1..5}/
```

## 設定

- 模型：`gpt-4o-mini`
- 方法：Direct LLM（`simple`）與 Reflexion（`max_iters=10`）
- 評測：`humaneval_public_test.jsonl`、`mbpp_public_test.jsonl`
- 輪數：5
- 指標：每輪 Pass@1、token（tiktoken 估算）、wall-clock 時間；最終報告 mean ± std

## 重新彙總

```bash
cd programming_runs
python run_repeat_experiments.py --aggregate-only
```
