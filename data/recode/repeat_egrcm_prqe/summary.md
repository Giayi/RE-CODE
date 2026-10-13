# MetaGPT + EGRCM + PRQE 重複實驗

- 方法：`metagpt_egrcm_prqe`
- 模型：`gpt-4o-mini`
- 輪數：5
- 資料集：HumanEval `humaneval_public_test.jsonl`（159 題）、MBPP `mbpp_public_test.jsonl`（427 題）
- Pass@1 標準差：樣本標準差（ddof=1）
- Token：生成過程中 `response.usage` 的實測值
- 時間：每一輪生成的牆鐘時間（評估時間另列，不含在生成時間內）
- 費用估計：輸入 $0.15 / 1M tokens，輸出 $0.60 / 1M tokens

## Pass@1（mean ± std）

| 資料集 | 完成輪數 | Pass@1 mean ± std |
|--------|--------:|------------------:|
| HumanEval | 5/5 | 92.2013 ± 1.0524% |
| MBPP | 5/5 | 84.3091 ± 0.5492% |

## HumanEval

- Pass@1：92.2013 ± 1.0524%
- 生成時間：749.7 ± 12.2 秒
- Token 合計：601,057（估計費用 $0.1795）

| 輪次 | 狀態 | Passed | Total | Pass@1 | Prompt tokens | Completion tokens | Total tokens | LLM calls | 估計費用 (USD) | 生成時間 | 評估時間 |
|-----:|------|-------:|------:|-------:|--------------:|------------------:|-------------:|----------:|---------------:|---------:|---------:|
| 1 | completed | 148 | 159 | 93.0818% | 78,296 | 36,853 | 115,149 | 243 | 0.033856 | 12分17秒 | 1秒 |
| 2 | completed | 144 | 159 | 90.5660% | 81,807 | 43,047 | 124,854 | 249 | 0.038099 | 12分43秒 | 1秒 |
| 3 | completed | 147 | 159 | 92.4528% | 81,333 | 38,226 | 119,559 | 246 | 0.035136 | 12分42秒 | 1秒 |
| 4 | completed | 146 | 159 | 91.8239% | 81,245 | 41,349 | 122,594 | 246 | 0.036996 | 12分22秒 | 1秒 |
| 5 | completed | 148 | 159 | 93.0818% | 79,750 | 39,151 | 118,901 | 244 | 0.035453 | 12分24秒 | 1秒 |

逐輪原始檔：

- 第 1 輪：`humaneval/round_01/meta.json`、`humaneval/round_01/eval_pass1.json`、`humaneval/round_01/metagpt_egrcm_prqe/`
- 第 2 輪：`humaneval/round_02/meta.json`、`humaneval/round_02/eval_pass1.json`、`humaneval/round_02/metagpt_egrcm_prqe/`
- 第 3 輪：`humaneval/round_03/meta.json`、`humaneval/round_03/eval_pass1.json`、`humaneval/round_03/metagpt_egrcm_prqe/`
- 第 4 輪：`humaneval/round_04/meta.json`、`humaneval/round_04/eval_pass1.json`、`humaneval/round_04/metagpt_egrcm_prqe/`
- 第 5 輪：`humaneval/round_05/meta.json`、`humaneval/round_05/eval_pass1.json`、`humaneval/round_05/metagpt_egrcm_prqe/`

## MBPP

- Pass@1：84.3091 ± 0.5492%
- 生成時間：14257.4 ± 954.4 秒
- Token 合計：15,462,410（估計費用 $3.7048）

| 輪次 | 狀態 | Passed | Total | Pass@1 | Prompt tokens | Completion tokens | Total tokens | LLM calls | 估計費用 (USD) | 生成時間 | 評估時間 |
|-----:|------|-------:|------:|-------:|--------------:|------------------:|-------------:|----------:|---------------:|---------:|---------:|
| 1 | completed | 356 | 427 | 83.3724% | 2,479,330 | 621,070 | 3,100,400 | 10,583 | 0.744541 | 4小時0分55秒 | 1秒 |
| 2 | completed | 360 | 427 | 84.3091% | 2,480,590 | 652,246 | 3,132,836 | 10,588 | 0.763436 | 4小時15分55秒 | 1秒 |
| 3 | completed | 361 | 427 | 84.5433% | 2,473,698 | 617,471 | 3,091,169 | 10,569 | 0.741537 | 4小時7分4秒 | 1秒 |
| 4 | completed | 361 | 427 | 84.5433% | 2,475,657 | 586,149 | 3,061,806 | 10,574 | 0.723038 | 3小時35分10秒 | 4秒 |
| 5 | completed | 362 | 427 | 84.7775% | 2,474,442 | 601,757 | 3,076,199 | 10,570 | 0.732220 | 3小時49分3秒 | 1秒 |

逐輪原始檔：

- 第 1 輪：`mbpp/round_01/meta.json`、`mbpp/round_01/eval_pass1.json`、`mbpp/round_01/metagpt_egrcm_prqe/`
- 第 2 輪：`mbpp/round_02/meta.json`、`mbpp/round_02/eval_pass1.json`、`mbpp/round_02/metagpt_egrcm_prqe/`
- 第 3 輪：`mbpp/round_03/meta.json`、`mbpp/round_03/eval_pass1.json`、`mbpp/round_03/metagpt_egrcm_prqe/`
- 第 4 輪：`mbpp/round_04/meta.json`、`mbpp/round_04/eval_pass1.json`、`mbpp/round_04/metagpt_egrcm_prqe/`
- 第 5 輪：`mbpp/round_05/meta.json`、`mbpp/round_05/eval_pass1.json`、`mbpp/round_05/metagpt_egrcm_prqe/`

## 目錄

```
results/repeat experiment/
  summary.md
  summary.json
  humaneval/round_01 … round_05/
  mbpp/round_01 … round_05/
```
