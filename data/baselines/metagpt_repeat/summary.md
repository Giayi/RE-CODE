# MetaGPT 基線重複實驗

- 方法：`metagpt`（直接生成程式碼，不含 EGRCM / PRQE）
- 模型：`gpt-4o-mini`
- 解碼：temperature = 0（與既有 MetaGPT 基線腳本一致）
- 輪數：5
- 資料集：HumanEval `results/datasets/humaneval/humaneval_public_test.jsonl`（159 題）、MBPP `results/datasets/mbpp/mbpp_public_test.jsonl`（427 題）
- Pass@1 標準差：樣本標準差（ddof=1）
- Token：生成過程中 `response.usage` 的實測值
- 時間：每一輪生成的牆鐘時間（評估時間另列，不含在生成時間內）
- 費用估計：輸入 $0.15 / 1M tokens，輸出 $0.60 / 1M tokens

## Pass@1（mean ± std）

| 資料集 | 完成輪數 | Pass@1 mean ± std |
|--------|--------:|------------------:|
| HumanEval | 5/5 | 87.1698 ± 0.7171% |
| MBPP | 5/5 | 75.7377 ± 0.6541% |

## HumanEval

- Pass@1：87.1698 ± 0.7171%
- 生成時間：271.8 ± 7.3 秒
- Token 合計（五輪加總，完成輪）：329,157（估計費用 $0.0882）

| 輪次 | 狀態 | Passed | Total | Pass@1 | Prompt tokens | Completion tokens | Total tokens | LLM calls | 估計費用 (USD) | 生成時間 | 評估時間 |
|-----:|------|-------:|------:|-------:|--------------:|------------------:|-------------:|----------:|---------------:|---------:|---------:|
| 1 | completed | 139 | 159 | 87.4214% | 48,559 | 16,937 | 65,496 | 159 | 0.017446 | 4分42秒 | 1秒 |
| 2 | completed | 140 | 159 | 88.0503% | 48,559 | 17,497 | 66,056 | 159 | 0.017782 | 4分26秒 | 1秒 |
| 3 | completed | 139 | 159 | 87.4214% | 48,559 | 17,257 | 65,816 | 159 | 0.017638 | 4分33秒 | 1秒 |
| 4 | completed | 138 | 159 | 86.7925% | 48,559 | 17,165 | 65,724 | 159 | 0.017583 | 4分23秒 | 1秒 |
| 5 | completed | 137 | 159 | 86.1635% | 48,559 | 17,506 | 66,065 | 159 | 0.017787 | 4分34秒 | 1秒 |

逐輪原始檔：

- 第 1 輪：`humaneval/round_01/meta.json`、`humaneval/round_01/eval_pass1.json`、`humaneval/round_01/metagpt/`
- 第 2 輪：`humaneval/round_02/meta.json`、`humaneval/round_02/eval_pass1.json`、`humaneval/round_02/metagpt/`
- 第 3 輪：`humaneval/round_03/meta.json`、`humaneval/round_03/eval_pass1.json`、`humaneval/round_03/metagpt/`
- 第 4 輪：`humaneval/round_04/meta.json`、`humaneval/round_04/eval_pass1.json`、`humaneval/round_04/metagpt/`
- 第 5 輪：`humaneval/round_05/meta.json`、`humaneval/round_05/eval_pass1.json`、`humaneval/round_05/metagpt/`

## MBPP

- Pass@1：75.7377 ± 0.6541%
- 生成時間：610.6 ± 77.5 秒
- Token 合計（五輪加總，完成輪）：655,455（估計費用 $0.1636）

| 輪次 | 狀態 | Passed | Total | Pass@1 | Prompt tokens | Completion tokens | Total tokens | LLM calls | 估計費用 (USD) | 生成時間 | 評估時間 |
|-----:|------|-------:|------:|-------:|--------------:|------------------:|-------------:|----------:|---------------:|---------:|---------:|
| 1 | completed | 320 | 427 | 74.9415% | 102,090 | 38,867 | 140,957 | 427 | 0.038634 | 10分58秒 | 1秒 |
| 2 | completed | 321 | 427 | 75.1756% | 102,090 | 22,482 | 124,572 | 427 | 0.028803 | 8分40秒 | 1秒 |
| 3 | completed | 326 | 427 | 76.3466% | 102,090 | 22,458 | 124,548 | 427 | 0.028788 | 9分10秒 | 1秒 |
| 4 | completed | 324 | 427 | 75.8782% | 102,090 | 22,501 | 124,591 | 427 | 0.028814 | 11分50秒 | 1秒 |
| 5 | completed | 326 | 427 | 76.3466% | 102,090 | 38,697 | 140,787 | 427 | 0.038532 | 10分14秒 | 1秒 |

逐輪原始檔：

- 第 1 輪：`mbpp/round_01/meta.json`、`mbpp/round_01/eval_pass1.json`、`mbpp/round_01/metagpt/`
- 第 2 輪：`mbpp/round_02/meta.json`、`mbpp/round_02/eval_pass1.json`、`mbpp/round_02/metagpt/`
- 第 3 輪：`mbpp/round_03/meta.json`、`mbpp/round_03/eval_pass1.json`、`mbpp/round_03/metagpt/`
- 第 4 輪：`mbpp/round_04/meta.json`、`mbpp/round_04/eval_pass1.json`、`mbpp/round_04/metagpt/`
- 第 5 輪：`mbpp/round_05/meta.json`、`mbpp/round_05/eval_pass1.json`、`mbpp/round_05/metagpt/`

## 目錄

```
results/repeat_metagpt/
  summary.md          總表（Pass@1 mean±std、逐輪 token 與時間）
  summary.json
  humaneval/
    summary.json
    round_01 … round_05/
      meta.json       該輪 Pass@1、token、時間
      eval_pass1.json 逐題對錯
      metagpt/        samples 與 token_usage
  mbpp/
    summary.json
    round_01 … round_05/
```
