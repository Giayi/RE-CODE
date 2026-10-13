# Reflexion vs Direct LLM：Token 消耗與統計顯著性（gpt-4o-mini）

模型：`gpt-4o-mini`；資料集：HumanEval public test、MBPP public test。

## 1. Token 消耗

| Dataset | Method | Problems | Total tokens | Mean tokens/problem | Prompt | Completion | Mean LLM calls | Approx. cost (USD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HumanEval | direct | 159 | 56,695 | 356.57 | 32,818 | 23,877 | 1.0 | $0.0192 |
| HumanEval | reflexion | 159 | 535,512 | 3,368.0 | 449,502 | 86,010 | 3.99 | $0.119 |
| MBPP | direct | 424 | 72,161 | 170.19 | 43,169 | 28,992 | 1.0 | $0.0239 |
| MBPP | reflexion | 424 | 1,836,455 | 4,331.26 | 1,632,394 | 204,061 | 5.73 | $0.3673 |

### Token 差異統計（Reflexion − Direct，per-problem）

| Dataset | Mean Δ tokens | ↓CI | ↑CI | p-value | Cohen's d | Ratio (R/D) |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval | 3011.43 | 1930.39 | 4238.82 | 1e-06 | 0.4028 | 9.45x |
| MBPP | 4161.07 | 3471.23 | 4865.13 | 0.0 | 0.5667 | 25.45x |

## 2. Pass@1 與統計顯著性

### 各方法 Pass@1（Wilson 95% CI）

| Dataset | Method | Passed | Total | Pass@1 | ↓CI | ↑CI |
|---|---|---:|---:|---:|---:|---:|
| HumanEval | direct | 139 | 159 | 0.8742 | 0.8137 | 0.9171 |
| HumanEval | reflexion | 135 | 159 | 0.8491 | 0.7852 | 0.8964 |
| MBPP | direct | 263 | 427 | 0.6159 | 0.569 | 0.6608 |
| MBPP | reflexion | 262 | 427 | 0.6136 | 0.5666 | 0.6586 |

### 方法對比（Reflexion − Direct）

| Dataset | Δ Pass@1 | ↓CI | ↑CI | p-value (McNemar) | Cohen's d |
|---|---:|---:|---:|---:|---:|
| HumanEval | -0.0252 | -0.0692 | 0.0189 | 0.387695 | -0.0917 |
| MBPP | -0.0023 | -0.0258 | 0.0211 | 1.0 | -0.0093 |

## 方法說明

- Token：以既有 `gpt-4o-mini` 生成日誌還原對應 Chat prompt，用 `tiktoken`（`o200k_base`）計數；費用按 gpt-4o-mini $0.15/1M input、$0.6/1M output 估算。
- Pass@1 CI：Wilson score interval（95%）。
- Δ Pass@1 CI：paired bootstrap（10,000 resamples）對 per-problem 成敗差值均值。
- p-value：McNemar exact binomial test（雙側）。
- Cohen's d：paired effect size，`(mean(diff) / std(diff))`，其中 diff 為 per-problem（Reflexion − Direct）。
