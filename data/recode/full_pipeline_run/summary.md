# MetaGPT + EGRCM + PRQE 实验结果

模型：`gpt-4o-mini`  
方法：`metagpt_egrcm_prqe`  
输出目录：`results/output/`

## Pass@1

| Dataset | Passed | Total | Pass@1 |
|---------|-------:|------:|-------:|
| HumanEval | 146 | 159 | None |
| MBPP | 358 | 427 | 0.8384074941451991 |

## Token 消耗（live response.usage）

| Dataset | Problems | Total tokens | Avg/problem | Prompt | Completion | LLM calls | Est. cost (USD) |
|---------|----------:|-------------:|------------:|-------:|-----------:|----------:|----------------:|
| HumanEval | 159 | 124,836 | 785.13 | 82,019 | 42,817 | 250 | $0.037993 |
| MBPP | 427 | 3,104,896 | 7271.42 | 2,475,395 | 629,501 | 10575 | $0.74901 |

## 文件

- HumanEval samples: `results/output/metagpt_egrcm_prqe/samples_gpt-4o-mini.jsonl`
- HumanEval tokens: `results/output/metagpt_egrcm_prqe/token_usage_gpt-4o-mini.json`
- HumanEval eval: `results/output/evaluate/humaneval_pass1_result.json`
- MBPP samples: `results/output/metagpt_egrcm_prqe/samples_mbpp_gpt-4o-mini.jsonl`
- MBPP tokens: `results/output/metagpt_egrcm_prqe/token_usage_mbpp_gpt-4o-mini.json`
- MBPP eval: `results/output/evaluate/mbpp_pass1_result.json`
