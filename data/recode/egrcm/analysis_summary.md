# MetaGPT + EGRCM：需求熵 × Pass@1
模型：`gpt-4o-mini`
## HumanEval
- Pass@1: **89.9371%** (143/159)
- EGRCM skipped / run: 146 / 13
- Pass@1 (skipped): 0.9178082191780822
- Pass@1 (EGRCM run): 0.6923076923076923
- avg entropy_after: 0.303516
- avg entropy_delta: 0.053717

| bin | entropy range | n | passed | Pass@1 |
|-----|---------------|--:|-------:|-------:|
| 0 | [0.300, 0.309) | 146 | 134 | 0.9178 |
| 4 | [0.334, 0.343) | 13 | 9 | 0.6923 |

逐題記錄：`results/egrcm/humaneval/records_gpt-4o-mini.jsonl`
## MBPP
- Pass@1: **76.3466%** (326/427)
- EGRCM skipped / run: 3 / 424
- Pass@1 (skipped): 0.6666666666666666
- Pass@1 (EGRCM run): 0.7641509433962265
- avg entropy_after: 0.342698
- avg entropy_delta: 0.652384

| bin | entropy range | n | passed | Pass@1 |
|-----|---------------|--:|-------:|-------:|
| 0 | [0.300, 0.309) | 3 | 2 | 0.6667 |
| 4 | [0.334, 0.343) | 424 | 324 | 0.7642 |

逐題記錄：`results/egrcm/mbpp/records_gpt-4o-mini.jsonl`

## 欄位說明
- `entropy_before` / `entropy_after`：EGRCM 狀態熵（初始 1.0，每輪澄清 ×0.7）
- `egrcm_skipped`：gating 判定低熵而跳過
- `passed` / `pass_at_1`：該題是否通過（0/1）
