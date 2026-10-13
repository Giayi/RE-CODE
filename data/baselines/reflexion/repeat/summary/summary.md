# GPT-4o-mini 重複實驗（Direct LLM vs Reflexion）

- 模型：`gpt-4o-mini`
- 輪數：5
- temperature：0
- max_iters：10
- 生成時間：2026-10-12T09:56:54+08:00

## Pass@1（mean ± std）

| Dataset | Method | Rounds | Pass@1 (mean ± std) | Per-round Pass@1 |
|---|---|---:|---|---|
| humaneval | direct | 5 | 0.8767 ± 0.0035 | 0.8805, 0.8742, 0.8742, 0.8742, 0.8805 |
| humaneval | reflexion | 5 | 0.8566 ± 0.0093 | 0.8428, 0.8553, 0.8616, 0.8679, 0.8553 |
| mbpp | direct | 5 | 0.6164 ± 0.0031 | 0.6183, 0.6112, 0.6159, 0.6183, 0.6183 |
| mbpp | reflexion | 5 | 0.6150 ± 0.0097 | 0.6276, 0.6159, 0.6042, 0.6066, 0.6206 |

## Token 消耗（每輪總 tokens，mean ± std）

| Dataset | Method | Total tokens (mean ± std) | Per-round tokens |
|---|---|---|---|
| humaneval | direct | 58613.2 ± 501.7 | 58570, 58291, 57993, 58982, 59230 |
| humaneval | reflexion | 644816.0 ± 64759.7 | 571505, 593202, 653040, 672674, 733659 |
| mbpp | direct | 72383.4 ± 460.3 | 72256, 73188, 72182, 72268, 72023 |
| mbpp | reflexion | 1951828.0 ± 173381.4 | 1853182, 1772662, 2197515, 2061525, 1874256 |

## 時間消耗（每輪 wall time 秒，mean ± std）

| Dataset | Method | Time s (mean ± std) | Per-round time (s) |
|---|---|---|---|
| humaneval | direct | 336.8 ± 8.5 | 329.7, 328.8, 343.1, 334.4, 348.1 |
| humaneval | reflexion | 1847.8 ± 102.1 | 1915.1, 1705.7, 1846.4, 1801.9, 1969.8 |
| mbpp | direct | 693.6 ± 247.9 | 566.0, 594.1, 1136.6, 592.5, 579.0 |
| mbpp | reflexion | 7556.5 ± 1247.3 | 5679.5, 7212.0, 9025.8, 8196.9, 7668.5 |

## 各輪明細

### humaneval / direct

| Round | Pass@1 | Passed/Total | Prompt | Completion | Total tokens | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8805 | 140/159 | 34002 | 24568 | 58570 | 329.7 |
| 2 | 0.8742 | 139/159 | 34002 | 24289 | 58291 | 328.82 |
| 3 | 0.8742 | 139/159 | 34002 | 23991 | 57993 | 343.09 |
| 4 | 0.8742 | 139/159 | 34002 | 24980 | 58982 | 334.37 |
| 5 | 0.8805 | 140/159 | 34002 | 25228 | 59230 | 348.11 |

### humaneval / reflexion

| Round | Pass@1 | Passed/Total | Prompt | Completion | Total tokens | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8428 | 134/159 | 479828 | 91677 | 571505 | 1915.06 |
| 2 | 0.8553 | 136/159 | 499821 | 93381 | 593202 | 1705.73 |
| 3 | 0.8616 | 137/159 | 547692 | 105348 | 653040 | 1846.41 |
| 4 | 0.8679 | 138/159 | 564852 | 107822 | 672674 | 1801.94 |
| 5 | 0.8553 | 136/159 | 620033 | 113626 | 733659 | 1969.85 |

### mbpp / direct

| Round | Pass@1 | Passed/Total | Prompt | Completion | Total tokens | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6183 | 264/427 | 43169 | 29087 | 72256 | 566.0 |
| 2 | 0.6112 | 261/427 | 43169 | 30019 | 73188 | 594.08 |
| 3 | 0.6159 | 263/427 | 43169 | 29013 | 72182 | 1136.58 |
| 4 | 0.6183 | 264/427 | 43169 | 29099 | 72268 | 592.46 |
| 5 | 0.6183 | 264/427 | 43169 | 28854 | 72023 | 578.98 |

### mbpp / reflexion

| Round | Pass@1 | Passed/Total | Prompt | Completion | Total tokens | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6276 | 268/427 | 1645688 | 207494 | 1853182 | 5679.47 |
| 2 | 0.6159 | 263/427 | 1571958 | 200704 | 1772662 | 7212.0 |
| 3 | 0.6042 | 258/427 | 1955502 | 242013 | 2197515 | 9025.75 |
| 4 | 0.6066 | 259/427 | 1832065 | 229460 | 2061525 | 8196.87 |
| 5 | 0.6206 | 265/427 | 1668726 | 205530 | 1874256 | 7668.48 |

## 說明

- Pass@1 is evaluated on public tests (humaneval_public_test / mbpp_public_test).
- Tokens estimated with tiktoken o200k_base from generation logs (same as analyze_token_and_stats.py).
- Wall time includes generation + public evaluation for that round.
- Temperature follows models.yaml (0); variance across rounds may be small.

## 目錄結構

```
Output/repeat/
  summary/                 # 彙總表
  humaneval/direct/run_{1..5}/
  humaneval/reflexion/run_{1..5}/
  mbpp/direct/run_{1..5}/
  mbpp/reflexion/run_{1..5}/
```