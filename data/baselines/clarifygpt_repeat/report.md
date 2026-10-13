# ClarifyGPT 重複實驗報告

- 狀態：complete
- 更新時間：2026-10-11T18:42:39+08:00
- 模型：`gpt-4o-mini`
- 框架：ClarifyGPT（Pass@1，k=1，n_samples=1）
- 解碼：greedy temperature=0.0，sample temperature=0.0
- 推論設定：`three_shot`，max_tokens=800
- 重複輪數：5（各輪獨立呼叫模型，不沿用上一輪生成結果）
- Pass@1：該輪通過評測的題數 / 題數 × 100
- 標準差：樣本標準差（ddof=1）。輪數少於 2 時不計算。
- 題目耗時：該輪每一題的 API 與評測耗時加總，中斷後續跑仍可累計。
- 牆鐘時間：該輪實驗進程的實際經過時間；若中途續跑，會累加各段進程時間。

## 總覽

| 資料集 | 已完成輪數 | Pass@1 mean ± std | 總 token mean ± std | 題目耗時 mean ± std（秒） |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 5/5 | 86.0377 ± 2.1513 | 236431.60 ± 3098.83 | 515.714 ± 13.514 |
| MBPP public | 5/5 | 76.3934 ± 0.8180 | 693470.00 ± 7037.55 | 1565.689 ± 57.809 |

## 資料集說明

- HumanEval：題幹與隱藏評測來自 `humaneval/human-eval.jsonl`，一致性檢查使用 `humaneval/humaneval_public_test.jsonl`。只保留兩邊都存在的題目。
- MBPP public：題目範圍為 `mbpp/mbpp_public_test.jsonl` 的 427 題。題幹與完整評測來自對齊的 `mbpp/mbpp-new.jsonl`，一致性檢查使用公開測試中的 assert。

## HumanEval

| 輪次 | 通過 / 題數 | Pass@1 (%) | 澄清題數 | prompt tokens | completion tokens | total tokens | API 次數 | 題目耗時（秒） | 牆鐘時間（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 138/159 | 86.7925 | 33 | 200933 | 37291 | 238224 | 258 | 528.703 | 534.647 |
| 2 | 138/159 | 86.7925 | 32 | 195753 | 37227 | 232980 | 255 | 496.267 | 505.509 |
| 3 | 131/159 | 82.3899 | 33 | 201780 | 37705 | 239485 | 258 | 520.874 | 527.087 |
| 4 | 140/159 | 88.0503 | 33 | 201162 | 37121 | 238283 | 258 | 507.527 | 510.683 |
| 5 | 137/159 | 86.1635 | 32 | 196045 | 37141 | 233186 | 255 | 525.200 | 528.171 |

- Pass@1 mean ± std：`86.0377 ± 2.1513`
- 總 token mean ± std：`236431.60 ± 3098.83`
- prompt token mean ± std：`199134.60 ± 2971.68`
- completion token mean ± std：`37297.00 ± 238.07`
- API 次數 mean ± std：`256.80 ± 1.64`
- 題目耗時 mean ± std（秒）：`515.714 ± 13.514`
- 牆鐘時間 mean ± std（秒）：`521.219 ± 12.459`
- 五輪判定不一致的題數：16

- 判定不一致題目：HumanEval/40、HumanEval/76、HumanEval/77、HumanEval/82、HumanEval/91、HumanEval/93、HumanEval/108、HumanEval/116、HumanEval/126、HumanEval/127、HumanEval/133、HumanEval/137、HumanEval/150、HumanEval/159、HumanEval/160、HumanEval/163

## MBPP public

| 輪次 | 通過 / 題數 | Pass@1 (%) | 澄清題數 | prompt tokens | completion tokens | total tokens | API 次數 | 題目耗時（秒） | 牆鐘時間（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 324/427 | 75.8782 | 152 | 585727 | 100119 | 685846 | 883 | 1519.872 | 1537.992 |
| 2 | 331/427 | 77.5176 | 154 | 593325 | 100939 | 694264 | 889 | 1545.311 | 1564.467 |
| 3 | 322/427 | 75.4098 | 156 | 600144 | 101375 | 701519 | 895 | 1547.591 | 1563.725 |
| 4 | 328/427 | 76.8150 | 152 | 586112 | 100673 | 686785 | 883 | 1548.789 | 1563.705 |
| 5 | 326/427 | 76.3466 | 155 | 597352 | 101584 | 698936 | 892 | 1666.882 | 1681.927 |

- Pass@1 mean ± std：`76.3934 ± 0.8180`
- 總 token mean ± std：`693470.00 ± 7037.55`
- prompt token mean ± std：`592532.00 ± 6506.31`
- completion token mean ± std：`100938.00 ± 580.79`
- API 次數 mean ± std：`888.40 ± 5.37`
- 題目耗時 mean ± std（秒）：`1565.689 ± 57.809`
- 牆鐘時間 mean ± std（秒）：`1582.363 ± 56.784`
- 五輪判定不一致的題數：46

- 判定不一致題目：MBPP/631、MBPP/57、MBPP/776、MBPP/106、MBPP/228、MBPP/780、MBPP/758#1、MBPP/306、MBPP/607、MBPP/16、MBPP/610、MBPP/622、MBPP/390、MBPP/117、MBPP/640、MBPP/249、MBPP/446、MBPP/84、MBPP/393、MBPP/239、MBPP/749、MBPP/580、MBPP/251、MBPP/641、MBPP/417、MBPP/442、MBPP/265、MBPP/448、MBPP/424、MBPP/475、MBPP/415#1、MBPP/160、MBPP/260、MBPP/63、MBPP/290、MBPP/750、MBPP/120、MBPP/783、MBPP/751、MBPP/765 …（其餘 6 題見 summary/aggregate.json）

## 檔案分類

- `humaneval/run_01` … `humaneval/run_05`：HumanEval 各輪完整生成結果、摘要與 `run_meta.json`
- `mbpp_public/run_01` … `mbpp_public/run_05`：MBPP public 各輪完整生成結果、摘要與 `run_meta.json`
- `summary/aggregate.json`：Pass@1、token、時間的 mean ± std
- `summary/per_run.json` 與 `summary/per_run.csv`：每一輪的指標
- `experiment_config.json`：本次實驗設定（不含 API key）
- `progress.json`：目前輪次
