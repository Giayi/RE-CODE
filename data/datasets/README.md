# 資料集

本目錄為 RE-CODE 論文實驗所用**可公開子集**（程式題與需求文本）。

## 程式生成基準

| 路徑 | 說明 |
|------|------|
| `humaneval/humaneval_public_test.jsonl` | HumanEval 公開測試（159 題） |
| `humaneval/human-eval.jsonl` | HumanEval 題幹 |
| `mbpp/mbpp_public_test.jsonl` | MBPP 公開測試（427 題） |
| `mbpp/mbpp.jsonl`、`mbpp_test.jsonl` | MBPP 題目與測試對照 |

## 需求品質實驗（實驗一）

| 路徑 | 說明 |
|------|------|
| `pure/requirements_raw.json` | 原始需求列表 |
| `pure/requirements_clean.json` | 清洗後需求（實驗輸入常用） |
| `promise/` | PROMISE NFR 相關需求 |
| `frnfr/` | FR/NFR 樣本與清洗版本 |

HumanEval / MBPP 上游資料請同時引用原始基準論文；本包僅提供與本文實驗對齊的切片與公開測試檔。
