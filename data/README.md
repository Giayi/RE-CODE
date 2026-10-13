# 實驗資料索引

本目錄為論文附帶的**可公開實驗輸出**，預設模型為 **GPT-4o-mini**。評測基準：

- **HumanEval**：`datasets/humaneval/humaneval_public_test.jsonl`（159 題，公開測試）
- **MBPP**：`datasets/mbpp/mbpp_public_test.jsonl`（427 題，公開測試）

主要指標：**Pass@1**；重複實驗報告 **mean ± std**（樣本標準差，ddof=1）。

---

## `datasets/`

實驗一／需求品質與程式題幹資料。

| 子目錄 | 內容 |
|--------|------|
| `humaneval/` | HumanEval 題幹與公開測試 |
| `mbpp/` | MBPP 題目與公開測試 |
| `pure/`、`promise/`、`frnfr/` | 需求文本與 NFR 相關子集（實驗一） |

詳見 `datasets/README.md`。

---

## `baselines/`（對照方法）

| 目錄 | 方法 | 摘要文件 |
|------|------|----------|
| `direct_llm/LLM/` | Direct LLM（simple prompt）單次生成 | 各子目錄內 `*.public_eval_summary.json` |
| `reflexion/repeat/` | Direct vs Reflexion，各 5 輪 | `reflexion/repeat/summary/summary.md` |
| `reflexion/token_and_stats_gpt4o_mini/` | Reflexion token 統計彙總 | `token_consumption_and_stats.md` |
| `metagpt_repeat/` | MetaGPT 程式生成基線，5 輪 | `summary.md` |
| `clarifygpt_repeat/` | ClarifyGPT，5 輪 | `report.md` |
| `openhands/repeat_openhands/` | OpenHands CodeAct，5 輪 | `aggregate/REPORT.md` |

各輪次目錄通常包含：`metrics` / `summary` JSON、生成 `samples` 或 `results` JSON、`eval_pass1` 或 `public_eval*.jsonl`、以及 `generation.log` / `evaluation.log`（如有）。

---

## `recode/`（本文方法 RE-CODE）

RE-CODE = MetaGPT + **EGRCM**（熵引導需求澄清）+ **PRQE**（需求品質評估與精化），程式任務另經 Code Adapter 與 RPC。

| 目錄 | 說明 |
|------|------|
| `repeat_egrcm_prqe/` | 完整 RE-CODE 管線，HumanEval/MBPP 各 5 輪重複實驗 | `summary.md` |
| `experiment3_ablation/generations/` | 消融：`metagpt` / `metagpt_egrcm` / `metagpt_prqe` / `metagpt_egrcm_prqe` 生成樣本 |
| `experiment3_ablation/evaluate/` | 上述四種方法的 Pass@1 評估 JSON |
| `full_pipeline_run/` | 單次端到端跑通產物（樣本、token、評估） | `summary.md` |
| `egrcm/` | EGRCM 對話/記錄與彙總 | `analysis_summary.md`（如有） |
| `token_analysis/` | MBPP 上各環節 token 分解 | `stage_token_analysis.md` |

---

## 檔案命名慣例

- `samples_*.jsonl` / `gen/*.jsonl`：模型生成程式
- `eval_pass1.json`、`pass1_result.json`、`*_summary.json`：Pass@1 與通過題數
- `token_usage_*.json`：按題或按輪的 token 統計
- `meta.json` / `run_meta.json`：單輪實驗配置與時間戳

復現命令與舊路徑對照見 [`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md)。
