# 實驗復現指引（路徑已對齊本資料集目錄）

> 公開資料包僅包含**實驗輸出與腳本**；復現時需自行配置 OpenAI / 其他模型 API，勿將密鑰寫入版本庫。

工作目錄請設為 `code/recode/`（與整合後的 MetaGPT 環境對齊時，可將 `third_party/metagpt` 作為執行根目錄）。

## 實驗三：程式碼生成（HumanEval / MBPP）

### HumanEval 生成

```bash
cd code/recode
python scripts/run_experiment3_humaneval.py --method metagpt --model gpt-4o-mini
python scripts/run_experiment3_humaneval.py --method metagpt_egrcm --model gpt-4o-mini
python scripts/run_experiment3_humaneval.py --method metagpt_prqe --model gpt-4o-mini --delay 2
python scripts/run_experiment3_humaneval.py --method metagpt_egrcm_prqe --model gpt-4o-mini --delay 2
```

輸出對應公開資料：`data/recode/experiment3_ablation/generations/`、`data/recode/experiment3_ablation/evaluate/`。

### HumanEval 評估（單次消融）

```bash
cd code/recode
for m in metagpt metagpt_egrcm metagpt_prqe metagpt_egrcm_prqe; do
  python scripts/evaluate_humaneval_pass1.py \
    -i ../../data/recode/experiment3_ablation/generations/$m/samples_gpt-4o-mini.jsonl \
    -o ../../data/recode/experiment3_ablation/evaluate/$m/pass1_result.json
done
```

### MBPP 生成與評估

資料集：`data/datasets/mbpp/mbpp_public_test.jsonl`

```bash
cd code/recode
python scripts/run_experiment3_mbpp.py --method metagpt \
  --dataset ../../data/datasets/mbpp/mbpp_public_test.jsonl \
  --model gpt-4o-mini \
  -o ../../data/recode/experiment3_ablation/generations
```

評估範例：

```bash
python scripts/evaluate_mbpp_pass1.py \
  -i ../../data/recode/experiment3_ablation/generations/metagpt_egrcm_prqe/samples_mbpp_gpt-4o-mini.jsonl \
  -o ../../data/recode/experiment3_ablation/evaluate/metagpt_egrcm_prqe/pass1_mbpp_result.json
```

## 重複實驗（5 輪，報告 mean ± std）

| 方法 | 腳本（於 `code/recode/scripts/`） | 公開結果目錄 |
|------|-----------------------------------|--------------|
| MetaGPT 基線 | `run_repeat_metagpt.py` | `data/baselines/metagpt_repeat/` |
| RE-CODE（EGRCM+PRQE） | `run_repeat_egrcm_prqe.py` | `data/recode/repeat_egrcm_prqe/` |
| Reflexion / Direct | （Reflexion 原倉庫流程） | `data/baselines/reflexion/repeat/` |
| ClarifyGPT | （ClarifyGPT 原流程） | `data/baselines/clarifygpt_repeat/` |
| OpenHands | （OpenHands 原流程） | `data/baselines/openhands/repeat_openhands/` |

## 舊路徑對照（論文草稿或日誌中可能出現）

| 舊路徑 | 新路徑 |
|--------|--------|
| `RE-CODE/results/datasets/` | `data/datasets/` |
| `RE-CODE/results/repeat experiment/` | `data/recode/repeat_egrcm_prqe/` |
| `RE-CODE/results/repeat_metagpt/` | `data/baselines/metagpt_repeat/` |
| `RE-CODE/results/full_experiments/` | `data/recode/experiment3_ablation/generations/` |
| `llm/LLM/` | `data/baselines/direct_llm/LLM/` |
| `reflexion/repeat/` | `data/baselines/reflexion/repeat/` |
| `ClarifyGPT/repeatout/` | `data/baselines/clarifygpt_repeat/` |
| `Openhands/Output/repeat experiment_openhands/` | `data/baselines/openhands/repeat_openhands/` |
| `Openhands/Output/`（單次） | `data/baselines/openhands/`（根目錄下單次輸出） |
