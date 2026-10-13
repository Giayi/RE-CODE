# 重複實驗掛機狀態

更新時間：2026-10-11 16:26

## 目前進度

- 已完成：`7 / 20`
  - HumanEval Direct：5/5
  - HumanEval Reflexion：2/5（run_3 進行中）
  - MBPP Direct：0/5
  - MBPP Reflexion：0/5

## 掛機機制

1. Docker 容器：`zjy_reflexion_exp`（`restart=unless-stopped`，掛載 `/data/llm/zjy/reflexion-main` → `/workspace`）
2. 主程序：容器內 `python -u run_repeat_experiments.py --rounds 5`
3. Host watchdog：每 10 分鐘檢查一次；若 runner 掛掉會自動重啟；滿 20 輪後自動 `--aggregate-only` 並退出

## 監控指令

```bash
# 即時日誌
tail -f /data/llm/zjy/reflexion-main/Output/repeat/runner.log

# watchdog 日誌
tail -f /data/llm/zjy/reflexion-main/Output/repeat/watchdog.host.log

# 當前彙總
cat /data/llm/zjy/reflexion-main/Output/repeat/summary/summary.md

# 已完成輪數
find /data/llm/zjy/reflexion-main/Output/repeat -name metrics.json | wc -l
```

## 預估剩餘時間

- HumanEval Reflexion 剩餘約 3 輪：約 1.5–2 小時
- MBPP Direct 5 輪：約 1–1.5 小時
- MBPP Reflexion 5 輪：約十數小時（最久）

全程可能需 **約 1 天**，結果會持續寫入 `Output/repeat/summary/`。
