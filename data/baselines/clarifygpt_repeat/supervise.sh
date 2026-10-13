#!/bin/bash
# Resume the 5-round ClarifyGPT job if the runner exits before it finishes.
cd /data/llm/zjy/ClarifyGPT-main || exit 1
LOG=output1/repeatout/supervisor.log
echo "$(date -Is) supervisor start pid=$$" >> "$LOG"
while true; do
  state=$(python3 - <<'PY'
import os
import json
from pathlib import Path

base = Path("output1/repeatout")
progress_path = base / "progress.json"
if progress_path.exists():
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("status") == "complete":
        print("complete")
        raise SystemExit
pid_path = base / "experiment.pid"
alive = False
if pid_path.exists():
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        alive = True
    except (OSError, ValueError):
        alive = False
print("alive" if alive else "dead")
PY
)
  echo "$(date -Is) state=$state" >> "$LOG"
  if [ "$state" = "complete" ]; then
    echo "$(date -Is) experiment complete, supervisor exit" >> "$LOG"
    exit 0
  fi
  if [ "$state" = "dead" ]; then
    echo "$(date -Is) runner missing, resuming" >> "$LOG"
    python3 -u src/run_repeat_eval.py --runs 5 --datasets humaneval mbpp_public --output output1/repeatout >> output1/repeatout/experiment.log 2>&1
    echo "$(date -Is) runner exit=$?" >> "$LOG"
  fi
  sleep 90
done
