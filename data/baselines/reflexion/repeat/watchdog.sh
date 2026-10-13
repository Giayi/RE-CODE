#!/usr/bin/env bash
# Restart run_repeat_experiments.py if it is not running, until all 20 metrics exist.
set -euo pipefail

CONTAINER="${CONTAINER:-zjy_reflexion_exp}"
LOG="/workspace/Output/repeat/watchdog.log"
RUNNER_LOG="/workspace/Output/repeat/runner.log"
TARGET_METRICS=20

log() {
  echo "[$(date '+%F %T')] $*" | tee -a /data/llm/zjy/reflexion-main/Output/repeat/watchdog.host.log
}

count_ok_metrics() {
  docker exec "$CONTAINER" bash -lc '
    python3 - <<PY
from pathlib import Path
import json
root = Path("/workspace/Output/repeat")
n = 0
for p in root.glob("*/*/run_*/metrics.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if d.get("status") == "ok":
        n += 1
print(n)
PY
'
}

is_runner_alive() {
  docker exec "$CONTAINER" bash -lc 'pgrep -f "python -u run_repeat_experiments.py" >/dev/null'
}

start_runner() {
  docker exec -d "$CONTAINER" bash -lc '
    source /opt/conda/etc/profile.d/conda.sh
    conda activate reflexion_env
    mkdir -p /workspace/Output/repeat
    cd /workspace/programming_runs
    # append so history is preserved
    echo "[$(date -Iseconds)] watchdog restart" >> /workspace/Output/repeat/runner.log
    python -u run_repeat_experiments.py --rounds 5 >> /workspace/Output/repeat/runner.log 2>&1
  '
}

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  log "ERROR: container $CONTAINER is not running"
  exit 1
fi

ok=$(count_ok_metrics | tr -d '[:space:]')
log "ok_metrics=${ok}/${TARGET_METRICS}"

if [[ "${ok}" -ge "${TARGET_METRICS}" ]]; then
  log "All ${TARGET_METRICS} runs completed. Finalizing summary."
  docker exec "$CONTAINER" bash -lc '
    source /opt/conda/etc/profile.d/conda.sh
    conda activate reflexion_env
    cd /workspace/programming_runs
    python -u run_repeat_experiments.py --aggregate-only >> /workspace/Output/repeat/runner.log 2>&1
  '
  log "Done."
  exit 0
fi

if is_runner_alive; then
  log "Runner already alive; nothing to do."
  exit 0
fi

log "Runner not found; restarting..."
start_runner
sleep 5
if is_runner_alive; then
  log "Runner restarted successfully."
  exit 0
fi

log "ERROR: failed to restart runner"
exit 1
