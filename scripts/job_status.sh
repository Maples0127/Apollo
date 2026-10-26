#!/usr/bin/env bash
# One-shot, bounded status snapshot for the Apollo training run.
#
#   scripts/job_status.sh            # newest apollo log
#   scripts/job_status.sh 39460      # specific job id
#
# Safe for auditors/agents: only reads /data (node-local), never scans /home,
# finishes in a couple of seconds.
set -uo pipefail

EXP="${EXP_DIR:-/data/cse12212308/exp}"
USER_NAME="$(whoami)"
JOB="${1:-}"

echo "== when/host =="
date
hostname

echo
echo "== slurm (mine) =="
squeue -u "$USER_NAME" -o "%.8i %.16j %.10T %.10M %.10L %.20R" 2>/dev/null || true

if [ -n "$JOB" ]; then
  LOG=$(ls -1t "$EXP"/logs/*_"${JOB}".out 2>/dev/null | head -1)
else
  LOG=$(ls -1t "$EXP"/logs/*.out 2>/dev/null | head -1)
fi

if [ -z "${LOG:-}" ]; then
  echo "no training log found under $EXP/logs"
  exit 0
fi

echo
echo "== log: $LOG =="
ls -l --time-style=+%m-%d_%H:%M "$LOG" | awk '{print "size="$5" mtime="$6}'

echo
echo "== latest progress =="
tail -c 4000 "$LOG" | tr '\r' '\n' | grep -E "Epoch [0-9]+:" | tail -1 | cut -c1-200

echo
echo "== last epoch summary (G/D components) =="
tr '\r' '\n' < "$LOG" | grep -ao "\[epoch [0-9]*\][^[]*" | tail -3

echo
echo "== checkpoints =="
ls -1t "$EXP"/Apollo*/checkpoints/*.ckpt 2>/dev/null | head -5
echo "-- disk --"
df -h /data | tail -1

echo
echo "== latest held-out eval =="
for f in $(ls -1t "$EXP"/*/eval_*.json 2>/dev/null | head -2); do
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"{sys.argv[1]}\n  epoch={d.get('epoch')} codec={d.get('codec_sisdr_mean'):.2f} "
      f"model={d.get('model_sisdr_mean'):.2f} delta={d.get('delta_mean'):+.2f}")
PY
done
