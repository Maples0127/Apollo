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
echo "== latest PESQ eval (if any) =="
LATEST_PESQ=$(ls -1t "$EXP"/Apollo*/pesq_*.jsonl 2>/dev/null | head -1)
if [ -n "${LATEST_PESQ:-}" ]; then
  python3 - "$LATEST_PESQ" <<'PYEOF'
import collections
import json
import math
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
by_ckpt = collections.defaultdict(list)
for row in rows:
    by_ckpt[(row.get("epoch"), row.get("ckpt"))].append(row)

print(sys.argv[1])
for (epoch, _ckpt), group in sorted(by_ckpt.items(), key=lambda kv: (kv[0][0] is None, kv[0][0])):

    def mean(key):
        vals = [r[key] for r in group
                if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"  epoch={epoch} n={len(group)} "
          f"SI-SDR codec={mean('sisdr_codec'):.2f} model={mean('sisdr_model'):.2f} "
          f"| PESQ codec={mean('pesq_codec'):.3f} model={mean('pesq_model'):.3f}")
PYEOF
else
  echo "  none yet"
fi

echo
echo "== latest held-out eval =="
for f in $(ls -1t "$EXP"/*/eval_*.json 2>/dev/null | head -2); do
  python3 - "$f" <<'PYEOF'
import json
import sys

d = json.load(open(sys.argv[1]))
print(f"{sys.argv[1]}\n  epoch={d.get('epoch')} codec={d.get('codec_sisdr_mean'):.2f} "
      f"model={d.get('model_sisdr_mean'):.2f} delta={d.get('delta_mean'):+.2f}")
PYEOF
done
