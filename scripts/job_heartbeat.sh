#!/usr/bin/env bash
# Heartbeat / stall detector for the Apollo training job.
#
#   scripts/job_heartbeat.sh [jobid] [stall_minutes]
#
# exit 0 = running and the log advanced recently
# exit 1 = not running, or the log has not changed for >= stall_minutes
# exit 2 = no log found
#
# Only touches /data (node-local). Never scans /home.
set -uo pipefail

EXP="${EXP_DIR:-/data/cse12212308/exp}"
JOB="${1:-}"
STALL_MIN="${2:-15}"
USER_NAME="$(whoami)"

if [ -n "$JOB" ]; then
  LOG=$(ls -1t "$EXP"/logs/*_"${JOB}".out 2>/dev/null | head -1)
  STATE=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null || true)
  [ -z "$STATE" ] && STATE=$(sacct -j "$JOB" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')
else
  LOG=$(ls -1t "$EXP"/logs/*.out 2>/dev/null | head -1)
  JOB=$(basename "${LOG:-none}" | sed -E 's/.*_([0-9]+)\.out/\1/')
  STATE=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null || true)
fi

if [ -z "${LOG:-}" ] || [ ! -f "$LOG" ]; then
  echo "NO_LOG job=${JOB:-?} exp=$EXP"
  exit 2
fi

AGE_MIN=$(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 60 ))
STEP=$(tail -c 4000 "$LOG" | tr '\r' '\n' | grep -E "Epoch [0-9]+:" | tail -1 | grep -oE "\| [0-9]+/" | tr -d '| /')
VAL=$(tr '\r' '\n' < "$LOG" | grep -ao "\[epoch [0-9]*\][^[]*" | tail -1)

printf 'job=%s state=%s log_age=%smin step=%s\n' "${JOB:-?}" "${STATE:-UNKNOWN}" "$AGE_MIN" "${STEP:-?}"
[ -n "${VAL:-}" ] && printf 'last_epoch_summary: %s\n' "$VAL"

if [ "${STATE:-}" != "RUNNING" ] && [ "${STATE:-}" != "PENDING" ]; then
  echo "STATUS: not running (state=${STATE:-UNKNOWN})"
  exit 1
fi

if [ "$AGE_MIN" -ge "$STALL_MIN" ]; then
  echo "STATUS: STALLED (log untouched for ${AGE_MIN}min, threshold ${STALL_MIN}min)"
  exit 1
fi

echo "STATUS: OK"
exit 0
