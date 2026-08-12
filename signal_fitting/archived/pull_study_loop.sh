#!/bin/bash
# Run pull_study.py repeatedly, restarting Python every CHUNK masses to flush
# the RooFit/cppyy memory leak. Each invocation auto-resumes from results.csv.
# Stops when an invocation processes 0 new cells (everything done).
set -e

CHUNK=${CHUNK:-80}                  # masses per python invocation
N_TOYS=${N_TOYS:-100}
N_EVENTS=${N_EVENTS:-"5 10 15 20"}
CHANNELS=${CHANNELS:-"ee mumu"}
LOG_DIR=signal_fitting/outputs/RunIISummer20UL18/pull_study/logs
mkdir -p "$LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

iter=0
while true; do
  iter=$((iter+1))
  LOG=$LOG_DIR/pull_study_chunk${iter}_${RUN_TS}.log
  echo "[$(date)] === iteration $iter, log $LOG ==="
  prev_rows=$(wc -l < signal_fitting/outputs/RunIISummer20UL18/pull_study/results.csv 2>/dev/null || echo 0)
  python signal_fitting/pull_study.py \
    --all --n-toys $N_TOYS \
    --n-events $N_EVENTS \
    --channels $CHANNELS \
    --max-masses-per-run $CHUNK \
    > "$LOG" 2>&1 || echo "[$(date)] python exited non-zero (continuing)"
  new_rows=$(wc -l < signal_fitting/outputs/RunIISummer20UL18/pull_study/results.csv 2>/dev/null || echo 0)
  added=$((new_rows - prev_rows))
  echo "[$(date)] iteration $iter added $added rows (total $new_rows)"
  if [ "$added" -le 1 ]; then
    echo "[$(date)] no new rows; pull study complete."
    break
  fi
done
echo "[$(date)] DONE."
