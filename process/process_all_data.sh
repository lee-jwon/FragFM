#!/usr/bin/env bash

PROCESS_TAG="all" # e.g. "all", "20260101"
NUM_SHARDS=512 # Increase for larger datasets
MAX_JOBS=12 # Adjust based on your system's capacity

datasets=$1 # e.g. "debug, moses, guacamol, npgen"

cleanup() {
  echo -e "\nInterrupted. Killing all child processes..."
  kill -- -$$ 2>/dev/null
  exit 1
}
trap cleanup SIGINT SIGTERM

for dataset in $datasets; do
  running=0
  finished=0
  declare -A pid_to_shard
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    python process/process_to_lmdb.py "$dataset" brics "$PROCESS_TAG" "$i" "$NUM_SHARDS" &
    pid_to_shard[$!]=$i
    ((running++))
    if ((running >= MAX_JOBS)); then
      wait -n -p done_pid
      ((running--))
      ((finished++))
      echo "[${dataset}] Shard ${pid_to_shard[$done_pid]} done (${finished}/${NUM_SHARDS})"
    fi
  done
  while ((running > 0)); do
    wait -n -p done_pid
    ((running--))
    ((finished++))
    echo "[${dataset}] Shard ${pid_to_shard[$done_pid]} done (${finished}/${NUM_SHARDS})"
  done
  echo "All jobs for $dataset completed."

  python process/process_merge_lmdbs.py "data/processed/${dataset}_brics_${PROCESS_TAG}/"
  python process/process_fragment_from_lmdb.py "$dataset" brics "$PROCESS_TAG"
done

echo "All data processing completed."
