#!/bin/bash

mkdir -p output

sem() {
  local max_jobs=$1
  local job_count=$(jobs -r -p | wc -l)
  while [ "$job_count" -ge "$max_jobs" ]; do
    sleep 1
    job_count=$(jobs -r -p | wc -l)
  done
}

MAX_JOBS=3

# Find and execute each script, redirecting output to separate files
find ./scripts -name '*.sh' -print0 | while IFS= read -r -d '' script; do
  sem "$MAX_JOBS"
  (
    script_name=$(basename "$script" .sh)
    bash "$script" > "output/${script_name}.out" 2>&1
  ) &
done

wait
