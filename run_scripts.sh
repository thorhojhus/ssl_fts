#!/bin/bash

mkdir -p output

# Find and execute each script sequentially, redirecting output to separate files
find ./scripts -name '*.sh' -print0 | while IFS= read -r -d '' script; do
  script_name=$(basename "$script" .sh)
  echo "Running $script..."
  bash "$script" > "output/${script_name}.out" 2>&1
  echo "$script finished."
done
