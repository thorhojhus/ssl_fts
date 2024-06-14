#!/bin/bash

mkdir -p output

# Define an array of terms to exclude
exclude_terms=("arima" "baseline")

# Construct the find command with exclusion terms
find_command="find ./scripts -name '*.sh'"
for term in "${exclude_terms[@]}"; do
  find_command+=" ! -name '*${term}*'"
done
find_command+=" -print0"

# Execute the dynamically constructed find command
eval "$find_command" | while IFS= read -r -d '' script; do
  script_name=$(basename "$script" .sh)
  echo "Running $script..."
  bash "$script" > "output/${script_name}.out" 2>&1
  echo "$script finished."
done