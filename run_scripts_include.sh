#!/bin/bash

mkdir -p output

# Define an array of terms to include
include_terms=("baseline")

# Construct the find command with inclusion terms
find_command="find ./scripts -name '*.sh' \( -false"
for term in "${include_terms[@]}"; do
  find_command+=" -o -name '*${term}*'"
done
find_command+=" \) -print0"

# Execute the dynamically constructed find command
eval "$find_command" | while IFS= read -r -d '' script; do
  script_name=$(basename "$script" .sh)
  echo "Running $script..."
  bash "$script" > "output/${script_name}.out" 2>&1
  echo "$script finished."
done
