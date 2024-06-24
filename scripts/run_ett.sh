#!/bin/bash

# Define the seq_len and model values
seq_lens=(336 720)
models=("FITS" "DLinear" "DLinear_FITS")
scripts=("etth1.sh" "etth2.sh" "ettm1.sh" "ettm2.sh")

# Loop over each seq_len and model combination
for seq_len in "${seq_lens[@]}"; do
  for model in "${models[@]}"; do
    for script in "${scripts[@]}"; do
      echo "Running $script with seq_len=$seq_len and model=$model"
      sh scripts/EXP-LongForecasting/DLinear/$script -s $seq_len -m $model
    done
  done
done
