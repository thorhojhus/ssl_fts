#!/bin/bash

# Define the seq_len, model, and pred_len values
seq_lens=(336 720)
models=("FITS" "DLinear_FITS")
scripts=("mro.sh")

# Loop over each pred_len and seq_len combination
for seq_len in "${seq_lens[@]}"; do
  for script in "${scripts[@]}"; do
    for model in "${models[@]}"; do
      echo "Running $script with seq_len=$seq_len and model=$model"
      sh scripts/EXP-LongForecasting/DLinear/$script -s $seq_len -m $model
    done
  done
done
