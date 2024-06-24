#!/bin/bash

# Default values for seq_len
seq_len=336

# List of models to run
models=("DLinear" "FITS" "FITS_100" "DLinear_FITS" "FITS_DLinear")

# List of pred_len values
pred_lens=(96 192 336 720)

# Create necessary directories if they do not exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Target and dataset
target="Adj Close"
dataset="MRO"
features="M"

# Set channels based on dataset
if [ "$dataset" = "exchange_rate" ]; then
    channels=7
elif [ "$dataset" = "GD" ] || [ "$dataset" = "MRO" ]; then
    channels=6
else
    echo "Error: Unknown dataset. Please specify 'exchange_rate', 'GD', or 'MRO'."
    exit 1
fi

# Loop through each pred_len value
for pred_len in "${pred_lens[@]}"; do
    echo "Running experiments for seq_len=${seq_len} pred_len=${pred_len} and dataset=${dataset}:"

    # Loop through each model
    for model in "${models[@]}"; do
        if [ "$features" = "S" ]; then
            logname="${model}_${dataset}_${seq_len}_${pred_len}_${features}_channels_1_dominance_freq_10"
        else
            logname="${model}_${dataset}_${seq_len}_${pred_len}_${features}_channels_${channels}_dominance_freq_formula"
        fi

        echo "logs/LongForecasting/${logname}.log"

        python -u run_longExp.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path "$dataset.csv" \
          --model_id "${logname}" \
          --model "$model" \
          --data custom \
          --features "$features" \
          --target "$target" \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in $channels \
          --des 'Exp' \
          --itr 1 --batch_size 8 --learning_rate 0.0005 > "logs/LongForecasting/${logname}.log"

        python -u print_metrics.py
    done
done