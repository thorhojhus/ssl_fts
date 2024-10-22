#!/bin/bash

# Default values for seq_len
seq_len=336

# List of models to run
# models=("DLinear" "FITS_DLinear" "DLinear_FITS")
models=("DLinear")

# List of pred_len values
pred_lens=(96)

# List of datasets to run
datasets=("exchange_rate")
features="M"

# Create necessary directories if they do not exist
if [ ! -d "./results" ]; then
    mkdir ./results
fi

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Set target and features based on dataset
    if [ "$dataset" = "exchange_rate" ]; then
        target="OT"
        channels=8
    elif [ "$dataset" = "GD" ] || [ "$dataset" = "MRO" ] || [ "$dataset" = "AAPL" ]; then
        target="Adj Close"
        channels=6
    else
        echo "Error: Unknown dataset. Please specify 'exchange_rate', 'GD', or 'MRO'."
        continue
    fi

    # Loop through each model
    for model in "${models[@]}"; do
        # Loop through each pred_len value
        for pred_len in "${pred_lens[@]}"; do
            echo "Running experiments for seq_len=${seq_len} pred_len=${pred_len} and dataset=${dataset}:"

            logname="${dataset}_${seq_len}_${pred_len}_${features}_channels_${channels}"
            result_dir="./results/${logname}"

            # Create result directory
            if [ ! -d "$result_dir" ]; then
                mkdir "$result_dir"
            fi

            # Create logs directory inside result directory
            logs_dir="${result_dir}/logs"
            if [ ! -d "$logs_dir" ]; then
                mkdir "$logs_dir"
            fi

            log_file="${logs_dir}/${model}.log"
            echo "Log file: $log_file"

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
              --itr 1 --batch_size 8 --learning_rate 0.0005 > "$log_file"

            python -u print_metrics.py
        done
    done
done