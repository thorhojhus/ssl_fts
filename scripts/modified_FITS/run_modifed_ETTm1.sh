# !/bin/bash

# Array of models to run
models=("real_deep_FITS" "FITS_bypass_layer" "deep_FITS" "deep_FITS_after_upscaler")

# Iterate over each model and execute the python script
for model in "${models[@]}"; do
    echo "Running model: $model"
    python run.py \
        --normalize True \
        --seq_len 720 \
        --pred_len 720 \
        --dominance_freq 0 \
        --all_cols False \
        --batch_size 64 \
        --dataset ETTm1 \
        --epochs 50 \
        --use_original_datahandling \
        --train_and_finetune \
        --model $model \
        --num_layers 3 \
        --num_hidden 128 \
        --H_order 6
done