#!/bin/bash

models=("FITS" "real_deep_FITS" "FITS_bypass_layer" "deep_FITS" "deep_FITS_after_upscaler")
datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
pred_lens=(96 192 336 720)

# Iterate over each dataset, model and sequence length and execute the python script
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for pred_len in "${pred_lens[@]}"; do
            echo "Running model: $model with pred_len: $pred_len"
            python run.py \
                --normalize True \
                --seq_len 720 \
                --pred_len $pred_len \
                --dominance_freq 0 \
                --all_cols False \
                --batch_size 64 \
                --dataset $dataset \
                --epochs 50 \
                --use_original_datahandling \
                --train_and_finetune \
                --model $model \
                --num_layers 3 \
                --num_hidden 128 \
                --H_order 6
        done
    done
done