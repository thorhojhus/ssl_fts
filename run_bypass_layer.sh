#!/bin/bash

# Arrays for num_layers and num_hidden
pred_lens=(96 192 336 720)

# Outer loop for num_layers
for pred_len in "${pred_lens[@]}"
do
    echo "Pred_len $pred_len"
    
    python run.py \
        --normalize True \
        --seq_len 720 \
        --pred_len $pred_len \
        --dominance_freq 0 \
        --train_and_finetune \
        --all_cols False \
        --batch_size 64 \
        --model FITS_bypass_layer \
        --dataset ETTm1 \
        --epochs 50 \
        --use_original_datahandling \
        --no-print \
        --H_order 14 \
        --base_T 96
done