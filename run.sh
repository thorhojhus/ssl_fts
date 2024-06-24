#!/bin/bash

# Arrays for num_layers and num_hidden
#num_layers_arr=(2 3 4 5)
#num_layers_arr=(1)
#num_hidden_arr=(64 128 256 512)
#num_hidden_arr=(64)
pred_lens=(96 192 336 720)
#pred_lens=(720)

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
        --dataset weather \
        --epochs 50 \
        --use_original_datahandling \
        --no-print \
        --H_order 12 \
        --channels 21 \
        --individual \
        --base_T 144
done