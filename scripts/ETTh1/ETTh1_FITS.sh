#!/bin/bash

pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"; do
        echo "Pred len: $pred_len"
        python run.py \
            --normalize True \
            --seq_len 360 \
            --pred_len $pred_len \
            --dominance_freq 0 \
            --all_cols False \
            --batch_size 64 \
            --dataset ETTh1 \
            --epochs 50 \
            --use_original_datahandling \
            --train_and_finetune \
            --model FITS \
            --no-print \
            --H_order 6 \

done

