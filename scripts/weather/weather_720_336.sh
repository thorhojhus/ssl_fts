#!/bin/bash

python run.py \
    --normalize True \
    --seq_len 720 \
    --pred_len 336 \
    --dominance_freq 0 \
    --train_and_finetune \
    --all_cols False \
    --batch_size 64 \
    --dataset weather \
    --epochs 50 \
    --H_order 12 \
    --base_T 144 \
    --channels 21 \
    --individual \
    --use_original_datahandling \
