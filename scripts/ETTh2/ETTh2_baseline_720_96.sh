#!/bin/bash

python run.py \
    --normalize True \
    --seq_len 720 \
    --pred_len 96 \
    --dominance_freq 0 \
    --train_and_finetune \
    --all_cols False \
    --batch_size 64 \
    --dataset ETTh2 \
    --epochs 50 \
    --use_original_datahandling \
    --use_baseline \
    --H_order 6 \
