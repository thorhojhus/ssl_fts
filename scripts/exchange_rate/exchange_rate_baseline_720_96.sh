#!/bin/bash

python run.py \
    --normalize True \
    --seq_len 720 \
    --pred_len 96 \
    --dominance_freq 0 \
    --train_and_finetune \
    --all_cols False \
    --batch_size 64 \
    --dataset exchange_rate \
    --epochs 50 \
    --use_original_datahandling \
    --test_only \
    --model NF \
    --H_order 6
