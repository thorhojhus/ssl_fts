#!/bin/bash

python run.py \
    --normalize True \
    --seq_len 72 \
    --pred_len 36 \
    --dominance_freq 0 \
    --train_and_finetune \
    --all_cols True \
    --batch_size 16 \
    --dataset national_illness \
    --epochs 50 \
    --use_original_datahandling \
    --H_order 6
