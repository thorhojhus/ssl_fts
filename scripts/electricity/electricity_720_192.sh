#!/bin/bash

python run.py \
    --normalize True \
    --seq_len 720 \
    --pred_len 192 \
    --dominance_freq 0 \
    --train_and_finetune \
    --all_cols False \
    --batch_size 64 \
    --dataset electricity \
    --epochs 50 \
    --channels 321 \
    --use_original_datahandling \
    --H_order 10
