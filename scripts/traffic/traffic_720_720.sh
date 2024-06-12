#!/bin/bash

python run.py \
    --normalize True \
    --seq_len 720 \
    --pred_len 720 \
    --dominance_freq 0 \
    --train_and_finetune \
    --all_cols True \
    --batch_size 64 \
    --dataset traffic \
    --epochs 50 \
    --H_order 10 \
    --channels 862 \
    --use_original_datahandling \