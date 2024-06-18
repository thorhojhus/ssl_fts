#!/bin/bash

python run.py \
    --normalize True \
    --seq_len 720 \
    --pred_len 96 \
    --dominance_freq 0 \
    --train_and_finetune \
    --batch_size 64 \
    --dataset motor \
    --use_original_datahandling \
    --epochs 50 \
    --H_order 6 \
    --target "PM" \
    --features "MS" \
    --save_state_dict
