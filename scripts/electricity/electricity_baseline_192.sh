#!/bin/bash

python run.py \
    --seq_len 720 \
    --pred_len 192 \
    --dataset electricity \
    --channels 321 \
    --use_original_datahandling \
    --use_baseline \
