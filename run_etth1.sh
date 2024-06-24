#pred_lens=(96 192 336 720)
#pred_lens=(720)
pred_lens=(24 36 48 60)

# Outer loop for num_layers
for pred_len in "${pred_lens[@]}"
do
    echo "Pred_len $pred_len"
    
    python run.py \
        --normalize True \
        --seq_len 72 \
        --pred_len $pred_len \
        --dominance_freq 0 \
        --train_and_finetune \
        --all_cols False \
        --batch_size 64 \
        --dataset national_illness \
        --epochs 50 \
        --use_original_datahandling \
        --no-print \
        --base_T 52 \
        --H_order 1
done