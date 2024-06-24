# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model="FITS"

# Parse command line arguments
while getopts s:m: flag
do
    case "${flag}" in
        s) seq_len=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

target="OT"
dataset="ETTh2"
logname=""$model"1.0_"$dataset"_"$seq_len""

pred_len=96

echo "logs/LongForecasting/${logname}_"$pred_len".log"

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path "$dataset.csv" \
  --model_id "${dataset%.*}"_"$seq_len"_"$pred_len" \
  --model "$model" \
  --data custom \
  --features M \
  --target "$target" \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/${logname}_"$pred_len".log

python -u print_metrics.py

pred_len=192

echo "logs/LongForecasting/${logname}_"$pred_len".log"

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path "$dataset.csv" \
  --model_id "${dataset%.*}"_"$seq_len"_"$pred_len" \
  --model "$model" \
  --data custom \
  --features M \
  --target "$target" \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/${logname}_"$pred_len".log

python -u print_metrics.py

pred_len=336

echo "logs/LongForecasting/${logname}_"$pred_len".log"

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path "$dataset.csv" \
  --model_id "${dataset%.*}"_"$seq_len"_"$pred_len" \
  --model "$model" \
  --data custom \
  --features M \
  --target "$target" \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/${logname}_"$pred_len".log

python -u print_metrics.py

pred_len=720

echo "logs/LongForecasting/${logname}_"$pred_len".log"

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path "$dataset.csv" \
  --model_id "${dataset%.*}"_"$seq_len"_"$pred_len" \
  --model "$model" \
  --data custom \
  --features M \
  --target "$target" \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/${logname}_"$pred_len".log

python -u print_metrics.py