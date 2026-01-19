export CUDA_VISIBLE_DEVICES=1
ma_type=reg

beta=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

model_name=D2Net
seq_len=96

for model_name in D2Net
do
for pred_len in 96 192 336 720
do
    case $pred_len in
        96)  lradj='TSF';;
        192) lradj='sgdr';;
        336) lradj='sgdr';;
        720) lradj='sgdr';;
    esac
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id exchange_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Exp' \
    --itr 1 \
    --use_csm 0 \
    --use_mix 1 \
    --heads 3 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --lradj $lradj\
    --ma_type $ma_type 

done
done