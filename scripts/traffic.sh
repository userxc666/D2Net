ma_type=reg
#alpha=0.3
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
        96)  learning_rate=0.005 lradj='sgdr' heads=3 dropout_rate=0.1;;
        192) learning_rate=0.003 lradj='sgdr' heads=3 dropout_rate=0.1;;
        336) learning_rate=0.0018 lradj='TSF' heads=4 dropout_rate=0.6;;
        720) learning_rate=0.001 lradj='TSF' heads=2 dropout_rate=0.6;;
    esac
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --use_csm 1 \
    --use_mix 1 \
    --heads $heads \
    --batch_size 32 \
    --patience 20 \
    --dropout_rate $dropout_rate \
    --learning_rate $learning_rate \
    --lradj $lradj \
    --ma_type $ma_type 

done
done