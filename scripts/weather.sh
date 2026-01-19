#export CUDA_VISIBLE_DEVICES=1
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
        96)  lradj='sigmoid' heads=3 dropout_rate=0.6;;
        192) lradj='sigmoid' heads=3 dropout_rate=0.6;;
        336) lradj='TSF' heads=3 dropout_rate=0.6;;
        720) lradj='TSF' heads=3 dropout_rate=0.6;;
    esac
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 \
    --use_csm 1 \
    --use_mix 1 \
    --heads $heads \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --patience 20 \
    --dropout_rate $dropout_rate \
    --lradj $lradj\
    --ma_type $ma_type 

done
done