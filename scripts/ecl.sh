#export CUDA_VISIBLE_DEVICES=1
ma_type=reg
#alpha=0.1
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
        96)  learning_rate=0.002 heads=3 dropout_rate=0.1;;
        192) learning_rate=0.002 heads=3 dropout_rate=0.1;;
        336) learning_rate=0.0005 heads=2 dropout_rate=0.1;;
        720) learning_rate=0.0005 heads=2 dropout_rate=0.5;;
    esac
  python -u run.py \
    --is_training 1 \
    --root_path ../Amplifier-main/dataset/electricity/ \
    --data_path electricity.csv \
    --model_id electricity_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 \
    --use_csm 1 \
    --use_mix 1 \
    --heads $heads \
    --batch_size 48 \
    --learning_rate $learning_rate \
    --lradj 'TSF'\
    --patience 20 \
    --dropout_rate $dropout_rate \
    --ma_type $ma_type 

done
done