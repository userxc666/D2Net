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
seq_len=36

for model_name in D2Net
do 
for pred_len in 24 36 48 60
do
    case $pred_len in
        24) learning_rate=0.01 patience=20 lradj='sigmoid' train_epochs=100;;
        36) learning_rate=0.01 patience=20 lradj='sigmoid' train_epochs=100;;
        48) learning_rate=0.01 patience=20 lradj='sigmoid' train_epochs=20;;
        60) learning_rate=0.01 patience=20 lradj='sigmoid' train_epochs=100;;
    esac
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --use_csm 0 \
    --use_mix 1 \
    --heads 3 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 32 \
    --learning_rate $learning_rate \
    --lradj $lradj\
    --ma_type $ma_type 

done
done