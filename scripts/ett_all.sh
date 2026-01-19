#export CUDA_VISIBLE_DEVICES=1
ma_type=reg
#alpha=0.9
beta=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

model_name=D2Net
seq_len=96

for pred_len in 96 192 336 720
do
    case $pred_len in
        96)  learning_rate=0.0005 lradj='sgdr';;
        192) learning_rate=0.0005 lradj='TSF';;
        336) learning_rate=0.0005 lradj='sgdr';;
        720) learning_rate=0.0005 lradj='TSF';;
    esac
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --use_csm 1 \
    --use_mix 1 \
    --heads 3 \
    --batch_size 2048 \
    --learning_rate $learning_rate \
    --lradj $lradj\
    --ma_type $ma_type 

done

for pred_len in 96 192 336 720
 do
     case $pred_len in
         96)  lradj='sigmoid';;
         192) lradj='sigmoid';;
         336) lradj='sigmoid';;
         720) lradj='sigmoid';;
     esac
   python -u run.py \
     --is_training 1 \
     --root_path ./dataset/ETT-small/ \
     --data_path ETTh2.csv \
     --model_id ETTh2_$pred_len'_'$ma_type \
     --model $model_name \
     --data ETTh2 \
     --features M \
     --seq_len $seq_len \
     --pred_len $pred_len \
     --enc_in 7 \
     --des 'Exp' \
     --itr 1 \
     --use_csm 1 \
     --use_mix 1 \
     --heads 2 \
     --dropout_rate 0.6 \
     --batch_size 2048 \
     --learning_rate 0.0005 \
     --lradj $lradj\
     --ma_type $ma_type 

done

 for pred_len in 96 192 336 720
 do
     case $pred_len in
         96)  lradj='sigmoid';;
         192) lradj='sigmoid';;
         336) lradj='sigmoid';;
         720) lradj='sigmoid';;
     esac
   python -u run.py \
     --is_training 1 \
     --root_path ./dataset/ETT-small/ \
     --data_path ETTm1.csv \
     --model_id ETTm1_$pred_len'_'$ma_type \
     --model $model_name \
     --data ETTm1 \
     --features M \
     --seq_len $seq_len \
     --pred_len $pred_len \
     --enc_in 7 \
     --des 'Exp' \
     --itr 1 \
     --use_csm 1 \
     --heads 2 \
     --use_mix 1 \
     --dropout_rate 0.6 \
     --batch_size 2048 \
     --learning_rate 0.0005 \
     --lradj $lradj\
     --ma_type $ma_type

 done

for pred_len in 96 192 336 720
do
    case $pred_len in
        96)  lradj='sigmoid';;
        192) lradj='sigmoid';;
        336) lradj='sigmoid';;
        720) lradj='sigmoid';;
    esac
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --use_csm 1 \
    --use_mix 1 \
    --heads 2 \
    --dropout_rate 0.6 \
    --batch_size 2048 \
    --learning_rate 0.0001 \
    --lradj $lradj\
    --ma_type $ma_type 

done