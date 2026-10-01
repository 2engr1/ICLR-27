
# export CUDA_VISIBLE_DEVICES=0

model_name=TMT
d state 2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --dec_in 7 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --learning_rate 0.00008 \
  --itr 1\
  --M_len 256\
  --dropout 0.1 \
  --order 2 \
  --use_norm 0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 2 \
  --learning_rate 0.00004 \
  --itr 1 \
  --M_len 256\
  --dropout 0.1 \
  --order 2 \
  --use_norm 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.00003 \
  --itr 1 \
  --M_len 512\
  --dropout 0.1 \
  --order 2 \
  --use_norm 0
  

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.00003 \
  --itr 1 \
  --M_len 336\
  --dropout 0.1 \
  --order 2 \
  --use_norm 0
  