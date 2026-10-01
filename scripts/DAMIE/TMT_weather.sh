

model_name=TMT

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128\
  --learning_rate 0.0004 \
  --train_epochs 20\
  --d_state 2 \
  --d_ff 128\
  --itr 1 \
  --M_len 192 \
  --dropout 0.1 \
  --order 2 \
  --gpu 6 
  


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.0003 \
  --train_epochs 20\
  --d_model 192\
  --d_state 2 \
  --d_ff 192\
  --itr 1 \
  --M_len 336 \
  --dropout 0.1 \
  --gpu 6


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00009 \
  --train_epochs 20\
  --d_model 336\
  --d_state 2 \
  --d_ff 336\
  --itr 1 \
  --M_len 526 \
  --dropout 0.1 \
  --gpu 6


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs 20\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --itr 1 \
  --M_len 768 \
  --dropout 0.1 \
  --gpu 6