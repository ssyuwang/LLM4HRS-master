model_name=LLM4HRSI
#32 64 128 256
for batch_size in 16
do
for lr in 0.001
do
for gpt_layer in 1
do
for enc_in in 576
do


python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path cha_all1_down1.csv \
  --model_id cha_all1_down1_mask_0.1 \
  --mask_rate 0.1 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 0 \
  --pred_len 0 \
  --patch_size 1 \
  --d_ff 768 \
  --stride 1 \
  --gpt_layer $gpt_layer \
  --d_model 768 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out 576 \
  --batch_size $batch_size \
  --train_epoch 100 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate $lr


done
done
done
done