data_name="ecom"
pool_type="cls"
gpu='0,1'
bert_model="bert-base-chinese"
output_dir="output_model/"$data_name"_model/"$bert_model"_"$pool_type""

mkdir -p $output_dir

num_gpu=2
batch_size=32

train_data="../data/"$data_name"/train/"

CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --master_port 2025 --nproc_per_node $num_gpu run_training.py \
  --output_dir $output_dir \
  --model_name_or_path $bert_model \
  --do_train \
  --save_steps 1000 \
  --model_type bert \
  --per_device_train_batch_size $batch_size \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.1 \
  --learning_rate 2e-5 \
  --num_train_epochs 6 \
  --dataloader_drop_last \
  --overwrite_output_dir \
  --dataloader_num_workers 10 \
  --max_seq_length 256 \
  --train_dir $train_data \
  --weight_decay 0.01 \
  --pool_type $pool_type --fp16
