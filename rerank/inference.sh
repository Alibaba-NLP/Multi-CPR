DEV_DATA_PATH=dev/video
MODEL_PATH=fintune_models/video_bert_base
RESULT_PATH=result/video_bert_base_rank_res

CUDA_VISIBLE_DEVICES=4 python run_marco.py \
  --output_dir ${MODEL_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name bert-base-chinese \
  --do_predict \
  --max_len 256 \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 8 \
  --pred_path ${DEV_DATA_PATH}/dev.top1000.json  \
  --pred_id_file  ${DEV_DATA_PATH}/dev.top1000.label.txt \
  --rank_score_path ${RESULT_PATH}
