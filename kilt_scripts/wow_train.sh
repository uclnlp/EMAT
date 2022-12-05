#!/bin/bash -l

set -e
set -u

DST_DIR="case-augmented-transformer-master" # change to your project root
cd ${DST_DIR}

LOAD_EXP_NAME="KL=3;kdim=1536;CL=10;VL=11;VN=10;async_cat_k+v;t5-base;from-t5-base;"
EXP_NAME="base;wow_kilt;freeze;8e-5;20epoch;CL=10;VL=11;"
DATA_NAME="wow_kilt"

DEVICE="0"

echo "Use Device ${DEVICE}"

# Train wow-EMAT-SKSV

CUDA_VISIBLE_DEVICES=${DEVICE} python kilt_main.py \
  --key_layer=3 \
  --cat_layer=10 \
  --value_layer=11 \
  --query_batch_size=256 \
  --build_mem_batch_size=12000 \
  --project_name="${DATA_NAME^^}-CAT" \
  --exp_name=${EXP_NAME} \
  --batch_local_positive_num=5 \
  --pos_from_top=128 \
  --do_eval \
  --kvm_seg_n=5 \
  --values_with_order \
  --value_fusion_method="async_cat_k_delay+v" \
  --num_values=10 \
  --qa_data_name=${DATA_NAME} \
  --model_name_or_path="./outputs/checkpoints/${LOAD_EXP_NAME}/latest_ckpt" \
  --source_prefix="question: " \
  --per_device_train_batch_size=12 \
  --per_device_eval_batch_size=32 \
  --gradient_accumulation_steps=5 \
  --learning_rate=8e-5 \
  --num_train_epochs=20 \
  --lr_scheduler_type="linear" \
  --num_warmup_steps=1000 \
  --output_dir="./outputs/${DATA_NAME}_checkpoints/${EXP_NAME}" \
  --prefix_length=2 \
  --d_key=1536 \
  --key_encoder_type="conv" \
  --select_positive_strategy="softmax_sample" \
  --faiss_efsearch=128 \
  --gen_weight=1 \
  --match_weight=1 \
  --key_reduce_method="avg" \
  --qas_to_retrieve_from="PAQ_L1" \
  --local_size=384 \
  --separate_task \
  --early_stop_patience=8 \
  --negatives_num_each_example=16 \
  --do_train \
  --add_topic \
  --add_persona \
  --not_share_encoder
