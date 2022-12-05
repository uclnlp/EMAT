#!/bin/bash -l

set -e
set -u

DST_DIR="case-augmented-transformer-master" # change to your project root
cd ${DST_DIR}

LOAD_EXP_NAME="t5-base" # t5-base dir
EXP_NAME="KL=3;kdim=1536;CL=10;VL=11;VN=10;async_cat_k+v;t5-base;"
DATA_NAME="PAQ-L1-Pretrain"
DEVICE="0"

echo ${DEVICE}

CUDA_VISIBLE_DEVICES=${DEVICE} python kvm_pretrain.py \
  --key_layer=3 \
  --cat_layer=10 \
  --value_layer=11 \
  --value_fusion_method="async_cat_k_delay+v" \
  --project_name="CAT" \
  --pretrain_multi_values \
  --exp_name="${EXP_NAME}" \
  --pretrain_data_name=${DATA_NAME} \
  --num_values=10 \
  --key_reduce_method="avg" \
  --model_name_or_path="${LOAD_EXP_NAME}" \
  --source_prefix="question: " \
  --per_device_train_batch_size=128 \
  --per_device_eval_batch_size=256 \
  --gradient_accumulation_steps=2 \
  --preprocessing_num_workers=10 \
  --learning_rate=5e-5 \
  --num_train_epochs=5 \
  --lr_scheduler_type="constant" \
  --num_warmup_steps=5000 \
  --output_dir="./outputs/checkpoints/${EXP_NAME}" \
  --prefix_length=2 \
  --d_key=1536 \
  --key_encoder_type="conv" \
  --seed=42 \
  --gen_weight=1.0 \
  --key_ae_weight=0.5 \
  --value_ae_weight=0.5 \
  --value_with_self_prop=0.1 \
  --key_ae_target="question" \
  --value_ae_target="ans" \
  --do_train \
  --separate_task \
  --train_key \
  --train_value \
  --do_eval
