#!/bin/bash -l

set -e
set -u


DST_DIR="case-augmented-transformer-master" # change to your project root
cd ${DST_DIR}

LOAD_EXP_NAME="KL=3;kdim=1536;VL=7;VN=10;cat_k_delay+v;t5-base;" # load pre-trained model
EXP_NAME="base;KL=3;VL=7;VN=10;lr=5e-5;" # set experiment name
DATA_NAME="nq" # datasets: ["nq", "tq", "wq"]

DEVICE="0"

# Train nq-EMAT-FKSV
# use --kvm_fp16 if GPU OOM

CUDA_VISIBLE_DEVICES=${DEVICE} python qa_main.py \
  --project_name="${DATA_NAME^^}-CAT" \
  --exp_name=${EXP_NAME} \
  --query_batch_size=256 \
  --build_mem_batch_size=12000 \
  --batch_local_positive_num=5 \
  --pos_from_top=128 \
  --do_eval \
  --kvm_seg_n=2 \
  --values_with_order \
  --value_layer=7 \
  --value_fusion_method="cat_k_delay+v" \
  --num_values=10 \
  --qa_data_name=${DATA_NAME} \
  --model_name_or_path="./outputs/checkpoints/${LOAD_EXP_NAME}/latest_ckpt" \
  --source_prefix="question: " \
  --per_device_train_batch_size=64 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --num_train_epochs=30 \
  --lr_scheduler_type="linear" \
  --num_warmup_steps=1000 \
  --output_dir="./outputs/nq_checkpoints/${EXP_NAME}" \
  --prefix_length=2 \
  --d_key=1536 \
  --key_layer=3 \
  --key_encoder_type="conv" \
  --select_positive_strategy="softmax_sample" \
  --faiss_efsearch=128 \
  --gen_weight=1 \
  --match_weight=1 \
  --key_reduce_method="avg" \
  --qas_to_retrieve_from="PAQ_L1" \
  --local_size=384 \
  --update_kv_embeds \
  --update_local_target_each_batch \
  --update_local_qas \
  --separate_task \
  --value_ae_target="ans" \
  --key_ae_target="question" \
  --repaq_supervision_epoch=-1 \
  --early_stop_patience=8 \
  --negatives_num_each_example=32 \
  --do_train
