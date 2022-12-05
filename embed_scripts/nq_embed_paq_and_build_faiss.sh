#!/bin/bash -l

set -e
set -u

DST_DIR="case-augmented-transformer-master"
cd ${DST_DIR}

DEVICE="0"

PAQ_TYPE="PAQ_L1"
SAVE_DIR="./data/embedding_and_faiss/${PAQ_TYPE}_from_nq_ckpt"
NQ_MODEL_PATH="" # --

CUDA_VISIBLE_DEVICES=${DEVICE} python embed_and_build_index.py \
  --model_name_or_path=${NQ_MODEL_PATH} \
  --qas_to_retrieve_from=${PAQ_TYPE} \
  --embed_batch_size=2048 \
  --save_dir=${SAVE_DIR} \
  --add_nq_train \
  --add_nq_dev

CUDA_VISIBLE_DEVICES=${DEVICE} python emat/retriever/build_index.py \
  --embeddings_dir="./data/embedding_and_faiss/PAQ_from_nq_ckpt/embedding_index.pt" \
  --output_path="${SAVE_DIR}/key.sq8hnsw.80n80efc.faiss" \
  --hnsw \
  --store_n 80 \
  --ef_construction 80 \
  --ef_search 32 \
  --SQ8 \
  --indexing_batch_size=1 \
  --verbose
