# EMAT: An Efficient Memory-Augmented Transformer for Knowledge-Intensive NLP Tasks

## Installation and Setup

```shell
# create a conda environment
conda create -n emat -y python=3.8 && conda activate emat

# install pytorch
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # GPU
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1  # CPU

# install transformers
pip install transformers==4.14.1

# install faiss
pip install faiss-gpu==1.7.1.post3  # GPU
pip install faiss-cpu==1.7.1.post3  # CPU

# install dependencies
pip install -r requirements.txt

# install this package for development
pip install -e .
```

## Download datasets

[//]: # (NaturalQuestion, WebQuestion, TriviaQA, WoW_KILT, ELI5_KILT data:)

link: https://pan.baidu.com/s/1MwPzVLqZqslqCpWAtPVZ-Q

code: tynj

Download PAQ data from: https://github.com/facebookresearch/PAQ


## Run Interactive Script

Before running the following scripts, embeddings of key-value memory, index and PAQ should be prepared. 
See [Start](#Start) to build your key-value memory and index.


NQ: use torch-embedding as retrieval index:
```bash
python demo.py \
  --model_name_or_path="./EMAT_ckpt/FKSV-NQ" \
  --qas_to_retrieve_from="./data/PAQ_L1" \
  --test_task="nq" \
  --task_train_data="./annotated_datasets/NQ-open.train-train.jsonl" \
  --task_dev_data="./annotated_datasets/NQ-open.train-dev.jsonl" \
  --embedding_index="./embedding_and_faiss/PAQ_L1_from_nq_ckpt/embedding_index.pt"
  --key_memory_path="./embedding_and_faiss/PAQ_L1_from_nq_ckpt/key_memory.pt" \
  --value_memory_path="./embedding_and_faiss/PAQ_L1_from_nq_ckpt/value_memory.pt"
```

NQ: use faiss as retrieval index:
```bash
python demo.py \
  --model_name_or_path="./EMAT_ckpt/FKSV-NQ" \
  --qas_to_retrieve_from="./data/PAQ_L1" \
  --test_task="nq" \
  --task_train_data="./annotated_datasets/NQ-open.train-train.jsonl" \
  --task_dev_data="./annotated_datasets/NQ-open.train-dev.jsonl" \
  --use_faiss \
  --faiss_index_path="./embedding_and_faiss/PAQ_L1_from_nq_ckpt/key.sq8hnsw.80n80efc.faiss" \
  --key_memory_path="./embedding_and_faiss/PAQ_L1_from_nq_ckpt/key_memory.pt" \
  --value_memory_path="./embedding_and_faiss/PAQ_L1_from_nq_ckpt/value_memory.pt"
```

Use SKSV model with faiss parallely search:
```bash
python demo.py \
  --model_name_or_path="./EMAT_ckpt/SKSV-NQ" \
  --qas_to_retrieve_from="./data/PAQ_L1" \
  --test_task="nq" \
  --task_train_data="./annotated_datasets/NQ-open.train-train.jsonl" \
  --task_dev_data="./annotated_datasets/NQ-open.train-dev.jsonl" \
  --use_faiss \
  --faiss_index_path="./embedding_and_faiss/PAQ_L1_from_nq_SKSV_ckpt/key.sq8hnsw.80n80efc.faiss" \
  --key_memory_path="./embedding_and_faiss/PAQ_L1_from_nq_SKSV_ckpt/key_memory.pt" \
  --value_memory_path="./embedding_and_faiss/PAQ_L1_from_nq_SKSV_ckpt/value_memory.pt"
```

Run Wizard-of-Wikipedia Dialogue:
```bash
python demo.py \
  --model_name_or_path="./EMAT_ckpt/FKSV-WQ/" \
  --qas_to_retrieve_from="./tmp/PAQ_L1.pkl" \
  --test_task="wow_kilt" \
  --embedding_index_path="./embedding_and_faiss/debug_from_wow_ckpt/embedding_index.pt" \
  --key_memory_path="./embedding_and_faiss/PAQ_L1_from_wow_ckpt/key_memory.pt" \
  --value_memory_path="./embedding_and_faiss/PAQ_L1_from_wow_ckpt/value_memory.pt" \
  --inference_data_path="./annotated_datasets/wizard_of_wikipedia/wow-test_without_answers-kilt.jsonl.txt"
```

## Start
<span id="Start"></span>

### 1. Pre-training

Pre-train EMAT-FKSV: `bash pretrain_scripts/pretrain_emat.sh`

Pre-train EMAT-SKSV: `bash pretrain_scripts/pretrain_sksv_emat.sh`

### 2. Fine-tune:

Fine-tune on NQ: `bash scripts/nq_train_with_paql1.sh`

Fine-tune on TQ: `bash scripts/tq_train_with_paql1.sh`

Fine-tune on WQ: `bash scripts/wq_train_with_paql1.sh`

Fine-tune on WoW : `bash kilt_scripts/wow_train.sh`

Fine-tune on ELI5: `bash kilt_scripts/eli5_train.sh`


### 3. Evaluation:

Evaluate NQ/TQ/WQ: `bash scripts/nq_eval.sh`, switch ``DATA_NAME`` to evaluate different dataset.

Evaluate WoW/ELI5: `bash kilt_scirpts/eval_kilt.sh`. You can upload the output prediction file to http://kiltbenchmark.com/ to get evaluation results.

### 4. Embed PAQ using fine-tuned NQ model and build FAISS index:
```bash
bash embed_scripts/nq_embed_paq_and_build_faiss.sh
```

### 5. Inference Speed
Test inference speed on ```inference_with_faiss.py```

