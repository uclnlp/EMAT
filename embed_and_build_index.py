import os
import pickle
import argparse
import torch
from transformers import T5Tokenizer
import copy
from emat.t5 import T5WithKeyValueMemory
from emat.utils import load_jsonl
import logging
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.utils import get_key_value_encoder_inputs, reduce_query_or_key_embeds

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

QA_KB_PATHS = {
    "PAQ_L1": "./tmp/PAQ_L1_pickl_file.pkl",
    "PAQ": "./tmp/PAQ_full.pkl",
    "TAQ_TRAIN_NQ_TRAIN_PAQ": "./data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl",
    "debug": "./tmp/PAQ_L1_small.pkl"
}


def get_args():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Embed and build FAISS")
    parser.add_argument("--model_name_or_path", type=str, required=False,
                        default="./outputs/nq_checkpoints/KL=3;kdim=1536;VL=7;VN=10;cat_k_delay+v;t5-base;pos_from_top=128;/best_ckpt/")
    parser.add_argument("--qas_to_retrieve_from", choices=list(QA_KB_PATHS.keys()), default=f"debug")
    parser.add_argument("--add_nq_train", action="store_true")
    parser.add_argument("--add_nq_dev", action="store_true")
    parser.add_argument("--embed_batch_size", type=int, default=512)
    parser.add_argument("--save_dir", default=f"./data/embedding_and_faiss/debug_from_nq_ckpt")

    args = parser.parse_args()
    return args


def load_qas_to_embed(qas_to_retrieve_from, add_nq_train, add_nq_dev):
    logging.info("loading qas to retrieve")
    qas_to_retrieve_fp = QA_KB_PATHS[qas_to_retrieve_from]
    logging.info(f"loading qas from {qas_to_retrieve_fp}")
    if qas_to_retrieve_fp.endswith("pkl"):
        qas_to_embed = pickle.load(open(qas_to_retrieve_fp, 'rb'))
    elif qas_to_retrieve_fp.endswith("jsonl"):
        qas_to_embed = load_jsonl(qas_to_retrieve_fp)
    else:
        raise ValueError(f"{qas_to_retrieve_fp}")
    logging.info(f"load {len(qas_to_embed)} qas from PAQ.")

    # if qas_to_retrieve_from == "debug":
    #     qas_to_retrieve = qas_to_retrieve[:10000]

    if add_nq_train:
        logging.info("add nq-train qas.")
        qas_to_embed = qas_to_embed + load_jsonl("./data/annotated_datasets/NQ-open.train-train.jsonl")
    if add_nq_dev:
        logging.info("add nq-dev qas.")
        qas_to_embed = qas_to_embed + load_jsonl("./data/annotated_datasets/NQ-open.train-dev.jsonl")

    logging.info(f"load {len(qas_to_embed)} qas totally.")

    return qas_to_embed


@torch.no_grad()
def embed_key_value(model, tokenizer, data_to_embed, embed_batch_size, save_dir,
                    use_fp16_model=True, key_reduce_method="avg", max_source_length=1024, prefix="question: "):
    if use_fp16_model:
        model = model.half()
    logging.info("")

    # model.eval()
    # key_memory, value_memory, not_reduced_key_memory = build_memory(
    #     model, tokenizer, embed_key=True, embed_value=True, prefix=prefix, embed_as_fp16=True,
    #     key_reduce_method=key_reduce_method, return_memory=True, dump_memory=False,
    #     data_to_embed=data_to_embed, max_source_length=max_source_length, padding=True,
    #     batch_size=embed_batch_size, separate_task=True, return_not_reduced_key=True,
    #
    #     reused_key_memory=reused_key_memory,
    #     reused_value_memory=reused_value_memory,
    #     reused_not_reduced_key_memory=reused_not_reduced_key_memory
    # )
    # return key_memory, value_memory, not_reduced_key_memory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        logging.warning(f"{save_dir} is exists. re-write contents warning.")

    def collate_fn(examples):
        model_inputs = get_key_value_encoder_inputs(examples, True, tokenizer, max_source_length,
                                                    prefix=prefix, only_return_key_inputs=False)
        return model_inputs

    data_to_embed_dataloader = DataLoader(data_to_embed, batch_size=embed_batch_size,
                                          num_workers=4, collate_fn=collate_fn)
    import gc

    def save_embedding_index():
        reused_key_memory = torch.zeros((len(data_to_embed), model.model_dim), device="cpu", dtype=torch.float16)
        key_cnt = 0
        for batch in tqdm(data_to_embed_dataloader):
            batch_keys = list(batch.keys())
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embed_dict = model.wrapped_embed_kv(separate_task=True, **batch,
                                                compute_key=True, compute_value=False)
            for k in batch_keys:
                del batch[k]
            key_embeds = embed_dict.get("normed_key_embeds")
            key_embeds = reduce_query_or_key_embeds(key_embeds, key_reduce_method)
            cur_key_num = key_embeds.shape[0]
            key_embeds = key_embeds.half().cpu()
            reused_key_memory[key_cnt: key_cnt + cur_key_num] = copy.deepcopy(key_embeds)
            del key_embeds
        torch.save(reused_key_memory, os.path.join(save_dir, "embedding_index.pt"))
        logging.info("embedding index saved.")

    def save_value_memory():
        reused_value_memory = torch.zeros((len(data_to_embed), 2, model.model_dim), device="cpu", dtype=torch.float16)
        value_cnt = 0
        for batch in tqdm(data_to_embed_dataloader):
            batch_keys = list(batch.keys())
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embed_dict = model.wrapped_embed_kv(separate_task=True, **batch,
                                                compute_key=False, compute_value=True)
            for k in batch_keys:
                del batch[k]
            value_embeds = embed_dict.get("value_embeds")
            cur_value_num = value_embeds.shape[0]
            value_embeds = value_embeds.half().cpu()
            reused_value_memory[value_cnt: value_cnt + cur_value_num] = copy.deepcopy(value_embeds)
            del value_embeds
        torch.save(reused_value_memory, os.path.join(save_dir, "value_memory.pt"))
        logging.info("value memory saved.")

    def save_key_memory():
        reused_not_reduced_key_memory = torch.zeros((len(data_to_embed), 2, model.model_dim),
                                                    device="cpu", dtype=torch.float16)
        nr_key_cnt = 0
        for batch in tqdm(data_to_embed_dataloader):
            batch_keys = list(batch.keys())
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embed_dict = model.wrapped_embed_kv(separate_task=True, **batch,
                                                compute_key=True, compute_value=False)
            for k in batch_keys:
                del batch[k]
            not_normed_key_embeds = embed_dict["key_embeds"]
            cur_key_num = not_normed_key_embeds.shape[0]
            not_normed_key_embeds = not_normed_key_embeds.half().cpu()
            reused_not_reduced_key_memory[nr_key_cnt: nr_key_cnt + cur_key_num] = copy.deepcopy(not_normed_key_embeds)
            del not_normed_key_embeds
        torch.save(reused_not_reduced_key_memory, os.path.join(save_dir, "key_memory.pt"))
        logging.info("key memory saved.")

    save_embedding_index()
    gc.collect()
    save_value_memory()
    gc.collect()
    save_key_memory()
    gc.collect()


def main():
    args = get_args()
    logging.info("loading model")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model, load_info = T5WithKeyValueMemory.from_pretrained(args.model_name_or_path, output_loading_info=True)
    model = model.cuda()
    model.eval()
    logging.info(f"model load info: {load_info}")

    logging.info("loading data")
    data_to_embed = load_qas_to_embed(args.qas_to_retrieve_from, args.add_nq_train, args.add_nq_dev)

    logging.info("embedding")
    embed_key_value(model, tokenizer, data_to_embed, args.embed_batch_size, args.save_dir)
    # key_memory is normed and reduced
    # value_memory is not normed
    # not_reduced_key_memory is not normed and not reduced
    logging.info("embedding saved.")


if __name__ == '__main__':
    main()
