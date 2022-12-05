import faiss
import os
from utils.utils import update_CAT_config_from_args
import asyncio
import argparse
import torch
from transformers import T5Tokenizer, T5Config
from emat.t5 import T5WithKeyValueMemory
from emat.utils import load_jsonl
import logging
from embed_and_build_index import load_qas_to_embed
import time
from kilt_dataset import DialogDataset
from torch.nn.utils.rnn import pad_sequence

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
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Inference with faiss")
    parser.add_argument("--model_name_or_path", type=str, required=False,
                        default="./outputs/nq_checkpoints/KL=3;kdim=1536;VL=7;VN=10;cat_k_delay+v;t5-base;pos_from_top=128;/best_ckpt/")
    parser.add_argument("--f", choices=list(QA_KB_PATHS.keys()), default=f"debug")
    parser.add_argument("--add_nq_train", action="store_true")
    parser.add_argument("--add_nq_dev", action="store_true")
    parser.add_argument("--inference_batch_size", type=int, default=512)
    parser.add_argument("--load_dir", default=f"./data/embedding_and_faiss/debug_from_nq_ckpt")
    parser.add_argument("--inference_type", type=str, default="async", choices=["async", "serial", "t5"])
    parser.add_argument("--cat_layer", default=7, type=int)
    parser.add_argument("--test_task", default="wow", type=str, choices=["qa", "wow", "eli5"])
    parser.add_argument("--model_size", default="base", type=str, choices=["base", "large", "3B"])
    parser.add_argument("--faiss_path", default="", type=str, required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    logging.info("loading faiss index.")
    if args.faiss_path == "":
        faiss_path = os.path.join(args.load_dir, "key.sq8.hnsw.faiss")
    else:
        faiss_path = args.faiss_path
    key_faiss_index = faiss.read_index(faiss_path)
    logging.info("loaded faiss index.")

    logging.info("loading memory.")
    value_memory = torch.load(os.path.join(args.load_dir, "value_memory.pt"))
    key_memory = torch.load(os.path.join(args.load_dir, "key_memory.pt"))
    logging.info("loaded memory.")

    logging.info("loading data")
    qas_to_retrieve = load_qas_to_embed(args.qas_to_retrieve_from, args.add_nq_train, args.add_nq_dev)
    logging.info("loaded data")
    assert len(qas_to_retrieve) == value_memory.shape[0] == key_memory.shape[0]

    logging.info("loading model")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    if args.model_size == "3B":
        config = T5Config.from_pretrained(args.model_name_or_path)
        args.value_fusion_method = "cat_key_delay+v"
        args.num_values = 10
        args.prefix_length = 2
        args.key_encoder_type = "prefix"
        args.key_layer = 3
        args.value_layer = 7
        args.d_key = config.d_model * args.prefix_length
        args.use_two_prefix = False
        args.not_share_encoder = False
        update_CAT_config_from_args(config, args)
        model, load_info = T5WithKeyValueMemory.from_pretrained(args.model_name_or_path, config=config,
                                                                output_loading_info=True)
        logging.info("loaded T5-3B.")
    else:
        model, load_info = T5WithKeyValueMemory.from_pretrained(args.model_name_or_path, output_loading_info=True)
        model.eval()
    logging.info(f"model load info: {load_info}")

    if args.test_task == "qa":
        # test_data = load_jsonl("./data/annotated_datasets/NQ-open.test.jsonl")
        test_data = load_jsonl("./data/annotated_datasets/NQ-open.train-train.jsonl")[:512 * 40]
        logging.info(f"loaded {len(test_data)} test qas.")
    else:
        if args.test_task == "wow":
            dataset_kwargs = {
                "dataset_name": "wow_kilt",
                "max_source_length": 1024
            }
            test_data = load_jsonl("./data/annotated_datasets/wizard_of_wikipedia/wow-dev-kilt.jsonl")[:512]
            test_data = test_data * 10
            logging.info(f"loaded {len(test_data)} test history-response pairs.")
        else:
            dataset_kwargs = {
                "dataset_name": "eli5_kilt",
                "max_source_length": 384,
                "max_target_length": 1536
            }
            test_data = load_jsonl("./data/annotated_datasets/eli5/eli5-dev-kilt.jsonl")[:512]
            test_data = test_data * 10
            logging.info(f"loaded {len(test_data)} test long-form qas.")
        test_dataset = DialogDataset(test_data, tokenizer, qas_to_retrieve, max_utterances=13, **dataset_kwargs)
        test_data = test_dataset.data

    torch.cuda.empty_cache()

    model = model.cuda()
    if args.inference_type == "serial":
        serial_inference(model, tokenizer, test_data, args.inference_batch_size, key_faiss_index, value_memory,
                         key_memory, qas_to_retrieve, args.test_task)
    elif args.inference_type == "async":
        async_inference(model, tokenizer, test_data, args.inference_batch_size, key_faiss_index, value_memory,
                        key_memory, qas_to_retrieve, args.cat_layer, args.test_task)
    elif args.inference_type == "t5":
        t5_inference(model, tokenizer, test_data, args.inference_batch_size, key_faiss_index, value_memory,
                     key_memory, qas_to_retrieve, args.test_task)


def get_query_inputs(tokenizer, batch, device, test_task):
    if test_task == "qa":
        query_inputs = ["question: " + qa["question"] for qa in batch]
        query_inputs = tokenizer(query_inputs, max_length=1024,
                                 padding=True, truncation=True, return_tensors="pt")
        return query_inputs["input_ids"].to(device), query_inputs["attention_mask"].to(device)
    else:
        history_input_ids = [ex["input_ids"] for ex in batch]
        history_input_ids = pad_sequence(history_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        history_attention_mask = (history_input_ids != tokenizer.pad_token_id).long()
        return history_input_ids.to(device), history_attention_mask.to(device)


@torch.no_grad()
def serial_inference(model: T5WithKeyValueMemory, tokenizer, test_data, batch_size,
                     key_faiss_index, value_memory, not_reduced_key_memory, qas_to_retrieve, test_task):
    if test_task == "qa":
        gen_kwargs = {"num_beams": None, "max_length": 64}
    elif test_task == "wow":
        gen_kwargs = {"num_beams": None, "max_length": 28, "min_length": 28}
    else:
        gen_kwargs = {"num_beams": None, "max_length": 187, "min_length": 187}

    readout_top_k = model.config.num_values
    key_reduce_method = "avg"
    value_fusion_method = model.config.value_fusion_method

    time_log = []
    query_log = []
    for start_idx in range(0, len(test_data), batch_size):
        start_time = time.perf_counter()

        batch = test_data[start_idx: start_idx + batch_size]
        input_ids, attention_mask = get_query_inputs(tokenizer, batch, model.device, test_task)

        encoder_outputs = model.encoder.forward_with_faiss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            readout_top_k=readout_top_k,
            key_reduce_method=key_reduce_method,
            value_fusion_method=value_fusion_method,
            key_faiss_index=key_faiss_index,
            value_memory=value_memory,
            not_reduced_key_memory=not_reduced_key_memory
        )
        generated_tokens = model.generate(
            encoder_outputs=encoder_outputs,
            encoder_outputs_are_key_or_value=False,
            decoder_only_attend_on_prefix=False,
            attention_mask=attention_mask,
            value_fusion_method=value_fusion_method,
            **gen_kwargs,
        )
        cur_cost = time.perf_counter() - start_time
        time_log.append(cur_cost)
        query_log.append(len(batch))
        logging.info(f" {len(batch)} queries / {cur_cost} seconds")

    time_log = time_log[2:-1]
    query_log = query_log[2:-1]
    query_num = sum(query_log)
    total_time = sum(time_log)
    logging.info(f"average speed: {query_num} queries / {total_time} seconds = "
                 f"{query_num / total_time} queries per second")


@torch.no_grad()
def async_inference(model: T5WithKeyValueMemory, tokenizer, test_data, batch_size,
                    key_faiss_index, value_memory, not_reduced_key_memory, qas_to_retrieve, cat_layer, test_task):
    if test_task == "qa":
        gen_kwargs = {"num_beams": None, "max_length": 64}
    elif test_task == "wow":
        gen_kwargs = {"num_beams": None, "max_length": 28, "min_length": 28}
    else:
        gen_kwargs = {"num_beams": None, "max_length": 187, "min_length": 187}

    readout_top_k = model.config.num_values
    key_reduce_method = "avg"
    # value_fusion_method = "async_cat_k+v"
    model.encoder.key_layer = 3
    model.encoder.cat_layer = cat_layer
    model.encoder.value_layer = 10
    if model.encoder.cat_layer == model.encoder.value_layer:
        value_fusion_method = "async_cat_k+v"
    else:
        value_fusion_method = "async_cat_k_delay+v"

    logging.info(f"cat_layer: {cat_layer}")

    time_log = []
    query_log = []
    for start_idx in range(0, len(test_data), batch_size):
        start_time = time.perf_counter()

        batch = test_data[start_idx: start_idx + batch_size]
        input_ids, attention_mask = get_query_inputs(tokenizer, batch, model.device, test_task)

        encoder_outputs = asyncio.run(
            model.encoder.forward_with_async_faiss(
                input_ids, attention_mask, True, readout_top_k, key_reduce_method, value_fusion_method,
                key_faiss_index, value_memory, not_reduced_key_memory
            )
        )
        generated_tokens = model.generate(
            encoder_outputs=encoder_outputs,
            encoder_outputs_are_key_or_value=False,
            decoder_only_attend_on_prefix=False,
            attention_mask=attention_mask,
            value_fusion_method=value_fusion_method,
            **gen_kwargs,
        )
        cur_cost = time.perf_counter() - start_time
        time_log.append(cur_cost)
        query_log.append(len(batch))
        logging.info(f" {len(batch)} queries / {cur_cost} seconds")

    time_log = time_log[2:-1]
    query_log = query_log[2:-1]
    query_num = sum(query_log)
    total_time = sum(time_log)
    logging.info(f"cat_layer: {cat_layer}")
    logging.info(f"average speed: {query_num} queries / {total_time} seconds = "
                 f"{query_num / total_time} queries per second")


# ELI5 --inference_batch_size=128
# WoW --inference_batch_size=256

@torch.no_grad()
def t5_inference(model: T5WithKeyValueMemory, tokenizer, test_data, batch_size,
                 key_faiss_index, value_memory, not_reduced_key_memory, qas_to_retrieve, test_task):
    if test_task == "qa":
        gen_kwargs = {"num_beams": None, "max_length": 16}
    elif test_task == "wow":
        gen_kwargs = {"num_beams": None, "max_length": 28, "min_length": 28}
    else:
        gen_kwargs = {"num_beams": None, "max_length": 187, "min_length": 187}

    readout_top_k = model.config.num_values
    key_reduce_method = "avg"
    value_fusion_method = model.config.value_fusion_method
    model.key_layer = 1000
    model.value_layer = 1000
    model.cat_layer = 1000
    model.encoder.key_layer = 1000
    model.encoder.value_layer = 1000
    model.encoder.cat_layer = 1000

    time_log = []
    query_log = []
    for start_idx in range(0, len(test_data), batch_size):
        start_time = time.perf_counter()

        batch = test_data[start_idx: start_idx + batch_size]
        input_ids, attention_mask = get_query_inputs(tokenizer, batch, model.device, test_task)

        encoder_outputs = model.encoder.forward_with_faiss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            readout_top_k=readout_top_k,
            key_reduce_method=key_reduce_method,
            value_fusion_method=value_fusion_method,
            key_faiss_index=key_faiss_index,
            value_memory=value_memory,
            not_reduced_key_memory=not_reduced_key_memory
        )
        generated_tokens = model.generate(
            encoder_outputs=encoder_outputs,
            encoder_outputs_are_key_or_value=False,
            decoder_only_attend_on_prefix=False,
            attention_mask=attention_mask,
            value_fusion_method=value_fusion_method,
            **gen_kwargs,
        )
        cur_cost = time.perf_counter() - start_time
        time_log.append(cur_cost)
        query_log.append(len(batch))
        logging.info(f" {len(batch)} queries / {cur_cost} seconds")

    time_log = time_log[2:-1]
    query_log = query_log[2:-1]
    query_num = sum(query_log)
    total_time = sum(time_log)
    logging.info(f"average speed: {query_num} queries / {total_time} seconds = "
                 f"{query_num / total_time} queries per second")


if __name__ == '__main__':
    main()
