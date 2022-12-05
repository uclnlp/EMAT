import faiss
import asyncio
import argparse
import torch
from transformers import T5Tokenizer, T5Config
from emat.t5 import T5WithKeyValueMemory
from emat.utils import load_jsonl
import logging
from kilt_dataset import DialogDataset
import pickle
import random
from kilt_trainer import kilt_generate

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_args():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Inference with faiss")
    parser.add_argument("--model_name_or_path", type=str, required=False,
                        default="./outputs/nq_checkpoints/KL=3;kdim=1536;VL=7;VN=10;cat_k_delay+v;t5-base;pos_from_top=128;/best_ckpt/")
    parser.add_argument("--qas_to_retrieve_from", default="./tmp/PAQ_L1_pickl_file.pkl")
    parser.add_argument("--test_task", default="nq", type=str, choices=["nq", "wq", "tq", "wow_kilt"])
    parser.add_argument("--task_train_data", default=None, required=False, type=str)
    parser.add_argument("--task_dev_data", default=None, required=False, type=str)
    parser.add_argument("--use_faiss", action="store_true", help="default -- use torch embedding")
    parser.add_argument("--faiss_index_path", default=None, type=str, required=False)
    parser.add_argument("--embedding_index_path", default=None, type=str, required=False)
    parser.add_argument("--key_memory_path", required=True)
    parser.add_argument("--value_memory_path", required=True)
    parser.add_argument("--inference_type", type=str, default="serial", choices=["parallel", "serial"])
    parser.add_argument("--inference_data_path", type=str, default=None, required=False)
    parser.add_argument("--inference_batch_size", type=int, default=512)
    args = parser.parse_args()

    if args.use_faiss:
        assert args.faiss_index_path is not None
    else:
        assert args.embedding_index_path is not None

    return args


def main():
    args = get_args()

    # load model
    logging.info(f"loading model from {args.model_name_or_path}")
    model, load_info = T5WithKeyValueMemory.from_pretrained(args.model_name_or_path, output_loading_info=True)
    model.eval()
    model = model.cuda()
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    logging.info(f"model load info: {load_info}")
    # check
    if getattr(model, "cat_layer", None) == model.encoder.key_layer:
        assert args.inference_type != "parallel", "parallel can not used in cat_layer == key_layer"

    # load index and key-value memory
    faiss_index, embedding_index = None, None
    if args.use_faiss:
        logging.info(f"loading index from {args.faiss_index_path}")
        faiss_index = faiss.read_index(args.faiss_index_path)
        logging.info("loaded faiss index.")
    else:
        logging.info(f"loading index from {args.embedding_index_path}")
        embedding_index = torch.load(args.embedding_index_path)
        logging.info("loaded embedding index.")
    value_memory = torch.load(args.value_memory_path)
    key_memory = torch.load(args.key_memory_path)

    # load QAs to retrieve
    logging.info(f"loading PAQ from {args.qas_to_retrieve_from}")
    if args.qas_to_retrieve_from.endswith("pkl"):
        qas_to_retrieve = pickle.load(open(args.qas_to_retrieve_from, 'rb'))
    else:  # jsonl
        qas_to_retrieve = load_jsonl(args.qas_to_retrieve_from)
    logging.info("loaded PAQ")
    if args.test_task in ["nq", "wq", "tq"]:
        if args.task_train_data is not None:
            qas_to_retrieve = qas_to_retrieve + load_jsonl(args.task_train_data)
        if args.task_dev_data is not None:
            qas_to_retrieve = qas_to_retrieve + load_jsonl(args.task_dev_data)
    assert len(qas_to_retrieve) == value_memory.shape[0] == key_memory.shape[0]
    logging.info(f"numer of QAs to retrieve: {len(qas_to_retrieve)}")

    if args.test_task in ["nq", 'wq', 'tq']:
        gen_kwargs = {"num_beams": None, "max_length": 64}
    else:
        gen_kwargs = {"max_length": 1024, "num_beams": 8, "do_sample": True, "top_k": 64, "min_length": 8}

    print("input ``ctrl + c`` to exit the program.")
    if args.test_task in ["nq", 'wq', 'tq']:
        while True:
            question = input("Question: ")
            batch = [{"question": question.strip()}]
            ans, retrieved_qa = inference_qa(model, tokenizer, key_memory, value_memory, embedding_index,
                                             faiss_index, qas_to_retrieve, args.inference_type, batch, gen_kwargs)
            print(f"Answer: {ans[0]}")
            print(f"retrieved QAs: ")
            for qa in retrieved_qa[0]:
                print(qa)

    elif args.test_task == 'wow_kilt':
        print("input '-1' to exit current dialogue")
        dataset_kwargs = {"dataset_name": "wow_kilt", "max_source_length": 128}
        inference_data = load_jsonl(args.inference_data_path)
        while True:
            cur_dialogue = random.sample(inference_data, 1)[0]
            utterances = cur_dialogue["input"].split("\n")[:-1]
            for idx, u in enumerate(utterances):
                spk = "A" if idx % 2 == 0 else "B"
                print(f"{spk}: {u}")
            while True:
                spk = "A" if len(utterances) % 2 == 0 else "B"
                utterance = input(f"{spk}: ")
                if utterance == "-1":
                    break
                utterances.append(utterance)
                cur_dialogue["input"] = "\n".join(utterances)
                dataset = DialogDataset([cur_dialogue], tokenizer, qas_to_retrieve, **dataset_kwargs)
                retrieved_qa, response = kilt_generate(
                    model, tokenizer, embedding_index, key_memory, value_memory, dataset,
                    qas_to_retrieve, args.inference_batch_size, gen_kwargs
                )
                spk = "A" if len(utterances) % 2 == 0 else "B"
                print(f"{spk}: {response[0]}")
                utterances.append(response[0])
            print("")


@torch.no_grad()
def inference_qa(model, tokenizer, key_memory, value_memory, embedding_index, faiss_index,
                 qas_to_retrieve, inference_type, batch, gen_kwargs):
    inputs = ["question: " + qa["question"] for qa in batch]
    inputs = tokenizer(inputs, max_length=1024, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    attention_mask = inputs["attention_mask"].to('cuda')
    if embedding_index is not None:
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            readout_top_k=model.encoder.num_values,
            key_reduce_method="avg",
            value_fusion_method=model.encoder.value_fusion_method,
            embedding_index=embedding_index,
            key_memory=key_memory,
            value_memory=value_memory
        )
    else:
        if inference_type == "serial":
            encoder_outputs = model.encoder.forward_with_faiss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                readout_top_k=model.encoder.num_values,
                key_reduce_method="avg",
                value_fusion_method=model.encoder.value_fusion_method,
                key_faiss_index=faiss_index,
                value_memory=value_memory,
                not_reduced_key_memory=key_memory
            )
        else:
            encoder_outputs = asyncio.run(
                model.encoder.forward_with_async_faiss(
                    input_ids, attention_mask, True, model.encoder.num_values, "avg",
                    model.encoder.value_fusion_method, faiss_index, value_memory, key_memory
                )
            )
    generated_tokens = model.generate(
        encoder_outputs=encoder_outputs,
        encoder_outputs_are_key_or_value=False,
        decoder_only_attend_on_prefix=False,
        attention_mask=attention_mask,
        value_fusion_method=model.encoder.value_fusion_method,
        **gen_kwargs,
    )

    readout_qas = [[qas_to_retrieve[idx] for idx in indices] for indices in encoder_outputs.readout_indices]
    decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    decoded_tokens = [ans.strip() for ans in decoded_tokens]
    return decoded_tokens, readout_qas


if __name__ == '__main__':
    main()
