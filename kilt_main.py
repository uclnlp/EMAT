import json
import os
import pickle
from transformers import T5Tokenizer
from emat.utils import load_jsonl
from kilt_dataset import DialogDataset
from kilt_trainer import DialogTrainer
from utils.utils import CATArgs
import logging
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

DATA_PATHS = {
    "wow": {
        "train": "./data/annotated_datasets/wizard_of_wikipedia/train.json",
        "validation": "./data/annotated_datasets/wizard_of_wikipedia/valid_random_split.json",
        "test": "./data/annotated_datasets/wizard_of_wikipedia/test_random_split.json",
    },
    "wow_unseen": {
        "train": "./data/annotated_datasets/wizard_of_wikipedia/train.json",
        "validation": "./data/annotated_datasets/wizard_of_wikipedia/valid_topic_split.json",
        "test": "./data/annotated_datasets/wizard_of_wikipedia/test_topic_split.json",
    },
    "wow_kilt": {
        "train": "./data/annotated_datasets/wizard_of_wikipedia/wow-train-kilt.jsonl",
        "validation": "./data/annotated_datasets/wizard_of_wikipedia/wow-dev-kilt.jsonl",
        "test": "./data/annotated_datasets/wizard_of_wikipedia/wow-test_without_answers-kilt.jsonl.txt",
    },
    "eli5_kilt": {
        "train": "./data/annotated_datasets/eli5/eli5-train-kilt.jsonl",
        "validation": "./data/annotated_datasets/eli5/eli5-dev-kilt.jsonl",
        "test": "./data/annotated_datasets/eli5/eli5-test_without_answers-kilt.jsonl",
    }
}
QA_KB_PATHS = {
    "PAQ_L1": "./tmp/PAQ_L1_pickl_file.pkl",
    "PAQ": "./tmp/PAQ_full.pkl",
    "TAQ_TRAIN_NQ_TRAIN_PAQ": "./data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl",
}
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def load_dataset(args):
    assert args.qa_data_name in DATA_PATHS.keys(), f"available dataset names: {DATA_PATHS.keys()}"

    logging.info("loading normed answer of qas to retrieve")
    if "PAQ" == args.qas_to_retrieve_from:
        normed_answer_of_qas_to_ret = pickle.load(open("./tmp/PAQ_only_normalized_answer.pkl", 'rb'))
    else:
        normed_answer_of_qas_to_ret = json.load(open("./tmp/PAQL1_only_normalized_answer.json", 'r'))

    logging.info("loading qas to retrieve")
    if "debug" in args.exp_name.lower() or "full-paq-test" in args.exp_name.lower():
        if not os.path.exists("./tmp/PAQ_L1_small.pkl"):
            qas_to_retrieve = pickle.load(open("./tmp/PAQ_L1_pickl_file.pkl", 'rb'))
            qas_to_retrieve = qas_to_retrieve[:len(qas_to_retrieve) // 14]
            pickle.dump(qas_to_retrieve, open("./tmp/PAQ_L1_small.pkl", 'wb'))
        else:
            qas_to_retrieve = pickle.load(open("./tmp/PAQ_L1_small.pkl", 'rb'))
    else:
        qas_to_retrieve_fp = QA_KB_PATHS[args.qas_to_retrieve_from]
        logging.info(f"loading qas from {qas_to_retrieve_fp}")
        if qas_to_retrieve_fp.endswith("pkl"):
            qas_to_retrieve = pickle.load(open(qas_to_retrieve_fp, 'rb'))
        elif qas_to_retrieve_fp.endswith("jsonl"):
            qas_to_retrieve = load_jsonl(qas_to_retrieve_fp)
        else:
            raise ValueError(f"{qas_to_retrieve_fp}")

    if "debug" in args.exp_name.lower():
        qas_to_retrieve = qas_to_retrieve[:5000]
        normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret[:len(qas_to_retrieve)]

    if args.qas_to_retrieve_from == "PAQ" and args.PAQ_size is not None:
        qas_to_retrieve = qas_to_retrieve[:args.PAQ_size]
        normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret[:args.PAQ_size]
        assert len(qas_to_retrieve) == args.PAQ_size
        logging.info(f"select {args.PAQ_size}-size PAQ.")

    assert len(normed_answer_of_qas_to_ret) == len(qas_to_retrieve)
    loaded_data = {
        "qas_to_retrieve": qas_to_retrieve,
        "normed_answer_of_qas_to_ret": normed_answer_of_qas_to_ret
    }

    return loaded_data


def main():
    cat_args = CATArgs("dialog_cat")
    args = cat_args.parse_args()
    data_paths = DATA_PATHS[args.qa_data_name]
    logging.info("load datasets")
    if "kilt" in args.qa_data_name:
        train_data = load_jsonl(data_paths["train"])
        dev_data = load_jsonl(data_paths["validation"])
        test_data = load_jsonl(data_paths["test"])
    else:
        train_data = json.load(open(data_paths["train"], 'r'))
        dev_data = json.load(open(data_paths["validation"], 'r'))
        test_data = json.load(open(data_paths["test"], 'r'))

    loaded_data = load_dataset(args)
    logging.info("data loaded.")
    qas_to_retrieve = loaded_data["qas_to_retrieve"]
    normed_answer_of_qas_to_ret = loaded_data["normed_answer_of_qas_to_ret"]

    if "debug" in args.exp_name.lower():
        train_data = train_data[:50]
        dev_data = dev_data[:10]
        test_data = test_data[:10]
        qas_to_retrieve = qas_to_retrieve[:10000]
        normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret[:10000]

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    if args.qa_data_name != "eli5_kilt":
        dataset_kwargs = {
            "dataset_name": args.qa_data_name,
            "args": args,
            "normed_answer_of_qas_to_ret": normed_answer_of_qas_to_ret,
            "add_persona": args.add_persona,
            "add_topic": args.add_topic,
            "max_source_length": 1024
        }
    else:
        assert args.qa_data_name == "eli5_kilt"
        dataset_kwargs = {
            "dataset_name": args.qa_data_name,
            "args": args,
            "normed_answer_of_qas_to_ret": normed_answer_of_qas_to_ret,
            "max_source_length": 384,
            "max_target_length": 1536
        }
    mu = 10 if args.qa_data_name == "wow_kilt" else 13
    train_dataset = DialogDataset(train_data, tokenizer, qas_to_retrieve, max_utterances=mu, **dataset_kwargs)
    dev_dataset = DialogDataset(dev_data, tokenizer, qas_to_retrieve, **dataset_kwargs)
    test_dataset = DialogDataset(test_data, tokenizer, qas_to_retrieve, **dataset_kwargs)
    dialog_trainer = DialogTrainer(args, train_dataset, dev_dataset, test_dataset, qas_to_retrieve,
                                   normed_answer_of_qas_to_ret)

    if args.do_train:
        dialog_trainer.train()
    elif args.do_test:
        logging.info("Only do test.")
        ckpt_load_path = os.path.join(args.output_dir, "best_ckpt/pytorch_model.bin")
        gen_kwargs = {"max_length": 1024,
                      "num_beams": 5,
                      "do_sample": True,
                      "top_k": 64,
                      "no_repeat_ngram_size": 8}
        logging.warning("use dev dataset")
        use_dataset = dev_dataset

        metrics, ret_qas, gen_response = dialog_trainer.evaluate(use_dataset, update_key_memory=True,
                                                                 ckpt_load_path=ckpt_load_path, gen_kwargs=gen_kwargs)

        for k, v in metrics.items():
            logging.info(f"test_{k}: {v}")
        assert len(ret_qas) == len(gen_response) == len(use_dataset.data)
        results = []
        for retrieved, pred, input_item in zip(ret_qas, gen_response, use_dataset.data):
            results.append({
                "id": input_item["id"],
                "input": tokenizer.decode(input_item["input_ids"]),
                "target": tokenizer.decode(input_item["response_ids"]) if "response_ids" in input_item else "",
                "query": tokenizer.decode(input_item["query_ids"]),
                "output": {"answer": pred, "provenance": [{"wikipedia_id": "12904"}]},
                "retrieved_qas": [f"question: {qa['question']} answer: {qa['answer'][0]}" for qa in retrieved]
            })
        dump_path = os.path.dirname(ckpt_load_path)
        dump_path = os.path.join(dump_path, f"{time.strftime('%d %H-%M')}_predict_result.json")
        json.dump(results, open(dump_path, 'w'),
                  indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
