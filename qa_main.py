import json
import os
import pickle
from transformers import T5Tokenizer
from emat.utils import load_jsonl
from utils.utils import CATArgs
from qa_dataset import QADataset
from qa_trainer import QATrainer
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

DATA_PATHS = {
    "nq": {
        "train": "./data/annotated_datasets/NQ-open.train-train.jsonl",
        "validation": "./data/annotated_datasets/NQ-open.train-dev.jsonl",
        "test": "./data/annotated_datasets/NQ-open.test.jsonl"
    },
    "tq": {
        "train": "./data/annotated_datasets/triviaqa.train-train.jsonl",
        "validation": "./data/annotated_datasets/triviaqa.train-dev.jsonl",
        "test": "./data/annotated_datasets/triviaqa.test.jsonl"
    },
    "wq": {
        "train": "./data/annotated_datasets/WQ-trainmodel.jsonl",
        "validation": "./data/annotated_datasets/WQ-val.jsonl",
        "test": "./data/annotated_datasets/WQ-test.jsonl"
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
    data_paths = DATA_PATHS[args.qa_data_name]
    test_data = load_jsonl(DATA_PATHS[args.qa_data_name]["test"])
    train_data = load_jsonl(data_paths["train"])
    dev_data = load_jsonl(data_paths["validation"])

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
        train_data = train_data[:100]
        dev_data = dev_data[:500]
        qas_to_retrieve = qas_to_retrieve[:10000]
        normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret[:len(qas_to_retrieve)]

    if args.qas_to_retrieve_from == "PAQ" and args.PAQ_size is not None:
        qas_to_retrieve = qas_to_retrieve[:args.PAQ_size]
        normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret[:args.PAQ_size]
        assert len(qas_to_retrieve) == args.PAQ_size
        logging.info(f"select {args.PAQ_size}-size PAQ.")

    assert len(normed_answer_of_qas_to_ret) == len(qas_to_retrieve)
    loaded_data = {
        "train": train_data, "validation": dev_data, "test": test_data,
        "qas_to_retrieve": qas_to_retrieve,
        "normed_answer_of_qas_to_ret": normed_answer_of_qas_to_ret
    }

    return loaded_data


def main():
    cat_args = CATArgs("qa_cat")
    args = cat_args.parse_args()
    loaded_data = load_dataset(args)
    logging.info("data loaded.")
    train_data, dev_data, test_data = loaded_data["train"], loaded_data["validation"], loaded_data["test"]
    qas_to_retrieve = loaded_data["qas_to_retrieve"]
    normed_answer_of_qas_to_ret = loaded_data["normed_answer_of_qas_to_ret"]

    dataset_kwargs = {
        "max_source_length": args.max_source_length,
        "dataset_name": args.qa_data_name,
        "args": args,
        "normed_answer_of_qas_to_ret": normed_answer_of_qas_to_ret,
    }
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    train_dataset = QADataset(train_data, tokenizer, qas_to_retrieve, **dataset_kwargs)
    dev_dataset = QADataset(dev_data, tokenizer, qas_to_retrieve, **dataset_kwargs)
    test_dataset = QADataset(test_data, tokenizer, qas_to_retrieve, **dataset_kwargs)

    qa_trainer = QATrainer(args, train_dataset, dev_dataset, test_dataset, qas_to_retrieve, normed_answer_of_qas_to_ret)

    if args.do_train:
        qa_trainer.train()
    elif args.do_test:
        logging.info("Only do test.")
        ckpt_load_path = os.path.join(args.output_dir, "best_ckpt/pytorch_model.bin")
        em_score, match_metric, ret_qas, gen_ans = qa_trainer.evaluate(
            qa_trainer.test_dataset, extend_mem_from="train_dev",
            update_key_memory=True, ckpt_load_path=ckpt_load_path

        )

        logging.info(f"em_test: {em_score:.3f}")
        for k, v in match_metric.items():
            logging.info(f"test_{k}: {v}")
        results = []
        for idx, (input_qa, retrieved_qas, predict_answer) in enumerate(zip(qa_trainer.test_dataset, ret_qas, gen_ans)):
            results.append({
                "idx": idx,
                "question": input_qa["question"],
                "answer": input_qa["answer"],
                "retrieved_qas": [{"question": qa["question"], "answer": qa["answer"][0]} for qa in retrieved_qas],
                "generate_answer": predict_answer,
            })
        json.dump(results, open(os.path.join(args.output_dir, "best_ckpt/predict_results.json"), 'w'),
                  indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
