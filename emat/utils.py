import json
import logging

import torch

logger = logging.getLogger(__name__)

try:
    import apex
    from apex import amp

    apex.amp.register_half_function(torch, "einsum")
    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


def to_fp16(model):
    if is_apex_available():
        model = amp.initialize(model, opt_level="O1")
    else:
        model = model.half()
    return model


def load_jsonl(fn):
    all_data = []
    with open(fn, "r") as f:
        for line in f.readlines():
            all_data.append(json.loads(line))
    return all_data


def write_jsonl(all_data, fn):
    with open(fn, "w") as f:
        for data in all_data:
            f.write(json.dumps(data) + "\n")


def convert_repaq_results_from_file(in_file, num_candidates, out_file=None):
    data = load_jsonl(in_file)
    processed_data = []

    for sample in data:
        sample_dict = {
            "question": sample["input_qa"]["question"],
            "answer": sample["input_qa"]["answer"],
        }
        for i, qas in enumerate(sample["retrieved_qas"][:num_candidates]):
            q, a = qas["question"], qas["answer"][0]
            sample_dict[f"ret_q_{i}"] = q
            sample_dict[f"ret_a_{i}"] = a

        processed_data.append(sample_dict)

    if out_file is not None:
        write_jsonl(processed_data, out_file)

    return processed_data


def verbalise_qa(q, a) -> str:
    q = q.strip("?")
    return f'{q}? answer: {a}'
