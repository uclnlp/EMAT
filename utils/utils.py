import argparse
import json
import logging
import os
import pickle
import random

import functools
import numpy as np
import torch
from emat.evaluation.eval_retriever import eval_retriever

from emat.evaluation.exact_match import normalize_answer
from typing import List
from emat.utils import verbalise_qa
from copy import deepcopy
from transformers import MODEL_MAPPING, T5Config, T5Tokenizer, AutoConfig, AutoTokenizer, AutoModelForMultipleChoice
from tqdm.auto import tqdm
from itertools import chain

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def reduce_query_or_key_embeds(qk_embeds, key_reduce_method):
    batch_size, key_nums, hidden_size = qk_embeds.shape
    if key_reduce_method == "concat":
        reduced_embeds = qk_embeds.view(batch_size, key_nums * hidden_size)
    elif key_reduce_method == "avg":
        reduced_embeds = qk_embeds.sum(dim=1) / key_nums
    elif key_reduce_method == "sum":
        reduced_embeds = qk_embeds.sum(dim=1)
    else:
        raise NotImplementedError(f"Reduce method ``{key_reduce_method}`` is not defined.")

    return reduced_embeds


def load_reranker(model_name_or_path="./data/models/rerankers/reranker_multi_xxlarge"):
    logging.info(f'Loading rerank model from: {model_name_or_path}')
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )
    model = model.eval()
    return model, tokenizer


def get_key_value_ae_target(qas, tokenizer, key_ae_target, value_ae_target, max_target_length, prefix=""):
    if value_ae_target == "ans":
        targets_of_value = [qa["answer"][0] for qa in qas]
    else:  # question_ans
        targets_of_value = [f'question: {qa["question"]} answer: {qa["answer"][0]}' for qa in qas]
    if key_ae_target == "question_ans":
        targets_of_key = [f'question: {qa["question"]} answer: {qa["answer"][0]}' for qa in qas]
    else:  # question
        targets_of_key = [prefix + qa["question"] for qa in qas]

    with tokenizer.as_target_tokenizer():  # setup the tokenizer for targets
        key_labels = tokenizer(targets_of_key, max_length=max_target_length,
                               padding=True, truncation=True, return_tensors="pt")
        value_labels = tokenizer(targets_of_value, max_length=max_target_length,
                                 padding=True, truncation=True, return_tensors="pt")
    return {"key_labels_input_ids": process_labels(key_labels, tokenizer),
            "value_labels_input_ids": process_labels(value_labels, tokenizer)}


def get_key_value_encoder_inputs(qas, separate_task, tokenizer, max_source_length,
                                 prefix="", only_return_key_inputs=False, value_input_is_qa=False):
    # Used to get the input of Key-Value Encoder, qas are from PAQ-L1
    if separate_task:
        key_inputs = ["question: " + qa["question"] for qa in qas]
        key_inputs = tokenizer(key_inputs, max_length=max_source_length,
                               padding=True, truncation=True, return_tensors="pt")
        if only_return_key_inputs:
            return {"key_input_ids": key_inputs["input_ids"],
                    "key_attention_mask": key_inputs["attention_mask"]}
        else:
            if value_input_is_qa:
                value_inputs = [f'question: {qa["question"]} answer: {qa["answer"][0]}' for qa in qas]
                value_inputs = tokenizer(value_inputs, max_length=max_source_length,
                                         padding=True, truncation=True, return_tensors="pt")
            else:
                value_inputs = ["answer: " + qa["answer"][0] for qa in qas]
                value_inputs = tokenizer(value_inputs, max_length=max_source_length,
                                         padding=True, truncation=True, return_tensors="pt")
            return {"key_input_ids": key_inputs["input_ids"],
                    "key_attention_mask": key_inputs["attention_mask"],
                    "value_input_ids": value_inputs["input_ids"],
                    "value_attention_mask": value_inputs["attention_mask"]}
    else:
        key_value_inputs = [prefix + verbalise_qa(qa["question"], qa["answer"][0]) for qa in qas]
        key_value_inputs = tokenizer(key_value_inputs, max_length=max_source_length,
                                     padding=True, truncation=True, return_tensors="pt")
        return {"key_value_input_ids": key_value_inputs["input_ids"],
                "key_value_attention_mask": key_value_inputs["attention_mask"]}


def get_nli_group_value_inputs(examples, tokenizer, max_source_length):
    group_key_inputs = [ex["retrieved_key_seqs"] for ex in examples]
    group_key_inputs = list(chain(*group_key_inputs))
    group_key_inputs = tokenizer(group_key_inputs, max_length=max_source_length,
                                 padding=True, truncation=True, return_tensors="pt")
    group_value_inputs = [ex["retrieved_value_seqs"] for ex in examples]
    group_value_inputs = list(chain(*group_value_inputs))
    group_value_inputs = tokenizer(group_value_inputs, max_length=max_source_length,
                                   padding=True, truncation=True, return_tensors="pt")
    return {"key_input_ids": group_key_inputs["input_ids"],
            "key_attention_mask": group_key_inputs["attention_mask"],
            "value_input_ids": group_value_inputs["input_ids"],
            "value_attention_mask": group_value_inputs["attention_mask"]}


def get_query_encoder_inputs(qas, tokenizer, max_source_length, prefix=""):
    # Used to get the input of Query Encoder, qas are from NaturalQuestion
    query_inputs = [prefix + qa["question"] for qa in qas]
    query_inputs = tokenizer(query_inputs, max_length=max_source_length,
                             padding=True, truncation=True, return_tensors="pt")
    return {"query_input_ids": query_inputs["input_ids"],
            "query_attention_mask": query_inputs["attention_mask"]}


def get_nli_input_seq(item):
    return f"hypothesis: {item['hypothesis']} premise: {item['premise']}"


def get_query_encoder_inputs_nli(cases, tokenizer, max_source_length):
    query_inputs = [case["input_seq"] for case in cases]
    query_inputs = tokenizer(query_inputs, max_length=max_source_length,
                             padding=True, truncation=True, return_tensors="pt")
    return {"query_input_ids": query_inputs["input_ids"],
            "query_attention_mask": query_inputs["attention_mask"]}


label2str = {0: "entailment", 1: "neutral", 2: "contradiction"}


def get_key_value_encoder_inputs_nli(cases, tokenizer, max_source_length, only_return_key_inputs=False):
    key_inputs = [case["input_seq"] for case in cases]
    key_inputs = tokenizer(key_inputs, max_length=max_source_length,
                           padding=True, truncation=True, return_tensors="pt")
    if only_return_key_inputs:
        return {"key_input_ids": key_inputs["input_ids"],
                "key_attention_mask": key_inputs["attention_mask"]}
    else:
        value_inputs = [label2str[case["label"]] for case in cases]
        value_inputs = tokenizer(value_inputs, max_length=max_source_length,
                                 padding=True, truncation=True, return_tensors="pt")
        return {"key_input_ids": key_inputs["input_ids"],
                "key_attention_mask": key_inputs["attention_mask"],
                "value_input_ids": value_inputs["input_ids"],
                "value_attention_mask": value_inputs["attention_mask"]}


def get_qa_inputs(qas, tokenizer, max_source_length, max_target_length, prefix="",
                  targets=None):
    # Used to get the normal inputs of QA, qas are from NaturalQuestion
    # Normal inputs and outputs
    model_inputs = [prefix + qa["question"] for qa in qas]
    model_inputs = tokenizer(model_inputs, max_length=max_source_length,
                             padding=True, truncation=True, return_tensors="pt")
    if targets is None:
        targets = [qa["answer"][0] for qa in qas]
    elif targets == "$random$":
        targets = [random.choice(qa["answer"]) for qa in qas]
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(targets, max_length=max_target_length,
                            padding=True, truncation=True, return_tensors="pt")
    model_inputs["labels"] = process_labels(targets, tokenizer)
    return model_inputs


def process_labels(labels, tokenizer, label_pad_token_id=-100, pad_to_multiple_of=None):
    if getattr(labels, "input_ids", None) is not None:
        input_ids = labels["input_ids"]
        bsz, label_length = input_ids.size()
    else:
        input_ids = labels
        bsz = len(input_ids)
        label_length = len(input_ids[0])

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    # if args.ignore_pad_token_for_loss:
    input_ids[input_ids == tokenizer.pad_token_id] = label_pad_token_id

    if pad_to_multiple_of is not None:
        max_label_length = (
                (label_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        )
        remainder = max_label_length - label_length
        if remainder > 0:
            pad_ids = torch.full(
                (bsz, remainder),
                fill_value=label_pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            input_ids = torch.cat([input_ids, pad_ids], dim=1)

    return input_ids


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def load_model(model_class, args):
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.resume_training and args.output_dir is not None:
        args.model_name_or_path = os.path.join(args.output_dir, "latest_ckpt")
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        config = T5Config.from_pretrained(args.model_name_or_path)
        model = model_class.from_pretrained(args.model_name_or_path)
        return config, tokenizer, model

    config = T5Config.from_pretrained(args.model_name_or_path)
    update_CAT_config_from_args(config, args)
    print(config)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # model = model_class(config)
    model, load_info = model_class.from_pretrained(args.model_name_or_path, config=config, output_loading_info=True)
    state_dict = torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin"))
    logging.info(f"model-load-info: {load_info}")

    manually_initialized_params = []
    if args.not_share_encoder and "kv_encoder.final_layer_norm.weight" not in state_dict.keys():
        # "kv_encoder.final_layer_norm.weight not in state-dict" means the loaded model is share-encoder.
        kv_encoder_state_dict = dict()
        for k, v in state_dict.items():
            if k in ['encoder.qk_scorer.bias', 'encoder.qk_scorer.weight']:
                continue
            if k.startswith("encoder."):
                kv_encoder_state_dict[f"{k[len('encoder.'):]}"] = deepcopy(v)
                manually_initialized_params.append(f"kv_{k}")
        model.kv_encoder.load_state_dict(kv_encoder_state_dict, strict=True)
        logging.info("Not share encoder, and initialize Key-Value encoder using CAT-encoder.")
    else:
        logging.info("Share the Key-Value encoder and CAT encoder")
    if "encoder.key_layer_norm.weight" not in state_dict.keys():
        logging.info("Initialize key_layer_norm parameters.")
        key_layer_norm_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith("encoder.final_layer_norm."):
                k = k.replace("encoder.final_layer_norm.", "")
                key_layer_norm_state_dict[k] = deepcopy(v)
                manually_initialized_params.append(f"encoder.key_layer_norm.{k}")
        model.encoder.key_layer_norm.load_state_dict(key_layer_norm_state_dict)
    logging.info(f"manually initialized parameters: {manually_initialized_params}")
    # miss_keys = load_info["missing_keys"]
    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    assert config.value_fusion_method == args.value_fusion_method
    return config, tokenizer, model


def save_model(model, save_dir, accelerator=None, tokenizer=None, arguments=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
    else:
        model.save_pretrained(save_dir)

    if tokenizer is not None:
        if accelerator is None:
            tokenizer.save_pretrained(save_dir)
        elif accelerator.is_local_main_process:
            tokenizer.save_pretrained(save_dir, save_function=accelerator.save)

    if arguments is not None:
        json.dump(vars(arguments), open(os.path.join(save_dir, "args.json"), "w"), indent=4)


def update_CAT_config_from_args(config, args):
    # Key-value memory related config (restore from CAT configs if not specified)
    config.prefix_length = getattr(config, "prefix_length", args.prefix_length)
    config.key_layer = getattr(config, "key_layer", args.key_layer)

    if args.d_key is None:
        assert args.value_fusion_method is None or "+" in args.value_fusion_method
        args.d_key = config.d_model * args.prefix_length

    if args.key_encoder_type != "prefix":
        assert args.d_key is not None

        if args.d_key is None and getattr(config, "d_key", None) is None:
            config.d_key = config.d_model
        else:
            config.d_key = getattr(config, "d_key", args.d_key)

    config.key_encoder_type = getattr(config, "key_encoder_type", args.key_encoder_type)
    config.value_layer = args.value_layer if args.value_layer is not None else config.value_layer
    config.cat_layer = getattr(config, "cat_layer", args.cat_layer)
    config.num_values = args.num_values if args.num_values is not None else config.num_values
    config.use_two_prefix = getattr(config, "use_two_prefix", args.use_two_prefix)
    config.not_share_encoder = args.not_share_encoder
    # config.value_fusion_method = getattr(config, "value_fusion_method", args.value_fusion_method)
    config.value_fusion_method = args.value_fusion_method

    if args.adapter is not None:
        config.adapter = args.adapter
        config.adapter_out_dim = args.adapter_out_dim


def find_positive_and_k_negative(top_indices, target_answers: List[List], qas_to_retrieve_from, k=8):
    positive_idx_of_qas: List[List] = []
    negative_idx_of_qas: List[List] = []
    with_positive_flag = []
    for indexes, target in zip(top_indices, target_answers):
        normalized_target = [normalize_answer(t) for t in target]
        positive_indexes = []
        negative_indexes = []
        for index in indexes:
            retrieved_qa = qas_to_retrieve_from[index]
            retrieved_answer = normalize_answer(retrieved_qa["answer"][0])
            if retrieved_answer in normalized_target:
                # if len(negative_indexes) < k:
                positive_indexes.append(index)
            else:
                if len(negative_indexes) < k:
                    negative_indexes.append(index)
            if len(positive_indexes) > 0 and len(negative_indexes) >= k:
                break
        if len(positive_indexes) > 0:
            positive_idx_of_qas.append(positive_indexes)
            negative_idx_of_qas.append(negative_indexes)
            with_positive_flag.append(True)
        else:
            with_positive_flag.append(False)
    return positive_idx_of_qas, negative_idx_of_qas, with_positive_flag


class QAs:
    def __init__(self, max_qa_idx, paq_data_root, is_iter=False):
        self.paq_data_root = paq_data_root
        self.max_qa_idx = max_qa_idx
        self.__iter_idx = 0
        self.__is_iter = is_iter

    @functools.lru_cache(maxsize=int(5e6))
    def __getitem__(self, item):
        p = os.path.join(self.paq_data_root, f"{item}")
        if item >= self.max_qa_idx:
            raise IndexError
        return pickle.load(open(p, 'rb'))

    def __len__(self):
        return self.max_qa_idx

    def __not_cached_getitem(self, item):
        if item >= self.max_qa_idx:
            raise IndexError
        else:
            return pickle.load(open(os.path.join(self.paq_data_root, f"{item}"), 'rb'))

    def __iter__(self):
        if self.__is_iter:
            return self
        else:
            return QAs(paq_data_root=self.paq_data_root, max_qa_idx=self.max_qa_idx, is_iter=True)

    def __next__(self):
        try:
            item = self.__not_cached_getitem(self.__iter_idx)
        except IndexError:
            raise StopIteration
        self.__iter_idx += 1
        return item


class CATArgs:
    def __init__(self, exp_type=None):
        self.exp_type = exp_type
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(description="CAT Arguments")
        self.parser.add_argument("--project_name", type=str, required=True, help="Project name.")
        self.parser.add_argument("--exp_name", type=str, required=True, help="Experiment name.")

        self.add_basic_arguments()
        self.add_cat_arguments()

        if self.exp_type == "build_kvm":
            self.add_build_kvm_arguments()
        elif self.exp_type == "pretrain":
            self.add_pretrain_arguments()
        elif self.exp_type == "qa_cat":
            self.add_qa_arguments()
        elif self.exp_type == "nli_cat":
            self.add_nli_arguments()
        elif self.exp_type == "nli_pretrain":
            self.add_nli_pretrain_arguments()
        elif self.exp_type == "NLU_cat":
            self.add_NLU_arguments()
        elif self.exp_type == "dialog_cat":
            self.add_dialog_arguments()
        else:
            raise ValueError(f"Experiment type {self.exp_type} is not defined.")

    def add_cat_arguments(self):
        group = self.parser.add_argument_group("Arguments for CAT model.")
        group.add_argument("--prefix_length", type=int, default=None, help="Length of the prefix.")
        group.add_argument("--use_two_prefix", action="store_true",
                           help="Use two independent prefixes to represent key and value.")
        group.add_argument("--d_key", type=int, default=None, help="The dimension of key embeddings.")
        group.add_argument("--num_values", type=int, default=1, help="Number of values returned from KV memory.")
        group.add_argument("--value_layer", type=int, default=None, help="The layer that imports Value embedding.")
        group.add_argument("--key_layer", type=int, default=None, help="The layer that computes the Key embedding.")
        group.add_argument("--key_encoder_type", type=str, default="linear", choices=["linear", "conv", "prefix"],
                           help="The type of the key encoder module.")
        group.add_argument("--separate_task", action="store_true", help="Separate the input of Key-AE and Value-AE.")

        group.add_argument("--cat_layer", type=int, default=None, help="The layer that cats key-embedding.")

        group.add_argument("--not_share_encoder", action="store_true", help="Do not share Key-Value encoder with CAT.")

        group.add_argument("--adapter", help="can be assigned to `linear", required=False, type=str)
        group.add_argument("--adapter_out_dim", required=False, type=int)
        # group.add_argument("--adapter_ckpt_path", required=False, type=str)

    def add_nli_arguments(self):
        group = self.parser.add_argument_group("Arguments for training CAT-NLI model.")

        # training args
        # Key-Value Memory args
        group.add_argument("--key_reduce_method", type=str, required=True, help="The scheduler type to use.",
                           choices=["concat", "sum", "avg"])
        group.add_argument("--kvm_fp16", action="store_true", help="FP16 Key-Value Memory")
        group.add_argument("--retrieve_strategy", type=str, default="bm25", required=False)
        group.add_argument("--do_pretrain", action="store_true")
        group.add_argument("--add_ae_weight", type=float, default=0.0)
        group.add_argument("--use_triple_loss", action="store_true")

        group.add_argument("--filter_type", type=str, default=None, required=False, help="filter BM25-results.")
        group.add_argument("--select_type", type=str, default=None, required=False, help="select BM25-results.")
        group.add_argument("--local_size", type=int, default=512, required=False)

        group.add_argument("--key_matching_weight", type=float, default=0.0)
        group.add_argument("--add_vae", action="store_true", help="default only use kae if add_ae_weight > 0.")

        # CAT-NLI architecture settings
        group.add_argument("--dataset_name", type=str, choices=["mnli", "snli"], required=False, default="mnli")
        group.add_argument("--decoder_only_attend_on_prefix", action="store_true",
                           help="Set the decoder only attend on the prefix part of encoder's output.")
        group.add_argument("--value_fusion_method", type=str, required=True, help="Assign how to use Value.")
        group.add_argument("--values_with_order", action="store_true",
                           help="when num_values > 1, if we put values by the similarity order.")
        group.add_argument("--group_cases_by_label", action="store_true",
                           help="select_type must be ``select_different_labels``  ")
        group.add_argument("--order_strategy", type=str, default="order_by_label", required=False, )
        group.add_argument("--order_by_scores", action="store_true")

        # Continue pretraining task settings
        group.add_argument("--value_repr", type=str, default="label")
        group.add_argument("--key_repr", type=str, default="hyp_prem")
        # "mnli hypothesis: xxx premise: "
        group.add_argument("--do_test", action="store_true")

    def add_NLU_arguments(self):
        group = self.parser.add_argument_group("Arguments for General-CAT-NLU model.")  # for GLUE and ...
        group.add_argument("--dataset_name", type=str, required=False, default="snli",
                           choices=["mnli", "snli", "commonsense_qa"], )
        group.add_argument("--retrieve_strategy", type=str, default="bm25", required=False)
        group.add_argument("--key_reduce_method", type=str, required=True, help="The scheduler type to use.",
                           choices=["concat", "sum", "avg"])
        group.add_argument("--do_pretrain", action="store_true")
        group.add_argument("--kvm_fp16", action="store_true", help="FP16 Key-Value Memory")
        group.add_argument("--add_ae_weight", type=float, default=0.0)
        group.add_argument("--use_triple_loss", action="store_true")
        group.add_argument("--filter_type", type=str, default=None, required=False, help="filter BM25-results.")
        group.add_argument("--select_type", type=str, default=None, required=False, help="select BM25-results.")
        group.add_argument("--local_size", type=int, default=64, required=False)
        group.add_argument("--key_matching_weight", type=float, default=0.0)
        group.add_argument("--add_vae", action="store_true", help="default only use kae if add_ae_weight > 0.")
        group.add_argument("--decoder_only_attend_on_prefix", action="store_true",
                           help="Set the decoder only attend on the prefix part of encoder's output.")
        group.add_argument("--value_fusion_method", type=str, required=True, help="Assign how to use Value.")
        group.add_argument("--values_with_order", action="store_true",
                           help="when num_values > 1, if we put values by the similarity order.")
        group.add_argument("--group_cases_by_label", action="store_true",
                           help="select_type must be ``select_different_labels``  ")
        group.add_argument("--order_strategy", type=str, default="order_by_label", required=False, )
        group.add_argument("--order_by_scores", action="store_true")
        group.add_argument("--do_test", action="store_true")

    def add_qa_arguments(self):
        group = self.parser.add_argument_group("Arguments for training CAT-QA model.")
        # updated in 2022-06-04
        group.add_argument("--build_mem_batch_size", type=int, default=2048)
        group.add_argument("--batch_local_positive_num", type=int, default=5)
        group.add_argument("--truncate_exists_local_qa", type=int, required=False, default=None)
        group.add_argument("--use_fp16_rank", action="store_true")
        group.add_argument("--use_fp16_kvm", action="store_true", help="not implement")
        group.add_argument("--PAQ_size", type=int, required=False, help="truncate PAQ to target size.")
        group.add_argument("--do_test", action="store_true")
        group.add_argument("--query_batch_size", type=int, required=True)
        group.add_argument("--only_key_matching_n_epoch", type=int, required=False, default=-1)
        group.add_argument("--gen_target_is_key_match_target", action="store_true")

        # QA args
        group.add_argument("--qa_data_name", type=str, help="choose data files from pre-defined ``DATA_PATH``")

        # training args
        group.add_argument("--search_positive_in_top_k", type=int, required=False, default=2048,
                           help="Search positives to train the key-mathing task.")
        group.add_argument("--hard_negative_num", type=int, required=False, default=12)
        group.add_argument("--least_negative_num_per_batch", type=int, required=False, default=64)
        group.add_argument("--select_positive_strategy", type=str, required=True,
                           help="The strategy of selecting one positive example for HardEM training.")
        group.add_argument("--faiss_efsearch", type=int, required=False, default=128, help="hnsw ef_search parameter")
        group.add_argument("--gen_weight", type=float, required=False, default=1.0,
                           help="Answer generation loss weight.")
        group.add_argument("--match_weight", type=float, required=False, default=1.0,
                           help="Key matching loss weight.")
        group.add_argument("--repaq_supervision_epoch", type=int, required=False, default=-1,
                           help="Use RePAQ's retrieval results as Key-matching supervision."
                                "Where we do not change the local target and, because we only prepare"
                                "one ``cur_positive_qa``/``local_positive``, the local negative is also fixed")
        group.add_argument("--only_rank_exists_local_qa", action="store_true",
                           help="Do not collect Local-QAs from entire PAQ-L1, only rank the exists Local-QAs"
                                "that retrieved by RePAQ (x large)")
        group.add_argument("--negatives_num_each_example", type=int, required=False, default=50,
                           help="sample negatives from local qas")

        # Key-Value Memory args
        group.add_argument("--kvm_dir", type=str, default=None, required=False,
                           help="The directory of Key-Value Memory")
        group.add_argument("--key_reduce_method", type=str, required=True, help="The scheduler type to use.",
                           choices=["concat", "sum", "avg"])
        group.add_argument("--qas_to_retrieve_from", type=str, help="QAs corresponding to Key-Value Memory")
        group.add_argument("--kvm_fp16", action="store_true", help="Load FP16 Key-Value Memory")
        group.add_argument("--kvm_seg_n", type=int, required=False, default=1, help="when key-memory is too large, "
                                                                                    "segment it to kvm_seg_n pieces")

        # CAT-QA architecture settings
        group.add_argument("--update_kv_embeds", action="store_true", help="Re-embed Key and Value while training.")
        group.add_argument("--local_size", type=int, required=False, help="Number of local QAs to retrieve.")
        group.add_argument("--update_local_qas", action="store_true", help="Update local QAs every epoch.")
        group.add_argument("--update_local_target_each_batch", action="store_true",
                           help="Update positive and negative each batch. Otherwise, each epoch.")
        group.add_argument("--use_not_exactly_true", action="store_true",
                           help="Input the top-1 though it is not exactly true when training the generation. "
                                "The not exactly true example will not supervise the Key-Mathing.")
        group.add_argument("--decoder_only_attend_on_prefix", action="store_true",
                           help="Set the decoder only attend on the prefix part of encoder's output.")
        group.add_argument("--value_fusion_method", type=str, required=True, help="Assign how to use Value.")
        # group.add_argument("--update_kv_batch_size", default=1024, help="Batch size of KV Re-Embed.")
        group.add_argument("--try_to_put_one_positive_in_values", action="store_true",
                           help="when num_values > 1, if we force to put at least one positive QA to Values")
        group.add_argument("--values_with_order", action="store_true",
                           help="when num_values > 1, if we put values by the similarity order.")

        group.add_argument("--pos_from_top", type=int, required=False, default=50,
                           help="for each batch, select lexical-positive QAs from ranked-local-QAs from top-X")

        group.add_argument("--rerank_retrieved_values", action="store_true")

        # Continue pretraining task settings
        group.add_argument("--add_ae_task", action="store_true", help="Add the pretraining(Auto-Encoding) task.")
        group.add_argument("--ae_weight", type=float, required=False, default=0.1,
                           help="When set --add_ae_task, the auto-encoding loss weight.")
        group.add_argument("--ae_batch_size", type=int, default=None, help="The batch size of AE task.")
        group.add_argument("--value_ae_target", type=str, default="ans", choices=["ans", "question_ans"])
        group.add_argument("--key_ae_target", type=str, default="question_ans", choices=["question", "question_ans"])

        # adapter training
        group.add_argument("--only_train_adapter", action="store_true")
        group.add_argument("--use_adapter_to_select_positive_after_k_epoch", type=int,
                           default=float("inf"), required=False)

    def add_dialog_arguments(self):
        group = self.parser.add_argument_group("Arguments for training CAT-Dialog model.")
        # updated in 2022-06-18
        group.add_argument("--build_mem_batch_size", type=int, default=2048)
        group.add_argument("--batch_local_positive_num", type=int, default=5)
        group.add_argument("--do_test", action="store_true")
        group.add_argument("--query_batch_size", type=int, required=True)
        group.add_argument("--add_persona", action="store_true")
        group.add_argument("--add_topic", action="store_true")
        group.add_argument("--update_kv_embeds", action="store_true", help="Re-embed Key and Value while training.")
        group.add_argument("--eval_every_n_steps", required=False, default=None, type=int)
        group.add_argument("--shortest_answer_len", required=False, default=None, type=int)

        # training args
        group.add_argument("--select_positive_strategy", type=str, required=False, default="softmax_sample",
                           help="The strategy of selecting one positive example for HardEM training.")
        group.add_argument("--faiss_efsearch", type=int, required=False, default=128, help="hnsw ef_search parameter")
        group.add_argument("--gen_weight", type=float, required=False, default=1.0,
                           help="Answer generation loss weight.")
        group.add_argument("--match_weight", type=float, required=False, default=1.0,
                           help="Key matching loss weight.")
        group.add_argument("--negatives_num_each_example", type=int, required=False, default=50,
                           help="sample negatives from local qas")
        group.add_argument("--qa_data_name", type=str, help="choose data files from pre-defined ``DATA_PATH``")

        # Key-Value Memory args
        group.add_argument("--key_reduce_method", type=str, required=True, help="The scheduler type to use.",
                           choices=["concat", "sum", "avg"])
        group.add_argument("--qas_to_retrieve_from", type=str, help="QAs corresponding to Key-Value Memory")
        group.add_argument("--kvm_fp16", action="store_true", help="Load FP16 Key-Value Memory")
        group.add_argument("--kvm_seg_n", type=int, required=False, default=1, help="when key-memory is too large, "
                                                                                    "segment it to kvm_seg_n pieces")

        group.add_argument("--local_size", type=int, required=False, help="Number of local QAs to retrieve.")
        group.add_argument("--decoder_only_attend_on_prefix", action="store_true",
                           help="Set the decoder only attend on the prefix part of encoder's output.")
        group.add_argument("--value_fusion_method", type=str, required=True, help="Assign how to use Value.")
        group.add_argument("--values_with_order", action="store_true",
                           help="when num_values > 1, if we put values by the similarity order.")
        group.add_argument("--pos_from_top", type=int, required=False, default=128,
                           help="for each batch, select lexical-positive QAs from ranked-local-QAs from top-X")

    def add_nli_pretrain_arguments(self):
        group = self.parser.add_argument_group("Arguments for pretraining NLI-CAT.")
        group.add_argument("--value_ae_weight", type=float, default=1.0,
                           help="Weight for the Auto-Encoding loss of Value.")
        group.add_argument("--key_ae_weight", type=float, default=1.0,
                           help="Weight for the Auto-Encoding loss of Key.")
        group.add_argument("--train_value", action="store_true")
        group.add_argument("--train_key", action="store_true")
        group.add_argument("--separate_decode", action="store_true")

        group.add_argument("--value_fusion_method", type=str, required=True,
                           help="Assign how to use Value. Preassigned in pretrain.")

    def add_pretrain_arguments(self):
        group = self.parser.add_argument_group("Arguments for pretraining CAT.")
        # add in 2022-06-19
        group.add_argument("--value_input_is_qa", action="store_true")

        group.add_argument("--pretrain_data_name", type=str, help="choose data files from pre-defined ``DATA_PATH``")
        group.add_argument("--value_ae_weight", type=float, default=1.0,
                           help="Weight for the Auto-Encoding loss of Value.")
        group.add_argument("--key_ae_weight", type=float, default=1.0,
                           help="Weight for the Auto-Encoding loss of Key.")
        group.add_argument("--value_ae_target", type=str, default="ans", choices=["ans", "question_ans"])
        group.add_argument("--key_ae_target", type=str, default="question_ans", choices=["question", "question_ans"])
        group.add_argument("--train_value", action="store_true")
        group.add_argument("--train_key", action="store_true")
        group.add_argument("--pretrain_multi_values", action="store_true")
        group.add_argument("--value_fusion_method", type=str, required=False, help="Assign how to use Value.")
        group.add_argument("--decoder_only_attend_on_prefix", action="store_true",
                           help="Set the decoder only attend on the prefix part of encoder's output.")
        group.add_argument("--key_reduce_method", type=str, required=True, help="The scheduler type to use.",
                           choices=["concat", "sum", "avg"])
        group.add_argument("--gen_weight", type=float, default=1.0)
        group.add_argument("--value_with_self_prop", type=float, default=1.0)
        group.add_argument("--values_with_order", action="store_true",
                           help="when num_values > 1, if we put values by the similarity order.")

    def add_build_kvm_arguments(self):
        group = self.parser.add_argument_group("Arguments for building Key-Value Memory.")
        group.add_argument("--embed_key", action="store_true")
        group.add_argument("--embed_value", action="store_true")
        group.add_argument("--embed_as_fp16", action="store_true")
        group.add_argument("--key_reduce_method", type=str, required=True, help="The scheduler type to use.",
                           choices=["concat", "sum", "avg"])
        group.add_argument("--embed_data_name", type=str, help="choose data files from pre-defined ``DATA_PATH``")

    def parse_args(self, print_args=True):
        args = self.parser.parse_args()
        if print_args:
            args_str = json.dumps(vars(args), indent=4, ensure_ascii=False)
            logging.info(f"Show the arguments: \n{args_str}")

        assert args.value_layer >= args.key_layer
        if args.num_values > 1:
            assert args.value_fusion_method in ["cat_v", "cat_kv", "cat_k+v", "cat_avgk+v", "cat_k_delay+v",
                                                "cat_k+v_g(kq)", "cat_k_delay+v_g(kv)",
                                                "async_cat_k+v", "async_cat_k_delay+v"]
            if "delay" not in args.value_fusion_method:
                if args.cat_layer is None and "k" in args.value_fusion_method:
                    assert args.key_layer == args.value_layer
                elif "async_cat_k+v" == args.value_fusion_method:
                    assert args.cat_layer == args.value_layer

        if self.exp_type == "pretrain":
            assert args.train_key or args.train_value, "at least train one of them"
            assert not os.path.exists(os.path.join(args.output_dir, "best_ckpt")), \
                "The experiment has done before. Clear the dir before running"
        else:
            assert self.exp_type in ["build_kvm", "qa_cat", "nli_cat", "nli_pretrain", "NLU_cat", "dialog_cat"]

        return args

    def add_basic_arguments(self):
        group = self.parser.add_argument_group("Arguments for basic settings.")
        # generation args
        group.add_argument("--num_beams", type=int, default=None,
                           help="Number of beams to use for evaluation. This argument will be passed to "
                                "``model.generate``, which is used during ``evaluate`` and ``predict``.")
        # data args
        group.add_argument("--source_prefix", type=str, default=None,
                           help="A prefix to add before every source text (useful for T5 models).")
        group.add_argument("--max_source_length", type=int, default=1024,
                           help="The maximum total input sequence length after tokenization."
                                "Sequences longer than this will be truncated, sequences shorter will be padded.")
        group.add_argument("--max_target_length", type=int, default=128,
                           help="The maximum total sequence length for target text after tokenization. "
                                "Sequences longer than this will be truncated, "
                                "sequences shorter will be padded. during ``evaluate`` and ``predict``.")
        group.add_argument("--val_max_target_length", type=int, default=None,
                           help="The maximum total sequence length for validation target text after tokenization."
                                "Sequences longer than this will be truncated, sequences shorter will be padded."
                                "Will default to `max_target_length`. This argument is also used to override the "
                                "``max_length`` param of ``model.generate``, "
                                "which is used during ``evaluate`` and ``predict``.")
        group.add_argument("--pad_to_max_length", type=bool, default=False,
                           help="Whether to pad all samples to model maximum sentence length. If False, will pad "
                                "the samples dynamically when batching to the maximum length in the batch. More"
                                "efficient on GPU but very bad for TPU.", )

        # training args
        group.add_argument("--ignore_pad_token_for_loss", type=bool, default=True,
                           help="Whether to ignore the tokens corresponding to padded "
                                "labels in the loss computation")
        group.add_argument("--preprocessing_num_workers", type=int, default=None,
                           help="The number of processes to use for the preprocessing.")
        group.add_argument("--model_name_or_path", type=str, required=True,
                           help="Path to pretrained model or model identifier from huggingface.co/models.")
        group.add_argument("--config_name", type=str, default=None,
                           help="Pretrained config name or path if not the same as model_name")
        group.add_argument("--tokenizer_name", type=str, default=None,
                           help="Pretrained tokenizer name or path if not the same as model_name")
        group.add_argument("--use_slow_tokenizer", action="store_true",
                           help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
        group.add_argument("--per_device_train_batch_size", type=int, required=True,
                           help="Batch size (per device) for the training dataloader.")
        group.add_argument("--per_device_eval_batch_size", type=int, default=8,
                           help="Batch size (per device) for the evaluation dataloader.")

        group.add_argument("--learning_rate", type=float, default=5e-5,
                           help="Initial learning rate (after the potential warmup period) to use.")
        group.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
        group.add_argument("--num_train_epochs", type=int, default=20,
                           help="Total number of training epochs to perform.")
        group.add_argument("--early_stop_patience", type=int, default=1000000,
                           help="Early stop if the performance does not improve for this number of epochs .")

        group.add_argument("--max_train_steps", type=int, default=None,
                           help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
        group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Number of updates steps to accumulate before performing a backward/update pass.")
        group.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use.",
                           choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                    "constant_with_warmup"], )
        group.add_argument("--num_warmup_steps", type=int, default=0,
                           help="Number of steps for the warmup in the lr scheduler.")
        group.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")
        group.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        group.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.",
                           choices=MODEL_TYPES)

        group.add_argument("--do_train", action="store_true", help="Whether to train the model on the train set.")
        group.add_argument("--do_eval", action="store_true", help="Whether to evaluate on the dev set.")
        group.add_argument("--eval_freq", type=int, default=1,
                           help="Frequency of evaluation on the dev set (if do_eval is True).")
        group.add_argument("--resume_training", action="store_true", help="Resume training from the latest checkpoint.")
        group.add_argument("--freeze_t5_params", action="store_true", help="Freeze the original T5 parameters.")
        group.add_argument("--per_epoch_eval_times", type=int, default=1,
                           help="do eval many times per epoch", required=False)
