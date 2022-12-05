import argparse
import logging
import math
import os
import pickle
import random
import tempfile
from functools import partial
import time
from itertools import chain

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset, load_metric, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    get_scheduler,
    set_seed,
)

from transformers.utils.versions import require_version
import json

from emat.evaluation.eval_retriever import eval_generation_em
from emat.t5 import T5WithKeyValueMemory, CATEncoderOutput
from utils.utils import CATArgs, update_CAT_config_from_args, save_model, get_key_value_encoder_inputs, \
    get_key_value_ae_target, get_qa_inputs, load_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# Initialise wandb
try:
    import wandb

    wandb.login(key="09cf72d9c096a95d573fd2b857bfd601022bc4b7")
    os.environ["WANDB_API_KEY"] = "09cf72d9c096a95d573fd2b857bfd601022bc4b7"
    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn(
            "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False
    logger.info("no WANDB")
    exit()

DATA_PATHS = {
    "nq-repaq": {
        # "train": "repaq_results/nq.train-train.retriever_multi_base_256.jsonl",
        # "validation": "repaq_results/nq.train-dev.retriever_multi_base_256.jsonl",
        # "test": "repaq_results/nq.test.retriever_multi_base_256.jsonl"
        "train": "./cbqa_data/RePAQ/RePAQ-output-NQ-train.jsonl",
        "validation": "./cbqa_data/RePAQ/RePAQ-output-NQ-dev.jsonl",
    },
    "PAQ-L1-Pretrain": {
        "train": "./data/cbqa_data/pretrain_data/paq-l1-pretrain-train.jsonl",
        # "validation": "./data/cbqa_data/pretrain_data/paq-l1-pretrain-dev.jsonl"
        "validation": "./data/cbqa_data/pretrain_data/paq-l1-pretrain-dev-3000.jsonl"
    },
    "PAQ-L1-Small": {
        "train": "./data/cbqa_data/pretrain_data/paq-l1-small-train.jsonl",  # 10w examples from PAQ-L1
        "validation": "./data/cbqa_data/pretrain_data/paq-l1-pretrain-dev-3000.jsonl"
    },
    "data_for_debug": {
        "train": "./data/cbqa_data/pretrain_data/debug.jsonl",
        "validation": "./data/cbqa_data/pretrain_data/debug.jsonl"
    }
}
PAQ_PATH = "./data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl"
PAQ_L1_PATH = "./data/cbqa_data/pretrain_data/PAQ_L1/PAQ_L1.filtered.jsonl"

qas_to_retrieve = pickle.load(open("./tmp/PAQ_L1_pickl_file.pkl", 'rb'))


# all_data = load_jsonl("./data/cbqa_data/pretrain_data/paq-l1-pretrain-train.jsonl")
# train_data = all_data[:100000]
# write_jsonl(train_data, DATA_PATHS["PAQ-L1-Small"]["train"])

def load_pretrain_kvm_data(args) -> DatasetDict:
    assert args.pretrain_data_name in DATA_PATHS.keys(), f"available dataset names: {DATA_PATHS.keys()}"
    data_paths = DATA_PATHS[args.pretrain_data_name]
    data_files = {
        "train": [data_paths["train"]],
        "validation": [data_paths["validation"]]
    }
    return load_dataset("json", data_files=data_files)


def main():
    # Parse the arguments
    # args = parse_args()
    cat_args = CATArgs(exp_type="pretrain")
    args = cat_args.parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.

    logging.info(f"wandb {'available' if _has_wandb else 'unavailable'}")

    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process and _has_wandb:
        wandb.init(project=args.project_name, name=args.exp_name, dir=args.output_dir, config=vars(args))

    logging.info("loading model")
    config, tokenizer, model = load_model(T5WithKeyValueMemory, args)
    logging.info("Loading data.")

    if args.freeze_t5_params:
        for name, param in model.named_parameters():
            if 'prefix_embedding' in name or 'key_encoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length

    with_ae_task = args.key_ae_weight > 0.0 and args.value_ae_weight > 0.0
    assert with_ae_task or args.pretrain_multi_values

    def preprocess_pretrain_input_func(examples):
        # Normal inputs and outputs
        model_inputs = get_key_value_encoder_inputs(examples, args.separate_task, tokenizer, args.max_source_length,
                                                    prefix=prefix, only_return_key_inputs=False)
        model_inputs.update(get_key_value_ae_target(examples, tokenizer, args.key_ae_target, args.value_ae_target,
                                                    max_target_length))

        return model_inputs

    def preprocess_pretrain_multi_values_input_func(examples, value_with_self_prop=None):
        #  How to build the dataset
        # we hope two kinds of data contained in dataset:
        # 1) top-k data, and the golden answer is contained in top-k QAs
        # 2) top-k data do not contain the golden answer
        # but all data's target are golden
        model_inputs = get_qa_inputs(examples, tokenizer, args.max_source_length, max_target_length, prefix=prefix,
                                     targets=None)
        value_qas = []
        for ex in examples:
            if random.random() < value_with_self_prop:
                selected_values_indices = list(ex["retrieved_PAQL1_indices"][:args.num_values])
            else:
                selected_values_indices = list(ex["retrieved_PAQL1_indices"][1:args.num_values + 1])
            if not args.values_with_order:
                random.shuffle(selected_values_indices)
            selected_values = [qas_to_retrieve[idx] for idx in selected_values_indices]
            value_qas.append(selected_values)
        value_qas = list(chain(*value_qas))  # bs * num_values
        group_value_inputs = get_key_value_encoder_inputs(value_qas, args.separate_task, tokenizer,
                                                          args.max_source_length, prefix=prefix,
                                                          value_input_is_qa=args.value_input_is_qa)
        for dk in group_value_inputs:
            model_inputs[f"group_value_inputs_{dk}"] = group_value_inputs[dk]

        model_inputs.update(
            get_key_value_encoder_inputs(examples, args.separate_task, tokenizer, args.max_source_length,
                                         prefix=prefix, only_return_key_inputs=False,
                                         value_input_is_qa=args.value_input_is_qa))

        if with_ae_task:
            model_inputs.update(get_key_value_ae_target(examples, tokenizer, args.key_ae_target,
                                                        args.value_ae_target, max_target_length))

        return model_inputs

    # Evaluation metric: load the custom exact-match metric

    # Load the training/validation datasets
    if args.pretrain_multi_values:
        if "debug" not in args.exp_name:
            pretrain_data_path = "./tmp/PAQ_L1_with_50_xlarge_retrieved_qa_indices.pkl"
            # "tmp/PAQ_L1_with_retrieved_qa_indices.pkl"
            raw_datasets = pickle.load(open(pretrain_data_path, "rb"))
            train_dataset = raw_datasets[:-5000]
            eval_dataset = raw_datasets[-5000:]
        else:
            raw_datasets = pickle.load(open("./tmp/PAQ_L1_5k_with_retrieved_qa_indices.pkl", 'rb'))
            train_dataset = raw_datasets[:100]
            eval_dataset = raw_datasets[:100]
        collate_fn = partial(preprocess_pretrain_multi_values_input_func,
                             value_with_self_prop=args.value_with_self_prop)
    else:
        raw_datasets = load_pretrain_kvm_data(args)
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]
        collate_fn = preprocess_pretrain_input_func
    # DataLoaders creation:
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  collate_fn=collate_fn,
                                  batch_size=args.per_device_train_batch_size,
                                  num_workers=args.preprocessing_num_workers)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=collate_fn,
                                 batch_size=args.per_device_eval_batch_size,
                                 num_workers=args.preprocessing_num_workers)

    without_self_eval_dataloader = DataLoader(eval_dataset,
                                              collate_fn=partial(preprocess_pretrain_multi_values_input_func,
                                                                 value_with_self_prop=0.0),
                                              batch_size=args.per_device_eval_batch_size,
                                              num_workers=args.preprocessing_num_workers)

    if not args.do_train and args.do_eval:
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
        metric_key, metric_value, all_gen_ans = evaluate(args, model, config, eval_dataloader, accelerator, tokenizer)
        if args.train_key:
            key_em_score = metric_key.compute()["em"] * 100  # EM score is not in percentage points
            logger.info(f"eval - Key-EM: {key_em_score:.2f}")
        if args.train_value:
            value_em_score = metric_value.compute()["em"] * 100  # EM score is not in percentage points
            logger.info(f"eval - Value-EM: {value_em_score:.2f}")

        em_score = eval_generation_em(eval_dataset, all_gen_ans) * 100
        logger.info(f"em_test: {em_score:.2f}")
        exit()

    if args.do_train:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        eval_times = 0

        best_em, patience = None, args.early_stop_patience
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                assert args.separate_task
                loss = 0.
                loss_dict = dict()
                if args.pretrain_multi_values:
                    group_inputs = {k.replace("group_value_inputs_", ""): v for k, v in batch.items() if
                                    k.startswith("group_value_inputs_")}
                    embed_dict = model.wrapped_embed_kv(
                        separate_task=args.separate_task, compute_key=True, compute_value=True, **group_inputs
                    )
                    key_embeds_of_value = embed_dict["key_embeds"]
                    value_embeds = embed_dict["value_embeds"]
                    bs = batch["input_ids"].shape[0]
                    value_embeds = value_embeds.view(bs, args.num_values, args.prefix_length, -1)
                    key_embeds_of_value = key_embeds_of_value.view(bs, args.num_values, -1, model.model_dim)

                    loss_dict_gen = model.compute_qa_loss(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        decoder_only_attend_on_prefix=args.decoder_only_attend_on_prefix,
                        value_fusion_method=args.value_fusion_method,
                        encoder_outputs_are_key_or_value=False,
                        key_reduce_method=args.key_reduce_method,
                        value_embeds=value_embeds,
                        key_embeds_of_value=key_embeds_of_value
                    )
                    loss += args.gen_weight * loss_dict_gen["gen_loss"]
                    loss_dict.update({k: v for k, v in loss_dict_gen.items()})
                if with_ae_task:
                    loss_dict_ae = model.compute_key_value_ae_loss(
                        train_key=args.train_key,
                        train_value=args.train_value,
                        separate_task=True,
                        key_input_ids=batch["key_input_ids"],
                        key_attention_mask=batch["key_attention_mask"],
                        value_input_ids=batch["value_input_ids"],
                        value_attention_mask=batch["value_attention_mask"],
                        key_labels_input_ids=batch["key_labels_input_ids"],
                        value_labels_input_ids=batch["value_labels_input_ids"],
                    )
                    if args.train_key:
                        loss += args.key_ae_weight * loss_dict_ae["key_ae_loss"]
                    if args.train_value:
                        loss += args.value_ae_weight * loss_dict_ae["value_ae_loss"]
                    loss_dict.update({k: v.item() for k, v in loss_dict_ae.items()})

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                    if accelerator.is_local_main_process and _has_wandb:
                        wandb.log({"loss": loss * args.gradient_accumulation_steps, "step": completed_steps})
                        for k, v in loss_dict.items():
                            wandb.log({k: v, "step": completed_steps})

                    if completed_steps % 5555 == 0:
                        metric_key, metric_value, all_gen_ans = evaluate(
                            args, model, config, eval_dataloader, accelerator, tokenizer)

                        scores = []
                        if args.train_key:
                            key_em_score = metric_key.compute()["em"] * 100  # EM score is not in percentage points
                            logger.info(f"epoch {epoch} eval-time {eval_times} - Key-EM: {key_em_score:.2f}")
                            if accelerator.is_local_main_process and _has_wandb:
                                wandb.log({"key_em_dev": key_em_score, "eval_times": eval_times})
                            scores.append(key_em_score)
                        if args.train_value:
                            value_em_score = metric_value.compute()["em"] * 100  # EM score is not in percentage points
                            logger.info(f"epoch {epoch} eval - Value-EM: {value_em_score:.2f}")
                            if accelerator.is_local_main_process and _has_wandb:
                                wandb.log({"value_em_dev": value_em_score, "eval_times": eval_times})
                            scores.append(value_em_score)

                        em_score = eval_generation_em(eval_dataset, all_gen_ans) * 100
                        logger.info(f"em_test: {em_score:.2f}")
                        wandb.log({"em_test": em_score, "eval_times": eval_times})
                        if args.output_dir is not None:
                            if best_em is None or em_score > best_em:
                                best_em = em_score
                                save_model(model, os.path.join(args.output_dir, "best_ckpt"), accelerator,
                                           tokenizer=tokenizer, arguments=args)

                        # without self eval
                        _, _, all_gen_ans = evaluate(args, model, config, without_self_eval_dataloader,
                                                     accelerator, tokenizer)
                        em_score = eval_generation_em(eval_dataset, all_gen_ans) * 100
                        logger.info(f"w/o_self_em_test: {em_score:.2f}")
                        wandb.log({"w/o_self_em_test": em_score, "eval_times": eval_times})

                        eval_times += 1

                if completed_steps >= args.max_train_steps:
                    break

            if args.output_dir is not None:
                save_model(model, os.path.join(args.output_dir, "latest_ckpt"), accelerator, tokenizer=tokenizer,
                           arguments=args)

            if args.do_eval and epoch % args.eval_freq == 0:
                metric_key, metric_value, all_gen_ans = evaluate(args, model, config, eval_dataloader, accelerator,
                                                                 tokenizer)

                scores = []
                if args.train_key:
                    key_em_score = metric_key.compute()["em"] * 100  # EM score is not in percentage points
                    logger.info(f"epoch {epoch} eval - Key-EM: {key_em_score:.2f}")
                    if accelerator.is_local_main_process and _has_wandb:
                        wandb.log({"key_em_dev": key_em_score, "epoch": epoch})
                    scores.append(key_em_score)
                if args.train_value:
                    value_em_score = metric_value.compute()["em"] * 100  # EM score is not in percentage points
                    logger.info(f"epoch {epoch} eval - Value-EM: {value_em_score:.2f}")
                    if accelerator.is_local_main_process and _has_wandb:
                        wandb.log({"value_em_dev": value_em_score, "epoch": epoch})
                    scores.append(value_em_score)

                em_score = eval_generation_em(eval_dataset, all_gen_ans) * 100
                logger.info(f"em_test: {em_score:.2f}")
                wandb.log({"em_test": em_score, "epoch": epoch})
                if args.output_dir is not None:
                    if best_em is None or em_score > best_em:
                        best_em = em_score
                        save_model(model, os.path.join(args.output_dir, "best_ckpt"), accelerator,
                                   tokenizer=tokenizer,
                                   arguments=args)

                # without self eval
                _, _, all_gen_ans = evaluate(args, model, config, without_self_eval_dataloader,
                                             accelerator, tokenizer)
                em_score = eval_generation_em(eval_dataset, all_gen_ans) * 100
                logger.info(f"w/o_self_em_test: {em_score:.2f}")
                wandb.log({"w/o_self_em_test": em_score, "epoch": epoch})

        if best_em is not None:  # Log the best dev EM score
            wandb.log({"best_em_dev": best_em})


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


@torch.no_grad()
def evaluate(args, model, config, eval_dataloader, accelerator, tokenizer):
    metric_value = load_metric("emat/evaluation/exact_match.py")
    metric_key = load_metric("emat/evaluation/exact_match.py")
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }
    all_gen_ans = []

    for batch in tqdm(eval_dataloader):
        model.eval()
        group_inputs = {k.replace("group_value_inputs_", ""): v for k, v in batch.items() if
                        k.startswith("group_value_inputs_")}
        embed_dict = model.wrapped_embed_kv(
            separate_task=args.separate_task, compute_key=True, compute_value=True, **group_inputs
        )
        key_embeds_of_value = embed_dict["key_embeds"]
        value_embeds = embed_dict["value_embeds"]
        bs = batch["input_ids"].shape[0]
        value_embeds = value_embeds.view(bs, args.num_values, args.prefix_length, -1)
        key_embeds_of_value = key_embeds_of_value.view(bs, args.num_values, -1, model.model_dim)
        encoder_outputs = model.encoder(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            return_dict=True,
            value_embeds=value_embeds,
            readout_top_k=-1,
            key_reduce_method=args.key_reduce_method,
            value_fusion_method=args.value_fusion_method,
            key_embeds_of_value=key_embeds_of_value
        )
        generated_tokens = accelerator.unwrap_model(model).generate(
            encoder_outputs=encoder_outputs,
            encoder_outputs_are_key_or_value=False,
            decoder_only_attend_on_prefix=args.decoder_only_attend_on_prefix,
            attention_mask=batch["attention_mask"].to(model.device),
            value_fusion_method=args.value_fusion_method,
            **gen_kwargs,
        )
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_tokens = [ans.strip() for ans in decoded_tokens]
        all_gen_ans += decoded_tokens

        # Auto-Encoding loss
        embed_dict = model.wrapped_embed_kv(
            separate_task=args.separate_task,
            key_input_ids=batch["key_input_ids"],
            key_attention_mask=batch["key_attention_mask"],
            value_input_ids=batch["value_input_ids"],
            value_attention_mask=batch["value_attention_mask"],
            compute_key=args.train_key,
            compute_value=args.train_value,
            embed_for_ae_task=True
        )
        key_embeds = embed_dict["normed_key_embeds"]
        value_embeds = embed_dict["normed_value_embeds"]  # normed value for generation

        if args.train_key:
            key_labels = batch["key_labels_input_ids"]
            key_embeds = key_embeds.view(key_embeds.shape[0], -1, model.model_dim)

            recovered_from_key = accelerator.unwrap_model(model).generate(
                encoder_outputs=CATEncoderOutput(last_hidden_state=key_embeds, hidden_states=None, attentions=None),
                attention_mask=None, encoder_outputs_are_key_or_value=True, **gen_kwargs,
            )
            recovered_from_key = accelerator.pad_across_processes(
                recovered_from_key, dim=1, pad_index=tokenizer.pad_token_id
            )
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                key_labels = accelerator.pad_across_processes(key_labels, dim=1, pad_index=tokenizer.pad_token_id)
            recovered_from_key = accelerator.gather(recovered_from_key).cpu().numpy()
            key_labels = accelerator.gather(key_labels).cpu().numpy()
            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                key_labels = np.where(key_labels != -100, key_labels, tokenizer.pad_token_id)
            decoded_key = tokenizer.batch_decode(recovered_from_key, skip_special_tokens=True)
            decoded_key_labels = tokenizer.batch_decode(key_labels, skip_special_tokens=True)
            decoded_key, decoded_key_labels = postprocess_text(decoded_key, decoded_key_labels)
            metric_key.add_batch(predictions=decoded_key, references=decoded_key_labels)

        if args.train_value:
            value_labels = batch["value_labels_input_ids"]
            value_embeds = value_embeds.view(value_embeds.shape[0], -1, model.model_dim)
            recovered_from_value = accelerator.unwrap_model(model).generate(
                encoder_outputs=CATEncoderOutput(last_hidden_state=value_embeds, hidden_states=None, attentions=None),
                attention_mask=None, encoder_outputs_are_key_or_value=True, **gen_kwargs,
            )
            recovered_from_value = accelerator.pad_across_processes(
                recovered_from_value, dim=1, pad_index=tokenizer.pad_token_id
            )
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                value_labels = accelerator.pad_across_processes(value_labels, dim=1, pad_index=tokenizer.pad_token_id)
            recovered_from_value = accelerator.gather(recovered_from_value).cpu().numpy()
            value_labels = accelerator.gather(value_labels).cpu().numpy()
            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                value_labels = np.where(value_labels != -100, value_labels, tokenizer.pad_token_id)
            decoded_value = tokenizer.batch_decode(recovered_from_value, skip_special_tokens=True)
            decoded_value_labels = tokenizer.batch_decode(value_labels, skip_special_tokens=True)
            decoded_value, decoded_value_labels = postprocess_text(decoded_value, decoded_value_labels)
            metric_value.add_batch(predictions=decoded_value, references=decoded_value_labels)

    return metric_key, metric_value, all_gen_ans


if __name__ == "__main__":
    main()
