import argparse
import copy
import json
import logging
import math
import os
import random

import datasets
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset, load_metric, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    set_seed,
    T5Tokenizer,
)
from transformers.utils.versions import require_version

from emat.t5 import T5WithKeyValueMemory
from transformers import T5Config
from emat.utils import load_jsonl, write_jsonl, verbalise_qa
from utils.utils import reduce_query_or_key_embeds, save_model, CATArgs, update_CAT_config_from_args, load_model, \
    get_key_value_encoder_inputs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

DATA_PATHS = {
    "PAQ-L1": "./data/cbqa_data/pretrain_data/PAQ_L1/PAQ_L1.filtered.jsonl",
    "data_for_debug": "./data/cbqa_data/pretrain_data/paq-l1-pretrain-dev-3000.jsonl"
}


def load_paq_data(args) -> DatasetDict:
    assert args.embed_data_name in DATA_PATHS.keys(), f"available dataset names: {DATA_PATHS.keys()}"
    data_path = DATA_PATHS[args.embed_data_name]
    return load_dataset("json", data_files=data_path)


@torch.no_grad()
def build_memory(model, tokenizer, output_dir=None, embed_key=False, embed_value=False, prefix="",
                 embed_as_fp16=False, key_reduce_method=None, data_path=None, data_to_embed=None,
                 max_source_length=None, padding=None, batch_size=1, allow_overlay_old_memory=False,
                 dump_memory=False, return_memory=False, separate_task=False, kvm_seg_n=-1,
                 disable_tqdm=False, reused_key_memory=None, collate_fn=None, normed_key_memory=True,
                 return_not_reduced_key=False, reused_not_reduced_key_memory=None, reused_value_memory=None,
                 num_workers=4, use_retrieval_adapter=False):
    torch.cuda.empty_cache()
    if data_to_embed is None:
        data_to_embed = load_dataset("json", data_files=data_path)["train"]

    if collate_fn is None:
        def collate_fn(examples):
            model_inputs = get_key_value_encoder_inputs(examples, separate_task, tokenizer, max_source_length,
                                                        prefix=prefix, only_return_key_inputs=not embed_value)
            return model_inputs

    qas_to_embed_dataloader = DataLoader(data_to_embed, batch_size=batch_size, num_workers=num_workers,
                                         collate_fn=collate_fn)

    key_memory: list = []
    value_memory: list = []
    not_reduced_key_memory = [] if return_not_reduced_key else None
    model.eval()

    key_cnt = 0
    for batch in tqdm(qas_to_embed_dataloader, disable=disable_tqdm):
        # for start_idx in tqdm(range(0, len(data_to_embed), batch_size), total=len(data_to_embed) // batch_size):
        #     batch_qas = data_to_embed[start_idx: start_idx + batch_size]
        #     batch = get_key_value_encoder_inputs(batch_qas, separate_task, tokenizer, max_source_length,
        #                                          prefix=prefix, only_return_key_inputs=True)
        with torch.no_grad():
            batch_keys = list(batch.keys())

            # for k in batch_keys:
            #     v = batch.pop(k)
            #     batch[k] = v.to(model.device)
            #     del v
            batch = {k: v.to(model.device) for k, v in batch.items()}

            embed_dict = model.wrapped_embed_kv(
                separate_task=separate_task,
                compute_key=embed_key,
                compute_value=embed_value,
                # key_input_ids=batch["key_input_ids"].to(model.device),
                # key_attention_mask=batch["key_attention_mask"].to(model.device),
                # value_input_ids=batch.get("key_input_ids", None).to(model.device),
                # value_attention_mask=batch.get("key_attention_mask", None).to(model.device),
                **batch
            )

            for k in batch_keys:
                del batch[k]

            key_embeds = embed_dict.get("normed_key_embeds") if normed_key_memory else embed_dict.get("key_embeds")
            value_embeds = embed_dict.get("value_embeds")
            if embed_key:
                key_embeds = reduce_query_or_key_embeds(key_embeds, key_reduce_method)
                if use_retrieval_adapter:
                    key_embeds = model.adapter(key_embeds)
                cur_key_num = key_embeds.shape[0]

        if embed_key:
            if embed_as_fp16:
                key_embeds = key_embeds.half()
            if reused_key_memory is not None:
                key_embeds = key_embeds.cpu()
                reused_key_memory[key_cnt: key_cnt + cur_key_num] = copy.deepcopy(key_embeds)
                del key_embeds
            else:
                key_memory.append(key_embeds.cpu())  # [batch_size, hidden_size]

            if return_not_reduced_key:
                not_normed_key_embeds = embed_dict["key_embeds"]
                if embed_as_fp16:
                    not_normed_key_embeds = not_normed_key_embeds.half()
                if reused_not_reduced_key_memory is not None:
                    not_normed_key_embeds = not_normed_key_embeds.cpu()
                    reused_not_reduced_key_memory[key_cnt: key_cnt + cur_key_num] = copy.deepcopy(not_normed_key_embeds)
                    del not_normed_key_embeds
                else:
                    not_reduced_key_memory.append(not_normed_key_embeds.cpu())

        if embed_value:
            if embed_as_fp16:
                value_embeds = value_embeds.half()
            if reused_value_memory is not None:
                value_embeds = value_embeds.cpu()
                reused_value_memory[key_cnt: key_cnt + cur_key_num] = copy.deepcopy(value_embeds)
                del value_embeds
            else:
                value_memory.append(value_embeds.cpu())  # [batch_size, value_nums, hidden_size]

        key_cnt += cur_key_num

    if reused_key_memory is None:
        if embed_key:
            assert sum(i.shape[0] for i in key_memory) == len(data_to_embed)
            if return_not_reduced_key:
                assert sum(i.shape[0] for i in not_reduced_key_memory) == len(data_to_embed)
        if embed_value:
            assert sum(i.shape[0] for i in value_memory) == len(data_to_embed)

    if dump_memory:
        assert reused_key_memory is None, "Not Implement when reused_key_memory is set."
        chunk_num = 128
        chunk_batch_size = math.ceil(len(key_memory) / chunk_num)
        if embed_key:
            logger.info("dump key")
            key_dir = os.path.join(output_dir, "key")
            os.makedirs(key_dir, exist_ok=allow_overlay_old_memory)
            save_num = 0
            for cid, start_idx in tqdm(enumerate(range(0, len(key_memory), chunk_batch_size)), leave=True):
                chunk_key_memory = torch.cat(key_memory[start_idx: start_idx + chunk_batch_size])
                torch.save(chunk_key_memory, os.path.join(key_dir, f"{cid}.key.pt"))
                save_num = save_num + chunk_key_memory.shape[0]
            assert save_num == len(data_to_embed), \
                f"saved key num is {save_num}, but example num is {len(data_to_embed)}"
        if embed_value:
            logger.info("dump value")
            value_dir = os.path.join(output_dir, "value")
            os.makedirs(value_dir, exist_ok=allow_overlay_old_memory)
            save_num = 0
            for cid, start_idx in tqdm(enumerate(range(0, len(value_memory), chunk_batch_size)), leave=True):
                chunk_value_memory = torch.cat(value_memory[start_idx: start_idx + chunk_batch_size])
                torch.save(chunk_value_memory, os.path.join(value_dir, f"{cid}.value.pt"))
                save_num = save_num + chunk_value_memory.shape[0]
            assert save_num == len(data_to_embed), \
                f"saved value num is {save_num}, but example num is {len(data_to_embed)}"

    if return_memory:
        if kvm_seg_n > 1:
            all_chunk_key_memory = []
            if embed_key:
                if reused_key_memory is not None:
                    logger.info(f"Split reused_key_memory into {kvm_seg_n} chunks.")
                    chunk_batch_size = math.ceil(len(reused_key_memory) / kvm_seg_n)
                    for start_idx in range(0, len(reused_key_memory), chunk_batch_size):
                        end_idx = min(len(reused_key_memory), start_idx + chunk_batch_size)
                        all_chunk_key_memory.append(reused_key_memory[start_idx:end_idx])
                else:
                    logger.info(f"Combining the keys into {kvm_seg_n} chunks.")
                    chunk_batch_size = math.ceil(len(key_memory) / kvm_seg_n)
                    for cid, start_idx in tqdm(enumerate(range(0, len(key_memory), chunk_batch_size)), leave=True):
                        chunk_key_memory = torch.cat(key_memory[start_idx: start_idx + chunk_batch_size])
                        all_chunk_key_memory.append(chunk_key_memory)
                assert len(all_chunk_key_memory) == kvm_seg_n

                # if return_not_reduced_key:
                #     not_reduced_key_memory = torch.emat(not_reduced_key_memory)

            if embed_value:
                value_memory = torch.cat(value_memory)

            return all_chunk_key_memory, value_memory

        else:
            if embed_key:
                if reused_key_memory is not None:
                    key_memory = reused_key_memory
                else:
                    logger.info(f"Combining the result.")
                    key_memory = torch.cat(key_memory)

                if return_not_reduced_key:
                    if reused_not_reduced_key_memory is not None:
                        not_reduced_key_memory = reused_not_reduced_key_memory
                    else:
                        not_reduced_key_memory = torch.cat(not_reduced_key_memory)

            if embed_value:
                if reused_value_memory is not None:
                    value_memory = reused_value_memory
                else:
                    value_memory = torch.cat(value_memory)
            if return_not_reduced_key:
                return key_memory, value_memory, not_reduced_key_memory
            else:
                return key_memory, value_memory


def main():
    # Parse the arguments
    cat_args = CATArgs(exp_type="build_kvm")
    args = cat_args.parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
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

    config, tokenizer, model = load_model(T5WithKeyValueMemory, args)
    model.cuda()
    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else True

    # Load the datasets
    data_to_embed = load_paq_data(args)["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(data_to_embed)), 3):
        logger.info(f"Sample {index} of the training set: {data_to_embed[index]}.")

    batch_size = args.per_device_train_batch_size
    logger.info("***** Building Key-Value Memory *****")
    logger.info(f"  Num examples = {len(data_to_embed)}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    # Only show the progress bar once on each machine.
    build_memory(model, tokenizer, output_dir=args.output_dir, embed_key=args.embed_key, embed_value=args.embed_value,
                 prefix=prefix, embed_as_fp16=args.embed_as_fp16, key_reduce_method=args.key_reduce_method,
                 data_path=None, data_to_embed=data_to_embed, max_source_length=args.max_source_length, padding=padding,
                 batch_size=batch_size, allow_overlay_old_memory=False, dump_memory=True, return_memory=False,
                 separate_task=args.separate_task)

    pretrain_args = json.load(open(os.path.join(args.model_name_or_path, "args.json")))
    dict_args = vars(args)
    dict_args["loaded_model_args"] = pretrain_args
    json.dump(pretrain_args, open(os.path.join(args.output_dir, "kvm_args.json"), 'w'), indent=4)


if __name__ == '__main__':
    main()
