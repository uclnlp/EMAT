import copy
import logging
import math
import os
import random
from itertools import chain
from typing import List, Dict, Optional
from collections import Counter
from rouge import Rouge
from emat.utils import write_jsonl
from utils.dr_utils import update_dialog_local_qas_to_retrieve, update_batch_inputs
from utils.utils import reduce_query_or_key_embeds
import datasets
import torch
import transformers
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler, set_seed
from utils.utils import save_model, load_model
from build_kvm import build_memory
from emat.t5 import T5WithKeyValueMemory
from kilt_dataset import DialogDataset
from nltk.translate.bleu_score import sentence_bleu
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn(
            "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False


def compute_batch_BLEU(references, candidates):
    def compute_BLEU(reference, candidate):
        b1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        b2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
        b3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
        b4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
        return b1, b2, b3, b4

    references = [[ref] for ref in references]

    bleu1_score = []
    bleu2_score = []
    bleu3_score = []
    bleu4_score = []
    for index in range(len(references)):
        bleu1, bleu2, bleu3, bleu4 = compute_BLEU(references[index], candidates[index])
        bleu1_score.append(bleu1)
        bleu2_score.append(bleu2)
        bleu3_score.append(bleu3)
        bleu4_score.append(bleu4)

    bleu1 = sum(bleu1_score) / len(bleu1_score)
    bleu2 = sum(bleu2_score) / len(bleu2_score)
    bleu3 = sum(bleu3_score) / len(bleu2_score)
    bleu4 = sum(bleu4_score) / len(bleu4_score)

    return {"bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4, }


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def max_score_over_ground_truths(fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


class DialogTrainer:

    def __init__(
            self,
            args,
            train_dataset: DialogDataset,
            dev_dataset: DialogDataset,
            test_dataset: DialogDataset,
            qas_to_retrieve: List[Dict],
            normed_answer_of_qas_to_ret,
    ):
        accelerator = Accelerator()
        logging.info(f"wandb {'available' if _has_wandb else 'unavailable'}")
        logger.info(accelerator.state)
        logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        if args.seed is not None:
            set_seed(args.seed)
        else:
            logging.info("Not set seed.")
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        if accelerator.is_local_main_process and _has_wandb:
            wandb.init(project=args.project_name, name=args.exp_name, dir=args.output_dir, config=vars(args))

        logging.info("loading model")
        config, tokenizer, self.model = load_model(T5WithKeyValueMemory, args)
        logging.info("Loading data.")

        if args.freeze_t5_params:
            logging.info("Freeze T5 parameters.")
            self.model.freeze_t5_params()

        if args.not_share_encoder and not args.update_kv_embeds:
            logging.info("Freeze kv-encoder parameters.")
            self.model.freeze_kv_encoder_params()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # reused_key_memory: pre-allocated memory to store full key_memory
        self.reused_key_memory = torch.zeros((len(qas_to_retrieve), self.model.model_dim),
                                             device="cpu", dtype=torch.float16)
        self.train_data_query_embeds = torch.zeros((len(train_dataset), self.model.model_dim),
                                                   device="cpu", dtype=torch.float16)
        self.key_memory: Optional[List[torch.tensor]] = None
        self.key_memory = []
        for start_idx in range(0, len(qas_to_retrieve), math.ceil(len(qas_to_retrieve) / args.kvm_seg_n)):
            self.key_memory.append(
                self.reused_key_memory[start_idx: start_idx + math.ceil(len(qas_to_retrieve) / args.kvm_seg_n)]
            )
        # logger.info(f"key num = {sum(len(i) for i in self.key_memory)}")

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.qas_to_retrieve = qas_to_retrieve
        self.prefix = args.source_prefix if args.source_prefix is not None else ""
        self.query_batch_size = args.query_batch_size
        # logger.info(f"PAQ-size: {len(self.qas_to_retrieve)}. PAQ's query batch size: {self.query_batch_size}.")
        self.normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret
        self.model = self.accelerator.prepare(self.model)

    @torch.no_grad()
    def update_key_memory(self, use_fp16_model=True):
        args = self.args
        if use_fp16_model:
            tmp_model = copy.deepcopy(self.model)
            tmp_model = tmp_model.half()
        else:
            tmp_model = self.model
        build_mem_batch_size = args.build_mem_batch_size
        tmp_model.eval()

        self.key_memory, _ = build_memory(
            tmp_model, self.tokenizer, embed_key=True, embed_value=False, prefix="question: ", embed_as_fp16=True,
            key_reduce_method=args.key_reduce_method, return_memory=True, dump_memory=False,
            data_to_embed=self.qas_to_retrieve, max_source_length=args.max_source_length, padding=True,
            batch_size=build_mem_batch_size, separate_task=True, kvm_seg_n=args.kvm_seg_n,
            reused_key_memory=self.reused_key_memory, num_workers=0
        )
        if type(self.key_memory) is not list:
            self.key_memory = [self.key_memory]
        del tmp_model

    @torch.no_grad()
    def update_local_qas(self, epoch, use_fp16_model=True):
        args = self.args
        if use_fp16_model:
            tmp_model = copy.deepcopy(self.model)
            tmp_model = tmp_model.half()
        else:
            tmp_model = self.model
        build_mem_batch_size = args.build_mem_batch_size
        tmp_model.eval()
        update_dialog_local_qas_to_retrieve(
            args, self.train_dataset, self.qas_to_retrieve, tmp_model, self.key_memory,
            self.normed_answer_of_qas_to_ret, train_data_query_embeds=self.train_data_query_embeds,
            build_mem_batch_size=build_mem_batch_size, query_batch_size=self.query_batch_size,
            local_size=args.local_size, pos_from_top=args.pos_from_top, neg_from_top=200
        )
        del tmp_model

    def train(self):
        args = self.args
        tokenizer = self.tokenizer
        num_workers = 5
        logging.info("Build Memory")
        self.update_key_memory()

        train_dataloader = self.train_dataset.get_dataloader(batch_size=args.per_device_train_batch_size,
                                                             shuffle=True, num_workers=num_workers)
        optimizer, train_dataloader = self.accelerator.prepare(self.optimizer, train_dataloader)

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
        total_batch_size = args.per_device_train_batch_size * self.accelerator.num_processes \
                           * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0

        best_score, patience = float("-inf"), args.early_stop_patience
        best_score_epoch = -1
        eval_times = 0
        select_mt = "f1" if self.train_dataset.dataset_name != "eli5_kilt" else "RougeL"

        for epoch in range(args.num_train_epochs):
            if args.qas_to_retrieve_from == "PAQ" and (epoch % 3 == 0):
                self.update_local_qas(epoch)
            elif args.qas_to_retrieve_from != "PAQ":
                self.update_local_qas(epoch)
            for step, batch in enumerate(train_dataloader):

                update_batch_inputs(args, batch, self.model)
                self.model.train()
                if args.match_weight > 0.0:
                    # Embed Positive Key and the Value to input.
                    embed_dict = self.model.wrapped_embed_kv(  # assert num_values > 1, otherwise set compute_value=True
                        separate_task=args.separate_task, compute_key=True, compute_value=False,
                        **batch.pop("positive_kv_inputs")
                    )
                    positive_key_embeds = embed_dict["normed_key_embeds"]
                    positive_key_embeds = reduce_query_or_key_embeds(positive_key_embeds, args.key_reduce_method)
                    # Embed Negative Key
                    embed_dict = self.model.wrapped_embed_kv(
                        separate_task=args.separate_task, compute_key=True, compute_value=False,
                        **batch.pop("negative_kv_inputs")
                    )
                    negative_key_embeds = embed_dict["normed_key_embeds"]
                    negative_key_embeds = reduce_query_or_key_embeds(negative_key_embeds, args.key_reduce_method)
                else:
                    negative_key_embeds, positive_key_embeds = None, None
                # Embed retrieved-Key-Value
                embed_dict = self.model.wrapped_embed_kv(
                    separate_task=args.separate_task, compute_key=True, compute_value=True,
                    **batch.pop("group_value_inputs")
                )
                key_embeds_of_value = embed_dict["key_embeds"]
                value_embeds = embed_dict["value_embeds"]
                bs = batch["query_input_ids"].shape[0]
                value_embeds = value_embeds.view(bs, args.num_values, args.prefix_length, -1)
                key_embeds_of_value = key_embeds_of_value.view(bs, args.num_values, -1, self.model.model_dim)

                loss_dict = self.model.compute_qa_loss(
                    input_ids=batch["history_input_ids"],
                    attention_mask=batch["history_attention_mask"],
                    labels=batch["labels"],
                    decoder_only_attend_on_prefix=False,
                    value_fusion_method=args.value_fusion_method,
                    encoder_outputs_are_key_or_value=False,
                    key_reduce_method=args.key_reduce_method,
                    positive_key_embeds=positive_key_embeds,
                    negative_key_embeds=negative_key_embeds,
                    value_embeds=value_embeds,
                    matching_targets=batch["matching_targets"],
                    key_embeds_of_value=key_embeds_of_value,
                    negative_mask=batch.get("negative_mask", None),
                )
                if args.match_weight > 0.0:
                    loss = loss_dict["gen_loss"]
                else:
                    loss = args.gen_weight * loss_dict["gen_loss"] + args.match_weight * loss_dict["match_loss"]
                loss_dict = {k: v.item() for k, v in loss_dict.items()}
                loss = loss / args.gradient_accumulation_steps
                self.accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    if args.match_weight > 0.0:
                        progress_bar.set_description(f"GL-[{loss_dict['gen_loss']:.5f}]")
                    else:
                        progress_bar.set_description(f"GL-[{loss_dict['gen_loss']:.5f}] "
                                                     f"RL-[{loss_dict['match_loss']:.5f}]")
                    completed_steps += 1
                    if self.accelerator.is_local_main_process and _has_wandb:
                        wandb.log({"loss": loss * args.gradient_accumulation_steps, "step": completed_steps})
                        wandb.log({"trainable_percentage": batch["trainable_percentage"][0].item(),
                                   "forward_step": completed_steps})
                        for k, v in loss_dict.items():
                            wandb.log({k: v, "step": completed_steps})

                    if args.eval_every_n_steps is not None and completed_steps % args.eval_every_n_steps == 0:
                        self.update_local_qas(epoch)
                        metric, _, _ = self.evaluate(dataset=self.dev_dataset, extend_mem_from="train")
                        for k, v in metric.items():
                            logger.info(f"eval_times {eval_times} eval - {k}: {v * 100:.5f}")
                        if _has_wandb:
                            for k, v in metric.items():
                                wandb.log({k: v * 100, "eval_times": eval_times})

                        if args.output_dir is not None:
                            cur_score = metric[select_mt]
                            if cur_score > best_score:
                                best_score = cur_score
                                best_score_epoch = epoch
                                save_model(self.model, os.path.join(args.output_dir, "best_ckpt"),
                                           self.accelerator, tokenizer=tokenizer, arguments=args)
                                patience = args.early_stop_patience
                            else:
                                patience -= 1
                                if patience <= 0:
                                    break

                if completed_steps >= args.max_train_steps:
                    break

            if args.output_dir is not None:
                save_model(self.model, os.path.join(args.output_dir, "latest_ckpt"),
                           self.accelerator, tokenizer=tokenizer, arguments=args)

            if args.update_kv_embeds:
                logging.info("Update Memory")
                self.update_key_memory()

            if args.do_eval and epoch % args.eval_freq == 0:
                # if args.do_eval: self.key_memory is up-to-date.
                metric, _, _ = self.evaluate(dataset=self.dev_dataset, extend_mem_from="train")
                for k, v in metric.items():
                    logger.info(f"epoch {epoch} eval - {k}: {v * 100:.5f}")
                if _has_wandb:
                    for k, v in metric.items():
                        wandb.log({k: v * 100, "epoch": epoch})

                if args.output_dir is not None:
                    cur_score = metric[select_mt]
                    if cur_score > best_score:
                        best_score = cur_score
                        best_f1_epoch = epoch
                        save_model(self.model, os.path.join(args.output_dir, "best_ckpt"),
                                   self.accelerator, tokenizer=tokenizer, arguments=args)
                        patience = args.early_stop_patience
                    else:
                        patience -= 1
                        if patience <= 0:
                            break

        logger.info(f"best_f1_dev: {best_score * 100:.5f}")
        logger.info(f"best_f1 epoch: {best_score_epoch}")
        if _has_wandb:
            wandb.log({"best_f1_dev": best_score * 100})

        # do-test
        best_model_state_dict = os.path.join(args.output_dir, "best_ckpt/pytorch_model.bin")
        metric, _, _ = self.evaluate(dataset=self.test_dataset, extend_mem_from="train_dev",
                                     update_key_memory=True, ckpt_load_path=best_model_state_dict)

        if self.accelerator.is_local_main_process:
            for k, v in metric.items():
                logger.info(f"test - {k}: {v * 100:.5f}")
            if _has_wandb:
                for k, v in metric.items():
                    wandb.log({f"test_{k}": v * 100})

    @torch.no_grad()
    def evaluate(self, dataset: DialogDataset = None, extend_mem_from="", update_key_memory=False, ckpt_load_path=None,
                 gen_kwargs=None):
        # not implement correctly in multi-GPUs
        tokenizer = self.tokenizer
        args = self.args
        self.model.eval()
        torch.cuda.empty_cache()

        if ckpt_load_path is not None:
            if args.update_kv_embeds:
                assert update_key_memory is True
            self.model.load_state_dict(torch.load(ckpt_load_path), strict=True)

        assert type(self.key_memory) == list
        if update_key_memory:
            logging.info("Update Memory")
            self.update_key_memory()

        dataloader = dataset.get_query_dataloader(batch_size=args.per_device_eval_batch_size,
                                                  shuffle=False, num_workers=1, add_history=True)
        dataloader = self.accelerator.prepare(dataloader)

        if gen_kwargs is None:
            gen_kwargs = {"max_length": args.max_target_length,
                          "num_beams": args.num_beams, }

        torch.cuda.empty_cache()

        all_retrieved_qas = []
        all_gen_response = []
        for batch in tqdm(dataloader):
            embed_dict = self.model.CAT_embed_q(
                input_ids=batch["query_input_ids"],
                attention_mask=batch["query_attention_mask"],
                compute_key=True, compute_value=False
            )
            query_embeds = embed_dict["normed_key_embeds"]
            query_embeds = reduce_query_or_key_embeds(query_embeds, args.key_reduce_method)
            query_embeds = query_embeds.half()

            # calculate topk in each chunk -> combine all-topk -> select final topk
            chunk_top_scores = []
            chunk_top_indices = []
            idx_shift = 0
            for chunk_key_memory in self.key_memory:
                chunk_key_memory_cuda = chunk_key_memory.cuda()
                chunk_topk = torch.mm(query_embeds, chunk_key_memory_cuda.t()).topk(50, dim=1)
                chunk_top_scores.append(chunk_topk.values)  # chunk_topk.scores: [query_batch, local_size]
                chunk_top_indices.append(chunk_topk.indices + idx_shift)
                idx_shift += len(chunk_key_memory)
                del chunk_key_memory_cuda
                torch.cuda.empty_cache()
            chunk_top_scores = torch.cat(chunk_top_scores, dim=1)  # q_batch, local_size*seg_n
            chunk_top_indices = torch.cat(chunk_top_indices, dim=1)  # q_batch, local_size*seg_n
            topk = chunk_top_scores.topk(args.num_values, dim=1)  # q_batch, local_size
            top_indices_indices = topk.indices
            top_indices = []
            for cur_indices_indices, cur_indices in zip(top_indices_indices, chunk_top_indices):
                top_indices.append([cur_indices[idx] for idx in cur_indices_indices])
            readout_qas = [[self.qas_to_retrieve[idx] for idx in indices] for indices in top_indices]

            value_qas = []
            for qas in readout_qas:
                selected_qas = qas[:args.num_values]
                if not args.values_with_order:
                    random.shuffle(selected_qas)
                value_qas.append(selected_qas)
            all_retrieved_qas += readout_qas

            squeezed_value_qas = list(chain(*value_qas))
            retrieved_qas_inputs = dataset.get_qa_key_value_inputs(squeezed_value_qas, only_return_key_inputs=False)
            embed_dict = self.model.wrapped_embed_kv(separate_task=args.separate_task, compute_key=True,
                                                     compute_value=True, **retrieved_qas_inputs)
            value_embeds = embed_dict["value_embeds"]
            key_embeds_of_value = embed_dict["key_embeds"]
            cur_batch_size = query_embeds.shape[0]
            value_embeds = value_embeds.view(cur_batch_size, args.num_values, args.prefix_length, -1)
            key_embeds_of_value = key_embeds_of_value.view(cur_batch_size, args.num_values, -1, self.model.model_dim)
            encoder_outputs = self.model.encoder(
                input_ids=batch["history_input_ids"],
                attention_mask=batch["history_attention_mask"],
                return_dict=True,
                value_embeds=value_embeds,
                readout_top_k=-1,
                key_reduce_method=args.key_reduce_method,
                value_fusion_method=args.value_fusion_method,
                key_embeds_of_value=key_embeds_of_value
            )
            generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                encoder_outputs=encoder_outputs,
                encoder_outputs_are_key_or_value=False,
                decoder_only_attend_on_prefix=False,
                attention_mask=batch["history_attention_mask"].to(self.model.device),
                value_fusion_method=args.value_fusion_method,
                **gen_kwargs,
            )
            decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_tokens = [ans.strip() for ans in decoded_tokens]
            all_gen_response += decoded_tokens

        torch.cuda.empty_cache()

        if "normalized_response" in dataset.data[0].keys():
            if dataset.dataset_name == "wow_kilt":
                reference = [item["normalized_response"] for item in dataset.data]  # only one target
            else:
                reference = [item["candidate_responses"] for item in dataset.data]  # multi candidates
        elif "response" in dataset.data[0].keys():
            reference = [DialogDataset.normalize_answer(item["response"]) for item in dataset.data]
        else:
            reference = None

        if reference is not None:
            assert len(all_gen_response) == len(reference)
            metrics = dict()
            if dataset.dataset_name == "eli5_kilt":
                avg_rougel = sum([max_score_over_ground_truths(rougel_score, pred, ref) for pred, ref in
                                  zip(all_gen_response, reference)]) / len(all_gen_response)
                metrics.update({"RougeL": avg_rougel})
                reference = [[DialogDataset.normalize_answer(s) for s in cands] for cands in reference]
                all_gen_response = [DialogDataset.normalize_answer(s) for s in all_gen_response]
                avg_f1 = sum([max_score_over_ground_truths(f1_score, pred, ref) for pred, ref in
                              zip(all_gen_response, reference)]) / len(all_gen_response)
            else:
                reference = [DialogDataset.normalize_answer(s) for s in reference]
                all_gen_response = [DialogDataset.normalize_answer(s) for s in all_gen_response]
                bleu_scores = compute_batch_BLEU(reference, all_gen_response)
                metrics.update(bleu_scores)
                avg_f1 = sum([f1_score(pred, ref) for pred, ref in
                              zip(all_gen_response, reference)]) / len(all_gen_response)
            metrics.update({"f1": avg_f1})
        else:
            metrics = dict()
            results = []
            assert len(dataset.data) == len(all_gen_response) == len(all_retrieved_qas)
            for input_item, pred, retrieved in zip(dataset.data, all_gen_response, all_retrieved_qas):
                results.append({
                    "id": input_item["id"],
                    "input": self.tokenizer.decode(input_item["input_ids"]),
                    "query": self.tokenizer.decode(input_item["query_ids"]),
                    "output": {"answer": pred, "provenance": [{"wikipedia_id": "12904"}]},
                    "retrieved_qas": [f"question: {qa['question']} answer: {qa['answer'][0]}" for qa in retrieved]
                })
            dump_path = os.path.dirname(ckpt_load_path)
            dump_path = os.path.join(dump_path, f"{time.strftime('%d %H-%M')}_predict_result.json")
            # save_path = os.path.join(args.output_dir, "kilt_test_predict.jsonl")
            write_jsonl(results, dump_path)

        return metrics, all_retrieved_qas, all_gen_response


@torch.no_grad()
def kilt_generate(model, tokenizer, embedding_index, key_memory, value_memory, dataset: DialogDataset,
             qas_to_retrieve, inference_batch_size, gen_kwargs):
    model.eval()

    dataloader = dataset.get_query_dataloader(batch_size=inference_batch_size, shuffle=False,
                                              num_workers=1, add_history=True)
    all_retrieved_qas = []
    all_gen_response = []

    for batch in dataloader:
        embed_dict = model.CAT_embed_q(
            input_ids=batch["query_input_ids"].cuda(),
            attention_mask=batch["query_attention_mask"].cuda(),
            compute_key=True, compute_value=False
        )
        query_embeds = embed_dict["normed_key_embeds"]
        query_embeds = reduce_query_or_key_embeds(query_embeds, "avg")
        query_embeds = query_embeds.half()
        bs = len(batch["query_input_ids"])
        # calculate topk in each chunk -> combine all-topk -> select final topk
        chunk_top_scores = []
        chunk_top_indices = []
        idx_shift = 0
        if type(embedding_index) != list:
            embedding_index = [embedding_index]
        for chunk_key_memory in embedding_index:
            chunk_key_memory_cuda = chunk_key_memory.cuda()
            chunk_topk = torch.mm(query_embeds, chunk_key_memory_cuda.t()).topk(50, dim=1)
            chunk_top_scores.append(chunk_topk.values)  # chunk_topk.scores: [query_batch, local_size]
            chunk_top_indices.append(chunk_topk.indices + idx_shift)
            idx_shift += len(chunk_key_memory)
            del chunk_key_memory_cuda
            torch.cuda.empty_cache()
        chunk_top_scores = torch.cat(chunk_top_scores, dim=1)  # q_batch, local_size*seg_n
        chunk_top_indices = torch.cat(chunk_top_indices, dim=1)  # q_batch, local_size*seg_n
        topk = chunk_top_scores.topk(model.encoder.num_values, dim=1)  # q_batch, local_size
        top_indices_indices = topk.indices
        top_indices = []
        for cur_indices_indices, cur_indices in zip(top_indices_indices, chunk_top_indices):
            top_indices.append([cur_indices[idx] for idx in cur_indices_indices])
        readout_qas = [[qas_to_retrieve[idx] for idx in indices] for indices in top_indices]

        value_qas = []
        for qas in readout_qas:
            selected_qas = qas[:model.encoder.num_values]
            value_qas.append(selected_qas)
        all_retrieved_qas += readout_qas

        top_indices = torch.tensor(top_indices)
        memory_size, hidden_num, hidden_size = value_memory.shape
        value_embeds = torch.index_select(value_memory, 0, top_indices.view(-1)).float().cuda()
        value_embeds = value_embeds.view(bs, model.encoder.num_values, hidden_num, hidden_size)
        key_embeds_of_value = torch.index_select(key_memory, 0, top_indices.view(-1)).float().cuda()
        key_embeds_of_value = key_embeds_of_value.view(bs, model.encoder.num_values, hidden_num, hidden_size)
        encoder_outputs = model.encoder(
            input_ids=batch["history_input_ids"].cuda(),
            attention_mask=batch["history_attention_mask"].cuda(),
            return_dict=True,
            value_embeds=value_embeds,
            readout_top_k=-1,
            key_reduce_method="avg",
            value_fusion_method=model.encoder.value_fusion_method,
            key_embeds_of_value=key_embeds_of_value
        )
        generated_tokens = model.generate(
            encoder_outputs=encoder_outputs,
            encoder_outputs_are_key_or_value=False,
            decoder_only_attend_on_prefix=False,
            attention_mask=batch["history_attention_mask"].to(model.device),
            value_fusion_method=model.encoder.value_fusion_method,
            **gen_kwargs,
        )
        decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_tokens = [ans.strip() for ans in decoded_tokens]
        all_gen_response += decoded_tokens

    return all_retrieved_qas, all_gen_response


if __name__ == '__main__':
    pass
