import copy
import logging
import math
import os
import random
from itertools import chain
from typing import List, Dict, Optional
from emat.evaluation.eval_retriever import eval_retriever, eval_generation_em
from utils.dr_utils import update_local_qas_to_retrieve, update_batch_inputs, rank_exist_local_qas
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
from qa_dataset import QADataset

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


class QATrainer:

    def __init__(
            self,
            args,
            train_dataset: QADataset,
            dev_dataset: QADataset,
            test_dataset: QADataset,
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
        logging.info("Loading model.")
        logging.info(f"model params: {self.model.num_parameters()}")
        if args.freeze_t5_params:
            logging.info("Freeze T5 parameters.")
            self.model.freeze_t5_params()
        if args.only_train_adapter:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.adapter.parameters():
                param.requires_grad = True

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
        logger.info(f"key num = {sum(len(i) for i in self.key_memory)}")

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.qas_to_retrieve = qas_to_retrieve
        self.prefix = args.source_prefix if args.source_prefix is not None else ""
        assert self.prefix == "question: "
        # self.query_batch_size = 550 if args.kvm_seg_n > 2 else 256
        # self.query_batch_size = 1024 if args.kvm_seg_n >= 4 else self.query_batch_size
        # self.query_batch_size = 3000 if args.kvm_seg_n >= 7 else self.query_batch_size
        # # if len(self.qas_to_retrieve) < 20000000:
        # self.query_batch_size = 512
        self.query_batch_size = args.query_batch_size
        logger.info(f"PAQ-size: {len(self.qas_to_retrieve)}. PAQ's query batch size: {self.query_batch_size}.")
        self.normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret
        self.model = self.accelerator.prepare(self.model)

    @torch.no_grad()
    def update_key_memory(self, use_fp16_model=True, use_retrieval_adapter=False):
        args = self.args
        if use_fp16_model:
            tmp_model = copy.deepcopy(self.model)
            tmp_model = tmp_model.half()
        else:
            tmp_model = self.model
        build_mem_batch_size = args.build_mem_batch_size
        tmp_model.eval()
        self.key_memory, _ = build_memory(
            tmp_model, self.tokenizer, embed_key=True, embed_value=False, prefix=self.prefix, embed_as_fp16=True,
            key_reduce_method=args.key_reduce_method, return_memory=True, dump_memory=False,
            data_to_embed=self.qas_to_retrieve, max_source_length=args.max_source_length, padding=True,
            batch_size=build_mem_batch_size, separate_task=True, kvm_seg_n=args.kvm_seg_n,
            reused_key_memory=self.reused_key_memory, use_retrieval_adapter=use_retrieval_adapter
        )
        if type(self.key_memory) is not list:
            self.key_memory = [self.key_memory]
        del tmp_model

    @torch.no_grad()
    def update_local_qas(self, epoch, use_fp16_model=True, use_retrieval_adapter=False):
        args = self.args
        if use_fp16_model:
            tmp_model = copy.deepcopy(self.model)
            tmp_model = tmp_model.half()
        else:
            tmp_model = self.model
        build_mem_batch_size = args.build_mem_batch_size
        tmp_model.eval()
        if args.update_kv_embeds and args.update_local_qas and epoch >= args.repaq_supervision_epoch:
            update_local_qas_to_retrieve(
                args, self.train_dataset, self.qas_to_retrieve, tmp_model, self.key_memory,
                self.normed_answer_of_qas_to_ret, train_data_query_embeds=self.train_data_query_embeds,
                build_mem_batch_size=build_mem_batch_size, query_batch_size=self.query_batch_size,
                local_size=args.local_size, pos_from_top=args.pos_from_top, neg_from_top=200,
                use_retrieval_adapter=use_retrieval_adapter
            )
        elif args.only_rank_exists_local_qa:
            logging.warning("Do not use!")
            embed_local_qas_batch_size = (build_mem_batch_size //
                                          len(self.train_dataset.data[0]["local_qas"]) + 1) * 2
            rank_exist_local_qas(args, self.train_dataset, self.qas_to_retrieve, tmp_model,
                                 self.normed_answer_of_qas_to_ret, build_mem_batch_size=build_mem_batch_size,
                                 train_data_query_embeds=self.train_data_query_embeds,
                                 embed_local_qas_batch_size=embed_local_qas_batch_size,
                                 local_size=args.local_size, pos_from_top=args.pos_from_top, neg_from_top=200,
                                 accelerator=self.accelerator)
        del tmp_model

    def train(self):
        args = self.args
        tokenizer = self.tokenizer
        num_workers = 5
        if args.update_kv_embeds and not args.only_rank_exists_local_qa:
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
        best_hit_at_1, best_em, patience = None, None, args.early_stop_patience

        for epoch in range(args.num_train_epochs):
            use_adapter_to_select_positive = epoch >= args.use_adapter_to_select_positive_after_k_epoch
            if args.only_train_adapter:
                if epoch == 0:
                    self.update_local_qas(epoch)
                    # convert original key memory with high dim to low dim through adapter
                    qas_num = self.reused_key_memory.shape[0]
                    train_qas_num = len(self.train_dataset)
                    dim = args.adapter_out_dim
                    self.reused_key_memory = torch.zeros((qas_num, dim), device="cpu", dtype=torch.float16)
                    self.train_data_query_embeds = torch.zeros((train_qas_num, dim), device="cpu", dtype=torch.float16)
                    self.key_memory: Optional[List[torch.tensor]] = None
                    self.key_memory = []
                    for start_idx in range(0, qas_num, math.ceil(qas_num / args.kvm_seg_n)):
                        self.key_memory.append(
                            self.reused_key_memory[start_idx: start_idx + math.ceil(qas_num / args.kvm_seg_n)]
                        )
                    self.update_key_memory(use_retrieval_adapter=True)
                elif use_adapter_to_select_positive:
                    self.update_local_qas(epoch, use_retrieval_adapter=True)
            else:
                if args.qas_to_retrieve_from == "PAQ" and (epoch % 3 == 0):
                    self.update_local_qas(epoch)
                elif args.qas_to_retrieve_from != "PAQ":
                    self.update_local_qas(epoch)
            for step, batch in enumerate(train_dataloader):

                update_batch_inputs(args, batch, self.model,
                                    use_adapter_to_select_positive=use_adapter_to_select_positive)
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
                    input_ids=batch["query_input_ids"],
                    attention_mask=batch["query_attention_mask"],
                    labels=batch["labels"],
                    decoder_only_attend_on_prefix=args.decoder_only_attend_on_prefix,
                    value_fusion_method=args.value_fusion_method,
                    encoder_outputs_are_key_or_value=False,
                    key_reduce_method=args.key_reduce_method,
                    positive_key_embeds=positive_key_embeds,
                    negative_key_embeds=negative_key_embeds,
                    value_embeds=value_embeds,
                    matching_targets=batch["matching_targets"],
                    key_embeds_of_value=key_embeds_of_value,
                    negative_mask=batch.get("negative_mask", None),
                    only_train_adapter=args.only_train_adapter
                )
                if args.match_weight > 0.0:
                    if epoch >= args.only_key_matching_n_epoch:
                        loss = args.gen_weight * loss_dict["gen_loss"] + args.match_weight * loss_dict["match_loss"]
                    else:
                        loss = args.match_weight * loss_dict["match_loss"]
                else:
                    loss = loss_dict["gen_loss"]
                loss_dict = {k: v.item() for k, v in loss_dict.items()}
                loss = loss / args.gradient_accumulation_steps
                self.accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    if self.accelerator.is_local_main_process and _has_wandb:
                        wandb.log({"loss": loss * args.gradient_accumulation_steps, "step": completed_steps})
                        wandb.log({"trainable_percentage": batch["trainable_percentage"][0].item(),
                                   "forward_step": completed_steps})
                        for k, v in loss_dict.items():
                            wandb.log({k: v, "step": completed_steps})

                if completed_steps >= args.max_train_steps:
                    break

            if args.output_dir is not None:
                save_model(self.model, os.path.join(args.output_dir, "latest_ckpt"),
                           self.accelerator, tokenizer=tokenizer, arguments=args)

            if (args.update_kv_embeds and not args.only_rank_exists_local_qa) or args.do_eval:
                logging.info("Update Memory")
                self.update_key_memory(use_retrieval_adapter=args.only_train_adapter)

            if args.do_eval and epoch % args.eval_freq == 0:
                # if args.do_eval: self.key_memory is up-to-date.
                em_score, matching_metric, _, _ = self.evaluate(dataset=self.dev_dataset, extend_mem_from="train",
                                                                use_retrieval_adapter=args.only_train_adapter)
                logger.info(f"epoch {epoch} eval - EM: {em_score:.3f}")
                if self.accelerator.is_local_main_process and _has_wandb:
                    wandb.log({"em_dev": em_score, "epoch": epoch})
                    for k, v in matching_metric.items():
                        wandb.log({f"{k}": v * 100, "epoch": epoch})

                if args.output_dir is not None:
                    if best_hit_at_1 is None or matching_metric["hit@1"] * 100 > best_hit_at_1:
                        best_hit_at_1 = matching_metric["hit@1"] * 100
                    if best_em is None or em_score > best_em:
                        best_em = em_score
                        save_model(self.model, os.path.join(args.output_dir, "best_ckpt"),
                                   self.accelerator, tokenizer=tokenizer, arguments=args)
                        patience = args.early_stop_patience
                    else:
                        patience -= 1
                        if patience <= 0:
                            break

        if best_em is not None:  # Log the best dev EM score
            logger.info(f"best_em_dev: {best_em}")
            if _has_wandb:
                wandb.log({"best_em_dev": best_em})
        if best_hit_at_1 is not None:
            logger.info(f"best_hit@1_dev: {best_hit_at_1}")
            if _has_wandb:
                wandb.log({"best_hit@1_dev": best_hit_at_1})

        # do-test
        best_model_state_dict = os.path.join(args.output_dir, "best_ckpt/pytorch_model.bin")
        em_score, matching_metric, _, _ = self.evaluate(dataset=self.test_dataset, extend_mem_from="train_dev",
                                                        update_key_memory=True, ckpt_load_path=best_model_state_dict,
                                                        use_retrieval_adapter=args.only_train_adapter)

        if self.accelerator.is_local_main_process:
            logger.info(f"em_test: {em_score:.3f}")
            for k, v in matching_metric.items():
                logger.info(f"test_{k}: {v}")
            if _has_wandb:
                wandb.log({"em_test": em_score})
                for k, v in matching_metric.items():
                    wandb.log({f"test_{k}": v})

    @torch.no_grad()
    def evaluate(self, dataset: QADataset = None, extend_mem_from="", update_key_memory=False, ckpt_load_path=None,
                 use_retrieval_adapter=False):
        # not implement correctly in multi-GPUs.
        tokenizer = self.tokenizer
        args = self.args
        self.model.eval()
        torch.cuda.empty_cache()

        assert extend_mem_from in ["train", "train_dev"]
        if ckpt_load_path is not None:
            assert update_key_memory is True
            loaded_state_dict = torch.load(ckpt_load_path)
            load_info = self.model.load_state_dict(loaded_state_dict, strict=False)
            logging.info(f"{load_info}")

        assert type(self.key_memory) == list
        original_key_length = sum(len(k) for k in self.key_memory)
        if update_key_memory:
            logging.info("Update Memory")
            self.update_key_memory(use_retrieval_adapter=use_retrieval_adapter)

        extend_length = 0
        last_chunk_memory = self.key_memory[-1]
        qas_to_retrieve_eval = self.qas_to_retrieve

        tmp_model = copy.deepcopy(self.model)
        if args.kvm_fp16:
            tmp_model = tmp_model.half()

        logging.info("Build train data memory to retrieve.")
        if args.qa_data_name == "tq":
            build_query_batch_size = 256
        else:
            build_query_batch_size = args.build_mem_batch_size
        if "train" in extend_mem_from:
            train_qas_key_memory, _ = build_memory(
                tmp_model, tokenizer, embed_key=True, embed_value=False, prefix=self.prefix, embed_as_fp16=True,
                key_reduce_method=args.key_reduce_method, return_memory=True, dump_memory=False, kvm_seg_n=-1,
                data_to_embed=self.train_dataset.data, max_source_length=args.max_source_length, padding=True,
                batch_size=build_query_batch_size, separate_task=args.separate_task, reused_key_memory=None,
                use_retrieval_adapter=use_retrieval_adapter
            )
            extend_length = extend_length + len(train_qas_key_memory)
            last_chunk_memory = torch.cat((last_chunk_memory, train_qas_key_memory))  # extend in the last chunk
            qas_to_retrieve_eval = qas_to_retrieve_eval + self.train_dataset.data

        if "dev" in extend_mem_from:
            logging.info("Build dev data memory to retrieve.")
            dev_qas_key_memory, _ = build_memory(
                tmp_model, tokenizer, embed_key=True, embed_value=False, prefix=self.prefix, embed_as_fp16=True,
                key_reduce_method=args.key_reduce_method, return_memory=True, dump_memory=False, kvm_seg_n=-1,
                data_to_embed=self.dev_dataset.data, max_source_length=args.max_source_length, padding=True,
                batch_size=build_query_batch_size, separate_task=args.separate_task, reused_key_memory=None,
                use_retrieval_adapter=use_retrieval_adapter
            )
            extend_length = extend_length + len(dev_qas_key_memory)
            last_chunk_memory = torch.cat((last_chunk_memory, dev_qas_key_memory))  # extend in the last chunk
            qas_to_retrieve_eval = qas_to_retrieve_eval + self.dev_dataset.data
        del tmp_model

        key_memory_eval = self.key_memory[:-1] + [last_chunk_memory]
        key_nums_eval = sum(len(k) for k in key_memory_eval)
        assert key_nums_eval == len(qas_to_retrieve_eval)

        # if use_retrieval_adapter:
        #     low_dim_key = []
        #     while len(key_memory_eval) > 0:
        #         chunk_key = key_memory_eval.pop(0)
        #         chunk_low_dim_key = []
        #         for start_idx in range(len(chunk_key)):
        #             chunk_low_dim_key.append(self.model.adapter(chunk_key[start_idx:start_idx + 512]))
        #         del chunk_key
        #         low_dim_key.append(torch.cat(chunk_low_dim_key))
        #     key_memory_eval = low_dim_key

        dataloader = dataset.get_query_dataloader(batch_size=args.per_device_eval_batch_size,
                                                  shuffle=False, num_workers=1)
        dataloader = self.accelerator.prepare(dataloader)

        gen_kwargs = {"max_length": args.max_target_length,
                      "num_beams": args.num_beams, }

        torch.cuda.empty_cache()

        all_retrieved_qas = []
        all_gen_ans = []
        for batch in tqdm(dataloader):
            embed_dict = self.model.CAT_embed_q(
                input_ids=batch["query_input_ids"],
                attention_mask=batch["query_attention_mask"],
                compute_key=True, compute_value=False
            )
            query_embeds = embed_dict["normed_key_embeds"]
            query_embeds = reduce_query_or_key_embeds(query_embeds, args.key_reduce_method)
            if use_retrieval_adapter:
                query_embeds = self.model.adapter(query_embeds)
            query_embeds = query_embeds.half()

            if key_nums_eval > 20000000:
                # if scale is large: calculate topk in each chunk -> combine all-topk -> select final topk
                chunk_top_scores = []
                chunk_top_indices = []
                idx_shift = 0
                for chunk_key_memory in key_memory_eval:
                    chunk_key_memory_cuda = chunk_key_memory.cuda()
                    chunk_topk = torch.mm(query_embeds, chunk_key_memory_cuda.t()).topk(50, dim=1)
                    chunk_top_scores.append(chunk_topk.values)  # chunk_topk.scores: [query_batch, local_size]
                    chunk_top_indices.append(chunk_topk.indices + idx_shift)
                    idx_shift += len(chunk_key_memory)
                    del chunk_key_memory_cuda
                    torch.cuda.empty_cache()
                chunk_top_scores = torch.cat(chunk_top_scores, dim=1)  # q_batch, local_size*seg_n
                chunk_top_indices = torch.cat(chunk_top_indices, dim=1)  # q_batch, local_size*seg_n
                topk = chunk_top_scores.topk(50, dim=1)  # q_batch, local_size
                top_indices_indices = topk.indices
                top_indices = []
                for cur_indices_indices, cur_indices in zip(top_indices_indices, chunk_top_indices):
                    top_indices.append([cur_indices[idx] for idx in cur_indices_indices])
                readout_qas = [[qas_to_retrieve_eval[idx] for idx in indices] for indices in top_indices]
            else:
                all_chunk_scores = []
                for chunk_key_memory in key_memory_eval:
                    chunk_key_memory_cuda = chunk_key_memory.cuda()
                    chunk_scores = torch.mm(query_embeds, chunk_key_memory_cuda.t())  # query_batch
                    all_chunk_scores.append(chunk_scores)
                    del chunk_key_memory_cuda
                scores = torch.cat(all_chunk_scores, dim=1)
                top_indices = scores.topk(50, dim=1).indices.tolist()
                readout_qas = [[qas_to_retrieve_eval[idx] for idx in indices] for indices in top_indices]
            value_qas = []
            for qas in readout_qas:
                selected_qas = qas[:args.num_values]
                if not args.values_with_order:
                    random.shuffle(selected_qas)
                value_qas.append(selected_qas)
            all_retrieved_qas += readout_qas

            squeezed_value_qas = list(chain(*value_qas))

            retrieved_qas_inputs = dataset.get_key_value_inputs(squeezed_value_qas, only_return_key_inputs=False)
            embed_dict = self.model.wrapped_embed_kv(separate_task=args.separate_task, compute_key=True,
                                                     compute_value=True, **retrieved_qas_inputs)
            value_embeds = embed_dict["value_embeds"]
            key_embeds_of_value = embed_dict["key_embeds"]
            cur_batch_size = query_embeds.shape[0]
            value_embeds = value_embeds.view(cur_batch_size, args.num_values, args.prefix_length, -1)
            key_embeds_of_value = key_embeds_of_value.view(cur_batch_size, args.num_values, -1, self.model.model_dim)
            encoder_outputs = self.model.encoder(
                input_ids=batch["query_input_ids"],
                attention_mask=batch["query_attention_mask"],
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
                decoder_only_attend_on_prefix=args.decoder_only_attend_on_prefix,
                attention_mask=batch["query_attention_mask"].to(self.model.device),
                value_fusion_method=args.value_fusion_method,
                **gen_kwargs,
            )
            generated_tokens = self.accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
            decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_tokens = [ans.strip() for ans in decoded_tokens]
            all_gen_ans += decoded_tokens

        torch.cuda.empty_cache()

        matching_metric = eval_retriever(dataset.data, all_retrieved_qas, "1,2,3,4,5,10,50")
        em_score = eval_generation_em(dataset.data, all_gen_ans) * 100

        assert original_key_length == sum(len(k) for k in self.key_memory)
        assert original_key_length == len(self.qas_to_retrieve)

        return em_score, matching_metric, all_retrieved_qas, all_gen_ans


if __name__ == '__main__':
    pass
