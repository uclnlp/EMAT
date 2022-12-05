import copy
from functools import partial
import random
from typing import List
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from build_kvm import build_memory
from emat.evaluation.eval_retriever import eval_retriever
from emat.evaluation.exact_match import normalize_answer
from kilt_dataset import DialogDataset
from qa_dataset import QADataset
from utils.utils import reduce_query_or_key_embeds
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# not used in QA
def query_collate_fn(batch, tokenizer=None, dataset=None):
    for item in batch:
        item.update(dataset.get_base_input(item))
    query_input_ids = [item["input_ids"] for item in batch]
    query_input_ids = pad_sequence(query_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    query_attention_mask = (query_input_ids != tokenizer.pad_token_id).long()
    return {"query_input_ids": query_input_ids, "query_attention_mask": query_attention_mask}


# not used in QA
def kvm_collate_fn(batch, tokenizer=None, train_dataset=None):
    for item in batch:
        item.update(train_dataset.get_base_input(item))
    key_input_ids = [item["input_ids"] for item in batch]
    key_input_ids = pad_sequence(key_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    key_attention_mask = (key_input_ids != tokenizer.pad_token_id).long()
    value_input_ids = [item["target_as_input_ids"] for item in batch]
    value_input_ids = pad_sequence(value_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    value_attention_mask = (value_input_ids != tokenizer.pad_token_id).long()
    return {"key_input_ids": key_input_ids, "key_attention_mask": key_attention_mask,
            "value_input_ids": value_input_ids, "value_attention_mask": value_attention_mask}


# not used in QA
@torch.no_grad()
def build_key_memory(model, tokenizer, train_dataset, data_to_retrieve_list: List[List[dict]], args,
                     build_memory_batch_size=2560) -> List[torch.tensor]:
    # key norm layer only used in key-mathing (retrieval)
    # if not train key-matching, no gradient to key norm layer.
    use_normed_key = True if args.key_matching_weight > 0.0 else False

    key_memory = []
    for data_to_retrieve in data_to_retrieve_list:
        cur_km, _ = build_memory(model, tokenizer, embed_key=True, embed_value=False, embed_as_fp16=True,
                                 key_reduce_method=args.key_reduce_method, data_to_embed=data_to_retrieve,
                                 batch_size=build_memory_batch_size, return_memory=True, separate_task=True,
                                 collate_fn=partial(kvm_collate_fn, tokenizer=tokenizer, train_dataset=train_dataset),
                                 normed_key_memory=use_normed_key)
        key_memory.append(cur_km)

    return key_memory


# not used in QA
@torch.no_grad()
def build_dataset_query(model, tokenizer, dataset, args, encode_query_batch_size=2560) -> torch.tensor:
    dataset_query_embeds = []
    query_to_embed_dataloader = DataLoader(dataset.data, batch_size=encode_query_batch_size, num_workers=16,
                                           collate_fn=partial(query_collate_fn, tokenizer=tokenizer, dataset=dataset))
    for query_batch_input in tqdm(query_to_embed_dataloader, desc="embed_query"):
        embed_dict = model.CAT_embed_q(
            input_ids=query_batch_input["query_input_ids"].to(model.device),
            attention_mask=query_batch_input["query_attention_mask"].to(model.device),
            compute_key=True, compute_value=False
        )
        use_normed_query = True if args.key_matching_weight > 0.0 else False
        query_embeds = embed_dict["normed_key_embeds"] if use_normed_query else embed_dict["key_embeds"]
        query_embeds = reduce_query_or_key_embeds(query_embeds, args.key_reduce_method)
        query_embeds = query_embeds.half().cpu()
        dataset_query_embeds.append(query_embeds)

    dataset_query_embeds = torch.cat(dataset_query_embeds)
    return dataset_query_embeds


# not used in QA
@torch.no_grad()
def prepare_local_cases(model, tokenizer, train_dataset, key_memory: List[torch.tensor],
                        data_to_retrieve_list: List[List[dict]], args,
                        max_num_to_ban=48, query_batch_size=512, fp16_query=False):
    model.eval()
    dataset_query_embeds = build_dataset_query(model, tokenizer, train_dataset, args)
    local_size = args.local_size
    retrieve_topk = local_size + 1  # retrieve retrieve_topk-cases from each key_memory, +1 is used to exclude itself.
    if args.filter_type != "not_filter":
        retrieve_topk = local_size + max_num_to_ban  # 48 > the max num to ban-to-retrieve
    for example in train_dataset.data:  # clear previous local_cases and their scores
        example["local_cases"] = []
        example["local_cases_scores"] = []
    for cur_key_memory, cur_data_to_retrieve in zip(key_memory, data_to_retrieve_list):
        cur_cuda_key_memory = cur_key_memory.cuda()
        for start_idx in tqdm(range(0, len(dataset_query_embeds), query_batch_size),
                              total=len(dataset_query_embeds) // query_batch_size, desc="retrieve local cases"):
            cur_cuda_query_embeds = dataset_query_embeds[start_idx: start_idx + query_batch_size].cuda()
            cur_batch_data = train_dataset.data[start_idx: start_idx + query_batch_size]
            if fp16_query:
                scores = torch.mm(cur_cuda_query_embeds, cur_cuda_key_memory.t())
            else:  # not use fp16_query to avoid overflow
                scores = torch.mm(cur_cuda_query_embeds.float(), cur_cuda_key_memory.t().float())
            topk = scores.topk(min(len(cur_key_memory), retrieve_topk), dim=1)
            top_indices = topk.indices.tolist()
            top_scores = topk.values.tolist()
            for cur_example, cur_indices, cur_scores in zip(cur_batch_data, top_indices, top_scores):
                cur_local_cases = []
                cur_local_cases_scores = []
                for case_idx, case_score in zip(cur_indices, cur_scores):
                    cur_case = cur_data_to_retrieve[case_idx]
                    cur_case_id = cur_case["id"]

                    if cur_case_id == cur_example["id"]:
                        continue  # exclude itself
                    if "ban_to_retrieve_list" in cur_example:
                        if cur_case_id in cur_example["ban_to_retrieve_list"]:
                            continue  # the retrieved case is baned for cur_example

                    cur_local_cases.append(cur_case_id)
                    cur_local_cases_scores.append(case_score)

                cur_local_cases = cur_local_cases[:local_size]
                assert len(cur_local_cases) == local_size
                cur_example["local_cases"] += cur_local_cases
                cur_example["local_cases_scores"] = cur_local_cases_scores

        del cur_cuda_key_memory
        torch.cuda.empty_cache()

    for example in train_dataset.data:  # rank to select local-cases
        sorted_cases_with_scores = sorted(zip(example["local_cases"], example["local_cases_scores"]),
                                          key=lambda x: x[1], reverse=True)[:local_size]
        example["local_cases"] = [sc[0] for sc in sorted_cases_with_scores]
        example["local_cases_scores"] = [sc[1] for sc in sorted_cases_with_scores]


# not used in QA
@torch.no_grad()
def update_batch_retrieve_from_local(model, batch, args):
    query_input_ids = batch["input_ids"]
    query_input_attention_mask = batch["attention_mask"]
    embed_dict = model.CAT_embed_q(
        input_ids=query_input_ids,
        attention_mask=query_input_attention_mask,
        compute_key=True, compute_value=False
    )
    use_normed_query = True if args.key_matching_weight > 0.0 else False
    query_embeds = embed_dict["normed_key_embeds"] if use_normed_query else embed_dict["key_embeds"]
    query_embeds = reduce_query_or_key_embeds(query_embeds, args.key_reduce_method)
    query_embeds = query_embeds
    cur_bs = query_input_ids.shape[0]

    squeezed_local_cases_input_ids = batch.pop("squeezed_local_cases_input_ids")
    squeezed_local_cases_attention_mask = batch.pop("squeezed_local_cases_attention_mask")
    squeezed_local_cases_target_as_input_ids = batch.pop("squeezed_local_cases_target_as_input_ids")
    squeezed_local_cases_target_as_input_attention_mask = batch.pop(
        "squeezed_local_cases_target_as_input_attention_mask")
    embed_dict = model.wrapped_embed_kv(
        separate_task=args.separate_task, compute_key=True, compute_value=False,
        key_input_ids=squeezed_local_cases_input_ids, key_attention_mask=squeezed_local_cases_attention_mask,
    )
    squeezed_local_cases_key = embed_dict["normed_key_embeds"] if use_normed_query else embed_dict["key_embeds"]
    squeezed_local_cases_key = reduce_query_or_key_embeds(squeezed_local_cases_key, args.key_reduce_method)
    local_cases_key = squeezed_local_cases_key.view(cur_bs, args.local_size, -1)
    scores = torch.bmm(query_embeds.unsqueeze(dim=1), local_cases_key.transpose(2, 1)).squeeze(dim=1)
    scores_topk = scores.topk(args.num_values, dim=1)
    retrieved_indices = scores_topk.indices
    gathered_indices = retrieved_indices.unsqueeze(dim=-1)

    local_cases_input_ids = squeezed_local_cases_input_ids.view(cur_bs, args.local_size, -1)
    local_cases_attention_mask = squeezed_local_cases_attention_mask.view(cur_bs, args.local_size, -1)
    local_cases_target_as_input_ids = squeezed_local_cases_target_as_input_ids.view(cur_bs, args.local_size, -1)
    local_cases_target_as_input_attention_mask = squeezed_local_cases_target_as_input_attention_mask. \
        view(cur_bs, args.local_size, -1)
    key_input_ids = torch.gather(local_cases_input_ids, 1, gathered_indices.
                                 repeat(1, 1, local_cases_input_ids.shape[-1]))
    key_attention_mask = torch.gather(local_cases_attention_mask, 1, gathered_indices.
                                      repeat(1, 1, local_cases_attention_mask.shape[-1]))
    value_input_ids = torch.gather(local_cases_target_as_input_ids, 1, gathered_indices.
                                   repeat(1, 1, local_cases_target_as_input_ids.shape[-1]))
    value_attention_mask = torch.gather(local_cases_target_as_input_attention_mask, 1, gathered_indices.
                                        repeat(1, 1, local_cases_target_as_input_attention_mask.shape[-1]))
    batch.update({"group_key_input_ids": key_input_ids, "group_key_attention_mask": key_attention_mask,
                  "group_value_input_ids": value_input_ids, "group_value_attention_mask": value_attention_mask})

    if args.key_matching_weight > 0.0:
        local_cases_label_ids = batch.pop("local_cases_label_ids")
        retrieved_cases_label = torch.gather(local_cases_label_ids, 1, retrieved_indices)
        squeezed_retrieved_cases_label = retrieved_cases_label.view(-1)
        label_ids = batch["label_ids"]
        matching_mask = []
        matching_target = []
        for bid, (cur_target_label, cur_retrieved_cases_label) in enumerate(zip(label_ids, retrieved_cases_label)):
            cur_mask = torch.ones_like(squeezed_retrieved_cases_label)
            cur_mask[squeezed_retrieved_cases_label == cur_target_label] = 0
            matched_pos = (cur_retrieved_cases_label == cur_target_label).nonzero().view(-1)
            if len(matched_pos) > 0:
                cur_pos_idx = matched_pos[0] + len(cur_retrieved_cases_label) * bid
                matching_target.append(cur_pos_idx)
                cur_mask[cur_pos_idx] = 1
            else:
                matching_target.append(-100)
            matching_mask.append(cur_mask)

        batch.update({"matching_target": torch.tensor(matching_target).to(model.device),
                      "matching_mask": torch.stack(matching_mask).to(model.device)})


# not used in QA
@torch.no_grad()
def retrieve_from_key_memory(model, tokenizer, dataset, key_memory: List[torch.tensor],
                             data_to_retrieve_list: List[List[dict]], args):
    model.eval()
    dataset_query_embeds = build_dataset_query(model, tokenizer, dataset, args)
    query_batch_size = 512
    for example in dataset.data:  # clear previous local_cases and their scores
        example["retrieved_cases"] = []
        example["retrieved_cases_scores"] = []
    for cur_key_memory, cur_data_to_retrieve in zip(key_memory, data_to_retrieve_list):
        cur_cuda_key_memory = cur_key_memory.cuda()
        for start_idx in tqdm(range(0, len(dataset_query_embeds), query_batch_size),
                              total=len(dataset_query_embeds) // query_batch_size, desc="query_key_memory"):
            cur_cuda_query_embeds = dataset_query_embeds[start_idx: start_idx + query_batch_size].cuda()
            cur_batch_data = dataset.data[start_idx: start_idx + query_batch_size]
            scores = torch.mm(cur_cuda_query_embeds, cur_cuda_key_memory.t())
            topk = scores.topk(args.num_values, dim=1)
            top_indices = topk.indices.tolist()
            top_scores = topk.values.tolist()
            for cur_example, cur_indices, cur_scores in zip(cur_batch_data, top_indices, top_scores):
                cur_local_cases = [cur_data_to_retrieve[case_idx] for case_idx in cur_indices]
                cur_local_cases_scores = cur_scores
                cur_example["retrieved_cases"] += cur_local_cases
                cur_example["retrieved_cases_scores"] = cur_local_cases_scores
        del cur_cuda_key_memory
        torch.cuda.empty_cache()
    for example in dataset.data:
        sorted_cases_with_scores = sorted(zip(example["retrieved_cases"], example["retrieved_cases_scores"]),
                                          key=lambda x: x[1], reverse=True)[:args.num_values]
        retrieved_cases = [sc[0] for sc in sorted_cases_with_scores]
        for item in retrieved_cases:
            item.update(dataset.get_base_input(item))
        retrieved_key_seqs = [case["input_ids"] for case in retrieved_cases]
        retrieved_value_seqs = [case["target_as_input_ids"] for case in retrieved_cases]
        example.update({"retrieved_key_seqs": retrieved_key_seqs, "retrieved_value_seqs": retrieved_value_seqs})
        # example["retrieved_cases_scores"] = [sc[1] for sc in sorted_cases_with_scores]


# it is used in QA if not rank-exists-local-qas
@torch.no_grad()
def update_local_qas_to_retrieve(args, train_dataset, qas_to_retrieve, model, key_memory: List[torch.tensor],
                                 normed_answer_of_qas_to_ret, train_data_query_embeds=None, build_mem_batch_size=1024,
                                 query_batch_size=128, local_size=1024, pos_from_top=50, neg_from_top=200,
                                 use_retrieval_adapter=False):
    model.eval()
    assert type(key_memory) == list

    logger.info(f"Prepare local QAs for each example to retrieve.")
    all_local_qas = []
    all_local_positive = []
    all_local_negative = []
    all_ret_qas = []

    if train_data_query_embeds is None:
        if use_retrieval_adapter:
            dim = args.adapter_out_dim
        else:
            dim = model.model_dim
        train_data_query_embeds = torch.zeros((len(train_dataset.data), dim), device='cpu', dtype=torch.float16)
    if args.qa_data_name == "tq":
        build_query_batch_size = 256
    else:
        build_query_batch_size = build_mem_batch_size
    embed_query_dataloader = train_dataset.get_query_dataloader(batch_size=build_query_batch_size,
                                                                shuffle=False, num_workers=1)
    start_idx = 0
    for query_inputs in tqdm(embed_query_dataloader, total=len(embed_query_dataloader), desc="Embed queries."):
        end_idx = start_idx + len(query_inputs["query_input_ids"])
        embed_dict = model.CAT_embed_q(
            input_ids=query_inputs["query_input_ids"].to(model.device),
            attention_mask=query_inputs["query_attention_mask"].to(model.device),
            compute_key=True, compute_value=False
        )
        query_embeds = embed_dict["normed_key_embeds"]
        query_embeds = reduce_query_or_key_embeds(query_embeds, args.key_reduce_method)
        if use_retrieval_adapter:
            query_embeds = model.adapter(query_embeds)
        query_embeds = query_embeds.half().cpu()
        train_data_query_embeds[start_idx: end_idx] = query_embeds
        start_idx = end_idx
    assert start_idx == len(train_dataset.data)

    torch.cuda.empty_cache()

    key_nums = sum(len(k) for k in key_memory)
    logger.info(f"key-memory seg-num: {len(key_memory)}. all key nums: {key_nums}.")

    for start_idx in tqdm(range(0, len(train_dataset.data), query_batch_size),
                          total=len(train_dataset.data) // query_batch_size + 1):
        cur_cuda_query_embeds = train_data_query_embeds[start_idx: start_idx + query_batch_size].cuda()
        if key_nums > 10000000:
            # if scale is large: calculate topk in each chunk -> combine all-topk -> select final topk
            chunk_top_scores = []
            chunk_top_indices = []
            idx_shift = 0
            for ckm_idx, chunk_key_memory in enumerate(key_memory):
                chunk_key_memory_cuda = chunk_key_memory.cuda()
                chunk_topk = torch.mm(cur_cuda_query_embeds, chunk_key_memory_cuda.t()).topk(local_size, dim=1)
                chunk_top_scores.append(chunk_topk.values)  # chunk_topk.scores: [query_batch, local_size]
                chunk_top_indices.append(chunk_topk.indices + idx_shift)
                idx_shift += len(chunk_key_memory)
                del chunk_key_memory_cuda
                torch.cuda.empty_cache()
            chunk_top_scores = torch.cat(chunk_top_scores, dim=1)  # q_batch, local_size*seg_n
            chunk_top_indices = torch.cat(chunk_top_indices, dim=1).tolist()  # q_batch, local_size*seg_n
            topk = chunk_top_scores.topk(local_size, dim=1)  # q_batch, local_size
            top_indices_indices = topk.indices.tolist()
            top_indices = []
            for cur_indices_indices, cur_indices in zip(top_indices_indices, chunk_top_indices):
                top_indices.append([cur_indices[idx] for idx in cur_indices_indices])
        else:
            # if scale is moderate: calculate score in each chunk -> combine score -> select topk
            all_chunk_scores = []
            for chunk_key_memory in key_memory:
                chunk_key_memory_cuda = chunk_key_memory.cuda()
                chunk_scores = torch.mm(cur_cuda_query_embeds, chunk_key_memory_cuda.t())
                all_chunk_scores.append(chunk_scores)  # q_batch, chunk_size
                del chunk_key_memory_cuda
                torch.cuda.empty_cache()
            scores = torch.cat(all_chunk_scores, dim=1).cuda()  # q_batch, key_memory_size
            topk = scores.topk(local_size, dim=1)
            top_indices = topk.indices.tolist()

        batch = train_dataset.data[start_idx: start_idx + query_batch_size]
        for cur_example, cur_indices in zip(batch, top_indices):
            local_positive, local_negative = [], []
            # cur_target = [normalize_answer(ans) for ans in cur_example["answer"]]
            cur_target = [na for na in cur_example["normalized_answer"]]

            cur_ret_qas = []
            for top_idx, qa_idx in enumerate(cur_indices):
                if normed_answer_of_qas_to_ret[qa_idx] in cur_target:
                    if top_idx < pos_from_top:
                        local_positive.append(qa_idx)
                else:
                    if top_idx < neg_from_top:
                        local_negative.append(qa_idx)
                    elif len(local_negative) < args.negatives_num_each_example:  # ensure 32 local_negative
                        # if len(local_negative) < args.negatives_num_each_example:  # ensure 32 local_negative
                        local_negative.append(qa_idx)
                cur_ret_qas.append(qas_to_retrieve[qa_idx])

            all_ret_qas.append(cur_ret_qas)

            all_local_positive.append(local_positive)
            all_local_negative.append(local_negative)

        all_local_qas += top_indices
        del cur_cuda_query_embeds
        del topk

    torch.cuda.empty_cache()
    assert len(all_ret_qas) == len(train_dataset.data)
    assert len(all_local_qas) == len(train_dataset.data) == len(all_local_positive) == len(all_local_negative)
    matching_metric = eval_retriever(train_dataset.data, all_ret_qas, "1,2,3,4,5")
    for k, v in matching_metric.items():
        logging.info({f"local_qas initial {k}": v})
    for i in range(len(train_dataset.data)):
        train_dataset.data[i]["local_positive"] = all_local_positive[i]
        train_dataset.data[i]["local_negative"] = all_local_negative[i]
        train_dataset.data[i]["local_qas"] = all_local_qas[i]

    logger.info(f"Local QAs updated.")


@torch.no_grad()
def update_batch_inputs(args, batch, model, use_adapter_to_select_positive=False):
    model.eval()
    embed_dict = model.CAT_embed_q(
        input_ids=batch["query_input_ids"],
        attention_mask=batch["query_attention_mask"],
        compute_key=True, compute_value=False
    )
    query_embeds = embed_dict["normed_key_embeds"]
    query_embeds = reduce_query_or_key_embeds(query_embeds, args.key_reduce_method)
    if use_adapter_to_select_positive:
        query_embeds = model.adapter(query_embeds)
    batch_size, hidden_size = query_embeds.shape

    local_positive_inputs_keys = [k for k in batch.keys() if k.startswith("local_positive_inputs_")]
    local_positive_inputs = {k.replace("local_positive_inputs_", ""):
                                 batch.pop(k).view(batch_size * args.batch_local_positive_num, -1)
                             for k in local_positive_inputs_keys}
    embed_dict = model.wrapped_embed_kv(separate_task=args.separate_task, compute_key=True,
                                        compute_value=False, **local_positive_inputs)
    local_positive_key_embeds = embed_dict["normed_key_embeds"]
    local_positive_key_embeds = reduce_query_or_key_embeds(local_positive_key_embeds, args.key_reduce_method)
    if use_adapter_to_select_positive:
        local_positive_key_embeds = model.adapter(local_positive_key_embeds)
    scores = torch.bmm(query_embeds.unsqueeze(dim=1), local_positive_key_embeds.view(
        batch_size, args.batch_local_positive_num, hidden_size).transpose(2, 1)).squeeze(dim=1)
    scores = scores + (batch["local_positive_qas_mask"] - 1) * 1e-4
    scores = torch.softmax(scores, dim=1)

    sampled_pos_local_idx = torch.multinomial(scores, 1).squeeze(dim=-1)  # [batch_size]

    # sampled_local_idx: [batch_size]
    # local_positive_inputs[key_input_ids/key_attention_mask]: [batch_size*max_pos_num, seq_length]
    all_pos_key_input_ids = local_positive_inputs["key_input_ids"].view(batch_size, args.batch_local_positive_num, -1)
    all_pos_key_attention_mask = local_positive_inputs["key_attention_mask"].view(all_pos_key_input_ids.shape)
    positive_gather_indices = sampled_pos_local_idx.unsqueeze(dim=1). \
        repeat(1, all_pos_key_input_ids.shape[-1]).unsqueeze(dim=1)
    positive_key_input_ids = torch.gather(all_pos_key_input_ids, 1, positive_gather_indices).squeeze(dim=1)
    positive_key_attention_mask = torch.gather(all_pos_key_attention_mask, 1, positive_gather_indices).squeeze(dim=1)
    # positive_key_input_ids: [bach_size, seq_len]

    if "labels_to_select" in batch:
        labels_to_select = batch.pop("labels_to_select")  # batch, args.batch_local_positive_num, seq_len
        target_gather_indices = sampled_pos_local_idx.unsqueeze(dim=1). \
            repeat(1, labels_to_select.shape[-1]).unsqueeze(dim=1)
        labels = torch.gather(labels_to_select, 1, target_gather_indices).squeeze(dim=1)
        batch.update({"labels": labels})

    # if args.num_values > 1 or args.use_not_exactly_true:
    local_mixed_inputs_keys = [k for k in batch.keys() if k.startswith("local_mixed_inputs_")]
    local_mixed_inputs = {k.replace("local_mixed_inputs_", ""): batch.pop(k) for k in local_mixed_inputs_keys}
    mixed_qas_num = args.negatives_num_each_example
    embed_dict = model.wrapped_embed_kv(separate_task=args.separate_task, compute_key=True,
                                        compute_value=True, **{k: v.view(batch_size * mixed_qas_num, -1)
                                                               for k, v in local_mixed_inputs.items()})
    mixed_key_embeds = embed_dict["normed_key_embeds"]
    mixed_key_embeds = reduce_query_or_key_embeds(mixed_key_embeds, args.key_reduce_method)
    if use_adapter_to_select_positive:
        mixed_key_embeds = model.adapter(mixed_key_embeds)
    mixed_key_embeds = mixed_key_embeds.view(batch_size, mixed_qas_num, -1)
    scores = torch.bmm(query_embeds.unsqueeze(dim=1), mixed_key_embeds.transpose(2, 1)).squeeze(dim=1)
    # assert args.values_with_order is True
    # if w/o order, shuffle the group_value_qas_indices of each example
    group_value_qas_indices = scores.topk(args.num_values, dim=1).indices  # [batch_size, num_values]
    if not args.values_with_order:
        group_value_qas_indices = group_value_qas_indices.tolist()
        for value_qas_indices in group_value_qas_indices:
            random.shuffle(value_qas_indices)
        group_value_qas_indices = torch.tensor(group_value_qas_indices).to(scores.device)

    # assert args.num_values > 1
    # if args.num_values == 1, the input is from the sampled_pos_local_idx.

    # if args.num_values > 1:
    mixed_key_input_ids = local_mixed_inputs["key_input_ids"]  # [batch_size, mixed_qas_num, seq_len]
    mixed_key_attention_mask = local_mixed_inputs["key_attention_mask"]
    mixed_value_input_ids = local_mixed_inputs["value_input_ids"]
    mixed_value_attention_mask = local_mixed_inputs["value_attention_mask"]
    key_gather_indices = group_value_qas_indices.unsqueeze(dim=-1).repeat(1, 1, mixed_key_input_ids.shape[-1])
    group_key_input_ids = torch.gather(mixed_key_input_ids, 1, key_gather_indices)
    group_key_attention_mask = torch.gather(mixed_key_attention_mask, 1, key_gather_indices)
    value_gather_indices = group_value_qas_indices.unsqueeze(dim=-1).repeat(1, 1, mixed_value_input_ids.shape[-1])
    group_value_input_ids = torch.gather(mixed_value_input_ids, 1, value_gather_indices)
    group_value_attention_mask = torch.gather(mixed_value_attention_mask, 1, value_gather_indices)

    # matching_targets
    matching_targets = torch.arange(batch_size).to(model.device)
    matching_targets[batch.pop("local_positive_num") == 0] = -100

    batch.update({
        "positive_kv_inputs": {
            "key_input_ids": positive_key_input_ids,
            "key_attention_mask": positive_key_attention_mask
        },
        "negative_kv_inputs": {
            "key_input_ids": batch.pop("local_negative_inputs_key_input_ids")
                .view(batch_size * args.negatives_num_each_example, -1),
            "key_attention_mask": batch.pop("local_negative_inputs_key_attention_mask")
                .view(batch_size * args.negatives_num_each_example, -1),
        },
        "matching_targets": matching_targets,
        "group_value_inputs": {
            "key_input_ids": group_key_input_ids.view(batch_size * args.num_values, -1),
            "key_attention_mask": group_key_attention_mask.view(batch_size * args.num_values, -1),
            "value_input_ids": group_value_input_ids.view(batch_size * args.num_values, -1),
            "value_attention_mask": group_value_attention_mask.view(batch_size * args.num_values, -1),
        },
    })


# not really rank-local, only prepare positive/negative qas from exists local-qas
@torch.no_grad()
def rank_exist_local_qas(args, train_dataset: QADataset, qas_to_retrieve, model, normed_answer_of_qas_to_ret,
                         train_data_query_embeds=None, build_mem_batch_size=1204,
                         embed_local_qas_batch_size=6, local_size=1024, pos_from_top=50, neg_from_top=200,
                         accelerator=None):
    model.eval()
    if args.use_fp16_rank:
        half_model = copy.deepcopy(model)
        half_model.eval()
        model = half_model.half()
        embed_local_qas_batch_size = int(embed_local_qas_batch_size * 1.5)
    logger.info(f"Rank local QAs for each example to retrieve. embed_local_qas_batch_size={embed_local_qas_batch_size}")

    if train_data_query_embeds is None:
        train_data_query_embeds = torch.zeros((len(train_dataset.data), model.model_dim), device='cpu',
                                              dtype=torch.float16)

    embed_query_dataloader = train_dataset.get_query_dataloader(batch_size=build_mem_batch_size,
                                                                shuffle=False, num_workers=5)
    start_idx = 0
    for query_inputs in tqdm(embed_query_dataloader, total=len(embed_query_dataloader), desc="Embed queries."):
        end_idx = start_idx + len(query_inputs["query_input_ids"])
        embed_dict = model.CAT_embed_q(
            input_ids=query_inputs["query_input_ids"].to(model.device),
            attention_mask=query_inputs["query_attention_mask"].to(model.device),
            compute_key=True, compute_value=False
        )
        query_embeds = embed_dict["normed_key_embeds"]
        query_embeds = reduce_query_or_key_embeds(query_embeds, args.key_reduce_method)
        query_embeds = query_embeds.half().cpu()
        train_data_query_embeds[start_idx: end_idx] = query_embeds
        start_idx = end_idx
    assert start_idx == len(train_dataset.data)

    torch.cuda.empty_cache()

    embed_local_dataloader = train_dataset.get_local_qas_dataloader(batch_size=embed_local_qas_batch_size,
                                                                    shuffle=False, num_workers=10)
    if accelerator is not None:
        embed_local_dataloader = accelerator.prepare(embed_local_dataloader)

    all_ret_qas = []

    start_idx = 0
    for local_qas_batch in tqdm(embed_local_dataloader):
        query_ids = local_qas_batch.pop("query_ids")
        bs = len(query_ids)
        cur_query_embeds = train_data_query_embeds[start_idx: start_idx + bs]
        cur_batch = train_dataset.data[start_idx: start_idx + bs]
        start_idx = start_idx + bs

        embed_dict = model.wrapped_embed_kv(
            separate_task=args.separate_task, compute_key=True, compute_value=False,
            **local_qas_batch
        )
        squeezed_local_key_embeds = embed_dict["normed_key_embeds"]
        squeezed_local_key_embeds = reduce_query_or_key_embeds(squeezed_local_key_embeds, args.key_reduce_method)
        local_key_embeds = squeezed_local_key_embeds.view(bs, -1, model.model_dim).half()
        # exists local-qas should be larger than expect local-size
        assert local_size <= squeezed_local_key_embeds.shape[1]

        cur_query_embeds = cur_query_embeds.cuda()
        # [bs, 1, hidden] [bs, hidden, exists-local-qas-num] --> [bs, exists-local-qas-num]
        scores = torch.bmm(cur_query_embeds.unsqueeze(dim=1), local_key_embeds.transpose(2, 1)).squeeze(dim=1)
        top_local_indices = scores.topk(local_size, dim=1).indices

        for cur_example, cur_indices in zip(cur_batch, top_local_indices):
            local_positive, local_negative = [], []
            cur_target = [normalize_answer(ans) for ans in cur_example["answer"]]

            qas_ids_of_top_local = [cur_example["local_qas"][local_idx] for local_idx in cur_indices]
            # qas_of_top_local = [qas_to_retrieve[qid] for qid in qas_ids_of_top_local]
            all_ret_qas.append([qas_to_retrieve[qid] for qid in qas_ids_of_top_local[:50]])

            for top_idx, qa_idx in enumerate(qas_ids_of_top_local):
                if normed_answer_of_qas_to_ret[qa_idx] in cur_target:
                    if top_idx < pos_from_top:
                        local_positive.append(qa_idx)
                else:
                    if top_idx < neg_from_top:
                        local_negative.append(qa_idx)
                    elif len(local_negative) < args.negatives_num_each_example:  # ensure 32 local_negative
                        local_negative.append(qa_idx)

            cur_example["local_positive"] = local_positive
            cur_example["local_negative"] = local_negative

    assert len(all_ret_qas) == len(train_dataset.data)
    matching_metric = eval_retriever(train_dataset.data, all_ret_qas, "1,2,3,4,5,10,50")
    for k, v in matching_metric.items():
        logging.info({f"local_qas initial {k}": v})

    if args.use_fp16_rank:
        del model
    torch.cuda.empty_cache()
    logger.info(f"Local QAs ranked.")


# Dialog
@torch.no_grad()
def update_dialog_local_qas_to_retrieve(args, train_dataset: DialogDataset, qas_to_retrieve, model,
                                        key_memory: List[torch.tensor], normed_answer_of_qas_to_ret,
                                        train_data_query_embeds=None, build_mem_batch_size=1024,
                                        query_batch_size=128, local_size=1024, pos_from_top=50,
                                        neg_from_top=200):
    model.eval()
    assert type(key_memory) == list

    logger.info(f"Prepare local QAs for each example to retrieve.")
    all_local_qas = []
    all_local_positive = []
    all_local_negative = []
    all_ret_qas = []

    if train_data_query_embeds is None:
        train_data_query_embeds = torch.zeros((len(train_dataset.data), model.model_dim), device='cpu',
                                              dtype=torch.float16)
    embed_query_dataloader = train_dataset.get_query_dataloader(batch_size=query_batch_size, shuffle=False,
                                                                num_workers=1, )
    start_idx = 0
    for query_inputs in tqdm(embed_query_dataloader, total=len(embed_query_dataloader), desc="Embed queries."):
        end_idx = start_idx + len(query_inputs["query_input_ids"])
        embed_dict = model.CAT_embed_q(
            input_ids=query_inputs["query_input_ids"].to(model.device),
            attention_mask=query_inputs["query_attention_mask"].to(model.device),
            compute_key=True, compute_value=False
        )
        query_embeds = embed_dict["normed_key_embeds"]
        query_embeds = reduce_query_or_key_embeds(query_embeds, args.key_reduce_method)
        query_embeds = query_embeds.half().cpu()
        train_data_query_embeds[start_idx: end_idx] = query_embeds
        start_idx = end_idx
    assert start_idx == len(train_dataset.data)

    torch.cuda.empty_cache()

    key_nums = sum(len(k) for k in key_memory)
    logger.info(f"key-memory seg-num: {len(key_memory)}. all key nums: {key_nums}.")
    query_batch_size = 2000
    for start_idx in tqdm(range(0, len(train_dataset.data), query_batch_size),
                          total=len(train_dataset.data) // query_batch_size + 1):
        cur_cuda_query_embeds = train_data_query_embeds[start_idx: start_idx + query_batch_size].cuda()

        # calculate topk in each chunk -> combine all-topk -> select final topk
        chunk_top_scores = []
        chunk_top_indices = []
        idx_shift = 0
        for ckm_idx, chunk_key_memory in enumerate(key_memory):
            chunk_key_memory_cuda = chunk_key_memory.cuda()
            chunk_topk = torch.mm(cur_cuda_query_embeds, chunk_key_memory_cuda.t()).topk(local_size, dim=1)
            chunk_top_scores.append(chunk_topk.values)  # chunk_topk.scores: [query_batch, local_size]
            chunk_top_indices.append(chunk_topk.indices + idx_shift)
            idx_shift += len(chunk_key_memory)
            del chunk_key_memory_cuda
            torch.cuda.empty_cache()
        chunk_top_scores = torch.cat(chunk_top_scores, dim=1)  # q_batch, local_size*seg_n
        chunk_top_indices = torch.cat(chunk_top_indices, dim=1).tolist()  # q_batch, local_size*seg_n
        topk = chunk_top_scores.topk(local_size, dim=1)  # q_batch, local_size
        top_indices_indices = topk.indices.tolist()
        top_indices = []
        for cur_indices_indices, cur_indices in zip(top_indices_indices, chunk_top_indices):
            top_indices.append([cur_indices[idx] for idx in cur_indices_indices])

        batch = train_dataset.data[start_idx: start_idx + query_batch_size]
        for cur_example, cur_indices in zip(batch, top_indices):
            local_positive, local_negative = [], []

            cur_target_words = cur_example["normalized_response_remove_stop_words_list"]  # a list of words

            cur_ret_qas = []
            for top_idx, qa_idx in enumerate(cur_indices):
                if normed_answer_of_qas_to_ret[qa_idx] in cur_target_words:
                    # if normed_answer_of_qas_to_ret[qa_idx] in cur_target:  # the QA's answer overlaps with response
                    if top_idx < pos_from_top:
                        local_positive.append(qa_idx)
                else:
                    if top_idx < neg_from_top:
                        local_negative.append(qa_idx)
                    elif len(local_negative) < args.negatives_num_each_example:  # ensure 32 local_negative
                        # if len(local_negative) < args.negatives_num_each_example:  # ensure 32 local_negative
                        local_negative.append(qa_idx)
                cur_ret_qas.append(qas_to_retrieve[qa_idx])

            all_ret_qas.append(cur_ret_qas)

            all_local_positive.append(local_positive)
            all_local_negative.append(local_negative)

        all_local_qas += top_indices
        del cur_cuda_query_embeds
        del topk

    torch.cuda.empty_cache()
    assert len(all_ret_qas) == len(train_dataset.data)
    assert len(all_local_qas) == len(train_dataset.data) == len(all_local_positive) == len(all_local_negative)
    for i in range(len(train_dataset.data)):
        train_dataset.data[i]["local_positive"] = all_local_positive[i]
        train_dataset.data[i]["local_negative"] = all_local_negative[i]
        train_dataset.data[i]["local_qas"] = all_local_qas[i]

    logger.info(f"Local QAs updated.")
