from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import random

from emat.evaluation.exact_match import normalize_answer
from utils.utils import process_labels
# from transformers.models.t5 import T5Tokenizer
from transformers import T5Tokenizer


def format_data(item, label2str, str2label, dataset_name, task):
    if dataset_name == "commonsense_qa":
        pass


class QADataset(Dataset):
    def __init__(
            self,
            data: List[Dict],
            tokenizer: T5Tokenizer,
            qas_to_retrieve,
            dataset_name,
            retrieve_strategy="dr",
            max_source_length=None,
            args=None,
            normed_answer_of_qas_to_ret=None,
    ):
        super(QADataset, self).__init__()
        self.data: List[Dict] = data

        for idx, i in enumerate(self.data):
            i["idx"] = idx
            i["normalized_answer"] = [normalize_answer(ans) for ans in i["answer"]]
        assert dataset_name in ["nq", "tq", "wq"]
        self.max_source_length = max_source_length if max_source_length is not None else 430
        self.dataset_name = dataset_name
        print(f"dataset-name: {dataset_name}")
        self.max_target_length = 64
        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id
        self.label_pad_idx = -100
        self.args = args
        self.add_ae_input = False
        self.qas_to_retrieve = qas_to_retrieve
        self.normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret

        self.pad_qa = {"question": "", "answer": [""]}

    def get_key_value_inputs(self, qas, only_return_key_inputs=False):
        # Used to get the input of Key-Value Encoder, qas are from PAQ-L1
        key_inputs = ["question: " + qa["question"] for qa in qas]
        key_inputs = self.tokenizer(key_inputs, max_length=self.max_source_length,
                                    padding=True, truncation=True, return_tensors="pt")
        if only_return_key_inputs:
            return {"key_input_ids": key_inputs["input_ids"],
                    "key_attention_mask": key_inputs["attention_mask"]}
        else:
            value_inputs = ["answer: " + qa["answer"][0] for qa in qas]
            value_inputs = self.tokenizer(value_inputs, max_length=self.max_source_length,
                                          padding=True, truncation=True, return_tensors="pt")
            return {"key_input_ids": key_inputs["input_ids"],
                    "key_attention_mask": key_inputs["attention_mask"],
                    "value_input_ids": value_inputs["input_ids"],
                    "value_attention_mask": value_inputs["attention_mask"]}

    def get_query_inputs(self, batch):
        query_inputs = ["question: " + qa["question"] for qa in batch]
        query_inputs = self.tokenizer(query_inputs, max_length=self.max_source_length,
                                      padding=True, truncation=True, return_tensors="pt")
        return {"query_input_ids": query_inputs["input_ids"],
                "query_attention_mask": query_inputs["attention_mask"]}

    def get_dataloader(self, batch_size, shuffle, num_workers):

        def base_collate_fn(batch):

            original_batch_size, filtered_batch_size = len(batch), len(batch)
            if not self.args.use_not_exactly_true:
                batch = [ex for ex in batch if len(ex["local_positive"]) > 0]
                filtered_batch_size = len(batch)
                while len(batch) == 0:  # avoid empty-batch
                    batch = random.sample(self.data, batch_size)
                    batch = [ex for ex in batch if len(ex["local_positive"]) > 0]
                    # do not change filtered_batch_size even change the batch again.

            model_inputs = {
                "batch_data_ids": torch.tensor([qa["idx"] for qa in batch]),
                "trainable_percentage": torch.tensor(filtered_batch_size / original_batch_size).repeat(len(batch)),
                # repeat ``len(batch)`` times to compatible in multi-GPUs.
            }
            model_inputs.update(self.get_query_inputs(batch))

            batch_local_positive_num = self.args.batch_local_positive_num
            neg_num_each_example = self.args.negatives_num_each_example
            local_positive_qas = []
            local_positive_num = []
            local_positive_qas_mask = []
            local_negative_qas = []
            local_pos_mix_neg_qas = []  # num = neg_num_each_example
            for ex in batch:
                cur_local_positive_qas_ids = [idx for idx in ex["local_positive"][:batch_local_positive_num]]
                cur_local_positive_qas = [self.qas_to_retrieve[idx] for idx in cur_local_positive_qas_ids]
                cur_pos_num = len(cur_local_positive_qas)
                local_positive_num.append(cur_pos_num)

                cur_local_negative_qas_idx = random.sample(ex["local_negative"], neg_num_each_example)
                cur_local_negative_qas = [self.qas_to_retrieve[idx] for idx in cur_local_negative_qas_idx]
                local_negative_qas.append(cur_local_negative_qas)
                cur_local_pos_mix_neg_qas = cur_local_positive_qas + \
                                            cur_local_negative_qas[:neg_num_each_example - cur_pos_num]
                local_pos_mix_neg_qas.append(cur_local_pos_mix_neg_qas)

                cur_pad_num = batch_local_positive_num - cur_pos_num
                cur_local_positive_qas_mask = [1] * cur_pos_num + [0] * cur_pad_num
                local_positive_qas_mask.append(cur_local_positive_qas_mask)
                cur_local_positive_qas.extend([self.pad_qa] * cur_pad_num)
                local_positive_qas.append(cur_local_positive_qas)

            model_inputs.update({"local_positive_qas_mask": torch.tensor(local_positive_qas_mask),
                                 "local_positive_num": torch.tensor(local_positive_num), })
            if self.dataset_name == "tq" or self.dataset_name == "wq":
                squeezed_positive_qas = list(chain(*local_positive_qas))
                squeezed_positive_target = [qa["answer"][0] for qa in squeezed_positive_qas]
                with self.tokenizer.as_target_tokenizer():
                    targets = self.tokenizer(squeezed_positive_target, max_length=self.max_target_length,
                                             padding=True, truncation=True, return_tensors="pt")
                model_inputs["labels_to_select"] = process_labels(targets, self.tokenizer). \
                    view(len(batch), batch_local_positive_num, -1)
            else:
                targets = [random.choice(qa["answer"]) for qa in batch]
                with self.tokenizer.as_target_tokenizer():
                    targets = self.tokenizer(targets, max_length=self.max_target_length,
                                             padding=True, truncation=True, return_tensors="pt")
                model_inputs["labels"] = process_labels(targets, self.tokenizer)

            assert self.args.select_positive_strategy == "softmax_sample"
            squeezed_positive_qas = list(chain(*local_positive_qas))
            local_positive_inputs = self.get_key_value_inputs(squeezed_positive_qas, only_return_key_inputs=True)
            model_inputs.update({f"local_positive_inputs_{k}": v.view(len(batch), batch_local_positive_num, -1)
                                 for k, v in local_positive_inputs.items()})

            squeezed_negative_qas = list(chain(*local_negative_qas))
            local_negative_inputs = self.get_key_value_inputs(squeezed_negative_qas, only_return_key_inputs=True)
            model_inputs.update({f"local_negative_inputs_{k}": v.view(len(batch), neg_num_each_example, -1)
                                 for k, v in local_negative_inputs.items()})

            squeezed_mixed_qas = list(chain(*local_pos_mix_neg_qas))
            local_mixed_inputs = self.get_key_value_inputs(squeezed_mixed_qas)
            model_inputs.update({f"local_mixed_inputs_{k}": v.view(len(batch), neg_num_each_example, -1)
                                 for k, v in local_mixed_inputs.items()})
            if self.dataset_name == "tq":
                all_targets = [[normalize_answer(an) for an in qa["answer"]] for qa in batch]
                negative_qas_answer = [normalize_answer(nqa["answer"][0]) for nqa in squeezed_negative_qas]
                negative_mask = [[1 if neg_ans not in cur_all_target else 0 for neg_ans in negative_qas_answer]
                                 for cur_all_target in all_targets]
                model_inputs.update({"negative_mask": torch.tensor(negative_mask)})

            # for multi-GPUs
            assert all(model_inputs[k].shape[0] == len(batch) for k in model_inputs.keys())

            return model_inputs

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=base_collate_fn, pin_memory=True)

    def get_query_dataloader(self, batch_size, shuffle, num_workers):

        def query_collate_fn(batch):
            return self.get_query_inputs(batch)

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=query_collate_fn, pin_memory=True)

    def get_local_qas_dataloader(self, batch_size, shuffle, num_workers):

        def local_qas_collate_fn(batch):
            # model_inputs = self.get_query_inputs(batch)
            local_qas = [[self.qas_to_retrieve[qid] for qid in ex['local_qas']] for ex in batch]
            query_ids = [ex["idx"] for ex in batch]
            squeezed_local_qas = list(chain(*local_qas))
            squeezed_local_qas_inputs = self.get_key_value_inputs(squeezed_local_qas, only_return_key_inputs=True)
            # model_inputs.update(squeezed_local_qas_inputs)
            # return model_inputs
            return {**squeezed_local_qas_inputs, "query_ids": torch.tensor(query_ids)}

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=local_qas_collate_fn, pin_memory=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
