import logging
import string
from itertools import chain
import copy
from torch.nn.utils.rnn import pad_sequence
import re
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import random
from emat.evaluation.exact_match import normalize_answer
from utils.utils import process_labels
from transformers import T5Tokenizer
from tqdm.auto import tqdm


class DialogDataset(Dataset):
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
            max_utterances=100,
            add_topic=True,
            add_persona=True,
            max_target_length=512,
    ):
        super(DialogDataset, self).__init__()

        assert dataset_name in ["wow", "wow_unseen", "wow_kilt", "eli5_kilt"]
        self.max_source_length = max_source_length if max_source_length is not None else 1024
        self.dataset_name = dataset_name
        # print(f"dataset-name: {dataset_name}")
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id
        self.label_pad_idx = -100
        self.args = args
        self.qas_to_retrieve = qas_to_retrieve
        self.normed_answer_of_qas_to_ret = normed_answer_of_qas_to_ret
        self.max_utterances = max_utterances
        self.add_topic = add_topic
        self.add_persona = add_persona

        self.pad_qa = {"question": "", "answer": [""]}
        if dataset_name == "wow_kilt":
            self.data: List[Dict] = self.process_kilt_input(data)
        elif dataset_name == "eli5_kilt":
            self.data: List[Dict] = self.process_eli5_kilt_input(data)
        else:
            self.data: List[Dict] = self.process_to_input_and_response_pairs(data)

        if "normalized_response" in self.data[0].keys():
            stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                          "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                          'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                          'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                          'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                          'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                          'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                          'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                          'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                          'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                          'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                          'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                          'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                          "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
                          "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                          'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
                          "wouldn't"]
            if "wow" in dataset_name:
                for item in self.data:
                    item["normalized_response_remove_stop_words_list"] = [
                        w for w in item["normalized_response"].split() if w not in stop_words
                    ]
            else:
                assert dataset_name == "eli5_kilt"
                for item in self.data:
                    item["normalized_response_remove_stop_words_list"] = [
                        w for w in item["normalized_response"].split() if w not in stop_words
                    ]
                    item["normalized_response_remove_stop_words_list"] = \
                        item["normalized_response_remove_stop_words_list"][:512]

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def process_kilt_input(self, dialog_data):
        processed_data = []
        query_prefix_ids = self.tokenizer("Query:", add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for dialog_idx, item in enumerate(dialog_data):

            if len(item["input"]) < 5:
                continue

            input_utterances = item["input"].split("\n")

            if len(input_utterances) > self.max_utterances:
                continue

            utterances_ids = []
            spk = "Wizard" if len(input_utterances) % 2 == 0 else "Apprentice"
            for utterance in input_utterances:
                utterance = f"{spk}: {utterance}"
                spk = "Wizard" if spk == "Apprentice" else "Apprentice"
                ids = self.tokenizer(utterance, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                utterances_ids.append(ids)

            if sum(len(u) for u in utterances_ids) > self.max_source_length:
                max_length_per_utterance = self.max_source_length // len(utterances_ids)
                utterances_ids = [u[:max_length_per_utterance] for u in utterances_ids]

            query_ids = query_prefix_ids + list(chain(*copy.deepcopy(utterances_ids[-2:]))) + [1]

            input_ids = list(chain(*utterances_ids)) + [1]

            cur_data = {
                "id": item["id"],
                "input_ids": torch.tensor(input_ids),
                "query_ids": torch.tensor(query_ids),
                "dialog_idx": torch.tensor(dialog_idx),
            }

            if "output" in item.keys():
                response = item["output"][0]["answer"]
                with self.tokenizer.as_target_tokenizer():
                    response_ids = self.tokenizer(response, max_length=self.max_target_length,
                                                  return_attention_mask=False)["input_ids"]
                cur_data.update({
                    "response_ids": torch.tensor(response_ids),
                    "normalized_response": self.normalize_answer(response)
                })

            processed_data.append(cur_data)

        # logging.info(f"process {len(dialog_data)} dialogs to {len(processed_data)} training examples.")
        return processed_data

    def process_eli5_kilt_input(self, eli5_data):
        assert self.max_target_length >= 1024

        def white_space_fix(text):
            return " ".join(text.split())

        processed_data = []
        query_prefix_ids = self.tokenizer("Query:", add_special_tokens=False, return_attention_mask=False)["input_ids"]

        for eli5_idx, item in tqdm(enumerate(eli5_data), total=len(eli5_data)):

            question = item["input"]
            question_ids = self.tokenizer(question, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            query_ids = (query_prefix_ids + copy.deepcopy(question_ids))[:255] + [1]
            question_ids = question_ids[:383] + [1]

            cur_data = {
                "id": item["id"],
                "input_ids": torch.tensor(question_ids),
                "query_ids": torch.tensor(query_ids),
                "dialog_idx": torch.tensor(eli5_idx),

            }

            if "output" in item.keys():
                answer = item["output"][0]["answer"]
                answer = white_space_fix(answer)

                with self.tokenizer.as_target_tokenizer():
                    response_ids = self.tokenizer(answer, max_length=self.max_target_length,
                                                  return_attention_mask=False)["input_ids"]
                cur_data.update({
                    "response_ids": torch.tensor(response_ids),
                    "normalized_response": self.normalize_answer(answer),
                    "candidate_responses": [ot['answer'] for ot in item["output"] if "answer" in ot]
                })

            processed_data.append(cur_data)

        logging.info(f"process {len(eli5_data)} dialogs to {len(processed_data)} training examples.")
        return processed_data

    def process_to_input_and_response_pairs(self, dialog_data):
        processed_data = []
        for dialog_idx, item in enumerate(dialog_data):
            dialog = item["dialog"][:self.max_utterances]
            inputs = "history:"
            if self.add_persona:
                inputs = f'persona: {item["persona"]} ' + inputs
            if self.add_topic:
                inputs = f'topic: {item["chosen_topic"]}. ' + inputs

            for turn_idx, turn in enumerate(dialog):
                speaker = turn["speaker"][2:]
                assert speaker in ["Wizard", "Apprentice"]
                if turn["speaker"][2:] == "Wizard":
                    if turn_idx == 0:
                        query = inputs
                    else:
                        query = f'topic: {item["chosen_topic"]}. {dialog[turn_idx - 1]["text"]}'
                    processed_data.append({
                        "inputs": inputs,
                        "response": turn["text"],
                        "query": query,
                        "normalized_response": self.normalize_answer(turn["text"]),
                        "dialog_idx": dialog_idx,
                        "turn_idx": turn_idx,
                    })
                inputs = inputs + f' {speaker}: {turn["text"]}'

        logging.info(f"process {len(dialog_data)} dialogs to {len(processed_data)} training examples.")
        return processed_data

    def get_qa_key_value_inputs(self, qas, only_return_key_inputs=False):
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
        if "kilt" in self.dataset_name:
            query_input_ids = [ex["query_ids"] for ex in batch]
            query_input_ids = pad_sequence(query_input_ids, batch_first=True, padding_value=self.pad_idx)
            query_attention_mask = (query_input_ids != self.pad_idx).long()
            return {"query_input_ids": query_input_ids,
                    "query_attention_mask": query_attention_mask}
        else:
            query_inputs = [ex["query"] for ex in batch]
            query_inputs = self.tokenizer(query_inputs, max_length=self.max_source_length,
                                          padding=True, truncation=True, return_tensors="pt")
            return {"query_input_ids": query_inputs["input_ids"],
                    "query_attention_mask": query_inputs["attention_mask"]}

    def get_history_inputs(self, batch):
        if "kilt" in self.dataset_name:
            history_input_ids = [ex["input_ids"] for ex in batch]
            history_input_ids = pad_sequence(history_input_ids, batch_first=True, padding_value=self.pad_idx)
            history_attention_mask = (history_input_ids != self.pad_idx).long()
            return {"history_input_ids": history_input_ids,
                    "history_attention_mask": history_attention_mask}
        else:
            history_inputs = [ex["inputs"] for ex in batch]
            history_inputs = self.tokenizer(history_inputs, max_length=self.max_source_length,
                                            padding=True, truncation=True, return_tensors="pt")
            return {"history_input_ids": history_inputs["input_ids"],
                    "history_attention_mask": history_inputs["attention_mask"]}

    def get_target_inputs(self, batch):
        if "kilt" in self.dataset_name:
            target_ids = [ex["response_ids"] for ex in batch]
            target_ids = pad_sequence(target_ids, batch_first=True, padding_value=self.pad_idx)
            return {"labels": process_labels(target_ids, self.tokenizer)}
        else:
            targets = [dialog["response"] for dialog in batch]
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(targets, max_length=self.max_target_length,
                                         padding=True, truncation=True, return_tensors="pt")
            return {"labels": process_labels(targets, self.tokenizer)}

    def get_dataloader(self, batch_size, shuffle, num_workers):

        def base_collate_fn(batch):
            original_batch_size, filtered_batch_size = len(batch), len(batch)
            # if not self.args.use_not_exactly_true:
            #     batch = [ex for ex in batch if len(ex["local_positive"]) > 0]
            #     filtered_batch_size = len(batch)
            #     while len(batch) == 0:  # avoid empty-batch
            #         batch = random.sample(self.data, batch_size)
            #         batch = [ex for ex in batch if len(ex["local_positive"]) > 0]
            #         # do not change filtered_batch_size even change the batch again.

            model_inputs = {
                "trainable_percentage": torch.tensor(filtered_batch_size / original_batch_size).repeat(len(batch)),
                # repeat ``len(batch)`` times to compatible in multi-GPUs.
            }
            model_inputs.update(self.get_query_inputs(batch))
            model_inputs.update(self.get_history_inputs(batch))
            model_inputs.update(self.get_target_inputs(batch))

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

            assert self.args.select_positive_strategy == "softmax_sample"
            squeezed_positive_qas = list(chain(*local_positive_qas))
            local_positive_inputs = self.get_qa_key_value_inputs(squeezed_positive_qas, only_return_key_inputs=True)
            model_inputs.update({f"local_positive_inputs_{k}": v.view(len(batch), batch_local_positive_num, -1)
                                 for k, v in local_positive_inputs.items()})

            squeezed_negative_qas = list(chain(*local_negative_qas))
            local_negative_inputs = self.get_qa_key_value_inputs(squeezed_negative_qas, only_return_key_inputs=True)
            model_inputs.update({f"local_negative_inputs_{k}": v.view(len(batch), neg_num_each_example, -1)
                                 for k, v in local_negative_inputs.items()})

            squeezed_mixed_qas = list(chain(*local_pos_mix_neg_qas))
            local_mixed_inputs = self.get_qa_key_value_inputs(squeezed_mixed_qas)
            model_inputs.update({f"local_mixed_inputs_{k}": v.view(len(batch), neg_num_each_example, -1)
                                 for k, v in local_mixed_inputs.items()})

            # all_targets = [[normalize_answer(an) for an in qa["response"]] for qa in batch]
            # negative_qas_answer = [normalize_answer(nqa["answer"][0]) for nqa in squeezed_negative_qas]
            # negative_mask = [[1 if neg_ans not in cur_all_target else 0 for neg_ans in negative_qas_answer]
            #                  for cur_all_target in all_targets]
            # model_inputs.update({"negative_mask": torch.tensor(negative_mask)})

            # for multi-GPUs
            assert all(model_inputs[k].shape[0] == len(batch) for k in model_inputs.keys())

            return model_inputs

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=base_collate_fn, pin_memory=True)

    def get_query_dataloader(self, batch_size, shuffle, num_workers, add_history=False):

        def query_collate_fn(batch):
            model_inputs = self.get_query_inputs(batch)

            if add_history:
                model_inputs.update(self.get_history_inputs(batch))

            return model_inputs

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=query_collate_fn, pin_memory=True, drop_last=False)

    def get_t5_dataloader(self, batch_size, shuffle, num_workers, is_train):

        def t5_collate_fn(batch):
            history_inputs = self.get_history_inputs(batch)
            response_inputs = self.get_target_inputs(batch)
            return {
                "input_ids": history_inputs["history_input_ids"],
                "attention_mask": history_inputs["history_attention_mask"],
                "labels": response_inputs["labels"]
            }

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=t5_collate_fn, pin_memory=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == '__main__':
    import json


    def load_json(fi):
        return json.load(open(fi, 'r'))


    def load_jsonl(fn):
        all_data = []
        with open(fn, "r") as f:
            for line in f.readlines():
                all_data.append(json.loads(line))
        return all_data


    tokenizer = T5Tokenizer.from_pretrained("./data/cbqa_data/pretrained_model/t5-base")
    # test_data = load_jsonl("wow-test_without_answers-kilt.jsonl.txt")
    # train_data = load_jsonl("wow-train-kilt.jsonl")
    dev_data = load_jsonl("./data/annotated_datasets/wizard_of_wikipedia/wow-dev-kilt.jsonl")

    exp = dev_data[0]
    print("")

    dataset = DialogDataset(dev_data, tokenizer, None, "wow_kilt",
                            max_source_length=768, max_utterances=10)
    # data: List[Dict],
    # tokenizer: T5Tokenizer,
    # qas_to_retrieve,
    # dataset_name,
    # retrieve_strategy = "dr",
    # max_source_length = None,
    # args = None,
    # normed_answer_of_qas_to_ret = None,
    # max_utterances = 100,
    # add_topic = True,
    # add_persona = True
