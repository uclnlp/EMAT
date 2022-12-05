#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse

from emat.evaluation.exact_match import metric_max_over_ground_truths, exact_match_score
from emat.utils import load_jsonl



def eval_generation_em(refs, preds):
    scores = []
    for ref, pred in zip(refs, preds):
        ref_answer = ref["answer"]
        em = metric_max_over_ground_truths(exact_match_score, pred, ref_answer)
        scores.append(em)
    avg_score = sum(scores) / len(scores)
    return avg_score

def eval_retriever(refs, preds, hits_at_k):
    if isinstance(hits_at_k, str):
        hits_at_k = sorted([int(k) for k in hits_at_k.split(',')])

    result_dict = {}
    for k in hits_at_k:
        scores = []
        dont_print = False
        for r, p in zip(refs, preds):
            if hits_at_k[-1] > len(p):  # p['retrieved_qas']
                print(f'Skipping hits@{k} eval as {k} is larger than number of retrieved results')
                dont_print = True
            ref_answers = r['answer']
            em = any([
                metric_max_over_ground_truths(exact_match_score, pred_answer['answer'][0], ref_answers)
                for pred_answer in p[:k]  # p['retrieved_qas'][:k]
            ])
            scores.append(em)

        avg_score = sum(scores) / len(scores)
        # if not dont_print:
        #     print(f'{k}: {100 * avg_score:0.1f}% \n({sum(scores)} / {len(scores)})')

        result_dict[f"hit@{k}"] = avg_score

    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str,
                        help="path to retrieval results to eval, in PAQ's retrieved results jsonl format")
    parser.add_argument('--references', type=str, help="path to gold answers, in jsonl format")
    parser.add_argument('--hits_at_k', type=str, help='comma separated list of K to eval hits@k for', default="1,10,50")
    args = parser.parse_args()

    refs = load_jsonl(args.references)
    preds = load_jsonl(args.predictions)
    assert len(refs) == len(preds), "number of references doesnt match number of predictions"

    hits_at_k = sorted([int(k) for k in args.hits_at_k.split(',')])
    eval_retriever(refs, preds, hits_at_k)
