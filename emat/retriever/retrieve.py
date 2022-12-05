import argparse
import logging
import time
from copy import deepcopy

import faiss
import numpy as np
import torch
from transformers import T5Tokenizer

from emat.evaluation.eval_retriever import eval_retriever
from emat.retriever.utils import get_mips_function, parse_vectors_from_file, mips
from emat.t5 import T5Config, T5KeyValueEncoder
from emat.utils import load_jsonl, write_jsonl, to_fp16

logger = logging.getLogger(__name__)

CUDA = torch.cuda.is_available()


def get_output_format(qas_to_answer, qas_to_retrieve_from, top_indices, top_scores):
    results = []
    for qa_ind, qa in enumerate(qas_to_answer):
        res = []
        for score_ind, ind in enumerate(top_indices[qa_ind]):
            score = top_scores[qa_ind][score_ind]
            ret_qa = deepcopy(qas_to_retrieve_from[ind])
            ret_qa['score'] = float(score)
            res.append(ret_qa)
        results.append(res)

    return [{'input_qa': in_qa, 'retrieved_qas': ret_qas} for in_qa, ret_qas in zip(qas_to_answer, results)]


def embed_query(model, tokenizer, qas, prefix="", bsz=256, max_length=1024, cuda=CUDA, fp16=False):
    def tokenize(batch_qas):
        input_strs = [prefix + ex["question"] for ex in batch_qas]
        inputs = tokenizer(input_strs, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        return inputs

    if cuda:
        model = model.cuda()
        model = to_fp16(model) if fp16 else model

    t = time.time()

    def log_progress(j, outputs):
        t2 = time.time()
        logger.info(
            f'Embedded {j + 1} / {len(list(range(0, len(qas), bsz)))} batches in {t2 - t:0.2f} seconds '
            f'({sum([len(o) for o in outputs]) / (t2 - t): 0.4f} QAs per second)')

    outputs = []
    with torch.no_grad():
        for j, batch_start in enumerate(range(0, len(qas), bsz)):
            batch = qas[batch_start: batch_start + bsz]

            inputs = tokenize(batch)
            inputs = {k: v.cuda() for k, v in inputs.items()} if cuda else inputs

            batch_outputs = model(**inputs, compute_value=False, return_dict=True)
            outputs.append(batch_outputs.key.cpu())
            if j % 10 == 0:
                log_progress(j, outputs)

    log_progress(j, outputs)

    return torch.cat(outputs, dim=0).cpu()





def run_queries(model, tokenizer, qas_to_retrieve_from, qas_to_answer, top_k, index=None, prefix="",
                batch_size=128, max_length=1024, fp16=False, n_queries_to_parallelize=2048):
    assert index is not None

    logger.info('Embedding QAs to answer:')
    embedded_qas_to_answer = embed_query(model, tokenizer, qas_to_answer, prefix=prefix, bsz=batch_size,
                                         max_length=max_length, cuda=CUDA, fp16=fp16)
    logger.info('Running MIPS search:')
    top_indices, top_scores = mips(index, embedded_qas_to_answer, top_k,
                                   n_queries_to_parallelize=n_queries_to_parallelize)

    return get_output_format(qas_to_answer, qas_to_retrieve_from, top_indices, top_scores)


def _load_index_if_exists(faiss_index_path, precomputed_embeddings_dir, n_vectors_to_load=None, memory_friendly=False,
                          efsearch=128):
    index = None
    if faiss_index_path is not None:
        assert precomputed_embeddings_dir is None, "Do not specify both a --faiss_index_path and --precomputed_embeddings_dir"
        logger.info('Loading Faiss index:')
        index = faiss.read_index(faiss_index_path)
        if hasattr(index, 'hnsw'):
            index.hnsw.efSearch = efsearch

    elif precomputed_embeddings_dir is not None:
        logger.info('Loading vectors index from file:')
        index = parse_vectors_from_file(precomputed_embeddings_dir).float()
        assert n_vectors_to_load == index.shape[0]

    logger.info('Index loaded') if index is not None else None
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='path to HF model dir')
    parser.add_argument("--max_source_length", type=int, default=1024,
                        help="The maximum total input sequence length after tokenization.Sequences "
                             "longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--qas_to_answer', type=str, required=True, help="path to questions to answer in jsonl format")
    parser.add_argument('--qas_to_retrieve_from', type=str, required=True,
                        help="path to QA-pairs to retrieve answers from in jsonl format")
    parser.add_argument('--top_k', type=int, default=50, help="top K QA-pairs to retrieve for each input question")
    parser.add_argument('--output_file', type=str, required=True, help='Path to write jsonl results to')
    parser.add_argument('--faiss_index_path', default=None, type=str,
                        help="Path to faiss index, if retrieving from a faiss index")
    parser.add_argument('--precomputed_embeddings_dir', default=None, type=str,
                        help="path to a directory of vector embeddings if retrieving from raw embeddign vectors")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for embedding questions for querying')
    parser.add_argument('--n_queries_to_parallelize', type=int, default=256, help="query batch size")
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--memory_friendly_parsing', action='store_true',
                        help='Pass this to load files more slowly, but save memory')
    parser.add_argument('--faiss_efsearch', type=int, default=128,
                        help='EFSearch search time parameter for hnsw, higher is more accurate but slower')

    parser.add_argument("--source_prefix", type=str, default="nq question: ",
                        help="A prefix to add before every source text " "(useful for T5 models).", )
    parser.add_argument('--hits_at_k', type=str, help='comma separated list of K to eval hits@k for', default="1,10,50")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    qas_to_answer = load_jsonl(args.qas_to_answer)
    qas_to_retrieve_from = load_jsonl(args.qas_to_retrieve_from)

    index = _load_index_if_exists(
        args.faiss_index_path,
        args.precomputed_embeddings_dir,
        n_vectors_to_load=len(qas_to_retrieve_from),
        memory_friendly=args.memory_friendly_parsing,
        efsearch=args.faiss_efsearch
    )

    config = T5Config.from_pretrained(args.model_name_or_path)
    model = T5KeyValueEncoder.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    retrieved_answers = run_queries(
        model,
        tokenizer,
        qas_to_retrieve_from,
        qas_to_answer,
        args.top_k,
        index,
        args.source_prefix,
        args.batch_size,
        args.max_source_length,
        args.fp16,
        args.n_queries_to_parallelize,
    )

    logger.info(f'Writing retrieval output to {args.output_file}')
    write_jsonl(retrieved_answers, args.output_file)

    hits_at_k = sorted([int(k) for k in args.hits_at_k.split(',')])
    result = eval_retriever(qas_to_answer, retrieved_answers, hits_at_k)
    with open(args.output_file + ".result", "w") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
