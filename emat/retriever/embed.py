import argparse
import logging
import os
import time

import torch
from transformers import T5Tokenizer

from emat.t5 import T5Config, T5KeyValueEncoder
from emat.utils import to_fp16, verbalise_qa as _verbalise, load_jsonl

logger = logging.getLogger(__name__)
CUDA = torch.cuda.is_available()


def embed_key(model, tokenizer, qas, prefix="", bsz=256, max_length=1024, cuda=CUDA, fp16=False,
              use_both_qa_for_key=False):
    """Compute the key/query embeddings.
    prefix: empty when encoding query, "encode: " when encoding key.
    """
    verbalise_qa = _verbalise if use_both_qa_for_key else lambda x, y: x

    def tokenize(batch_qas):
        input_strs = [prefix + verbalise_qa(ex["question"], ex["answer"][0]) for ex in batch_qas]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='path to HF model dir')
    parser.add_argument("--max_source_length", type=int, default=1024,
                        help="The maximum total input sequence length after tokenization.Sequences "
                             "longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--qas_to_embed', type=str, required=True, help='Path to questions to embed in jsonl format')
    parser.add_argument('--output_path', type=str, help='path to write vectors to')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('-v', '--verbose', action="store_true")

    parser.add_argument("--source_prefix", type=str, default="nq question: ",
                        help="A prefix to add before every source text " "(useful for T5 models).", )
    parser.add_argument("--use_both_qa_for_key", action="store_true", help="Use both Q and A for key embedding.")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.fp16 and not CUDA:
        raise Exception("Can't use --fp16 without a gpu, CUDA not found")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    qas_to_embed = load_jsonl(args.qas_to_embed)

    config = T5Config.from_pretrained(args.model_name_or_path)
    model = T5KeyValueEncoder.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    embed_mat = embed_key(model, tokenizer, qas_to_embed, prefix=args.source_prefix, bsz=args.batch_size,
                          max_length=args.max_source_length, fp16=args.fp16,
                          use_both_qa_for_key=args.use_both_qa_for_key)

    torch.save(embed_mat.half(), args.output_path)
