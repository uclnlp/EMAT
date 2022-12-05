import argparse
import logging
import os
import pickle
import random
import glob
import time

import faiss
import torch
from tqdm import tqdm

from emat.retriever.utils import parse_vectors_from_file

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_vector_sample(vector, sample_fraction):
    max_phi = -1
    N = 0

    phis = (vector ** 2).sum(1)
    max_phi = max(max_phi, phis.max())
    N += vector.shape[0]
    if sample_fraction == 1.0:
        vector_sample = vector
    else:
        vector_sample = vector[random.sample(range(0, len(vector)), int(len(vector) * sample_fraction))]

    return vector_sample, max_phi, N


def augment_vectors(vectors, max_phi):
    phis = (vectors ** 2).sum(1)
    aux_dim = torch.sqrt(max_phi.float() - phis.float())
    vectors = torch.cat([vectors, aux_dim.unsqueeze(-1)], -1)
    return vectors


def build_index_streaming(cached_embeddings_path,
                          output_path,
                          vector=None,
                          hnsw=False,
                          sq8_quantization=False,
                          fp16_quantization=False,
                          store_n=256,
                          ef_search=32,
                          ef_construction=80,
                          sample_fraction=0.1,
                          indexing_batch_size=5000000,
                          ):
    if vector is None:
        vector = parse_vectors_from_file(cached_embeddings_path)
    vector_size = vector.shape[1]

    if hnsw:
        if sq8_quantization:
            index = faiss.IndexHNSWSQ(vector_size + 1, faiss.ScalarQuantizer.QT_8bit, store_n)
        elif fp16_quantization:
            index = faiss.IndexHNSWSQ(vector_size + 1, faiss.ScalarQuantizer.QT_fp16, store_n)
        else:
            index = faiss.IndexHNSWFlat(vector_size + 1, store_n)

        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
    else:
        if sq8_quantization:
            index = faiss.IndexScalarQuantizer(vector_size, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        elif fp16_quantization:
            index = faiss.IndexScalarQuantizer(vector_size, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
        else:
            index = faiss.IndexIP(vector_size + 1, store_n)

    vector_sample, max_phi, N = get_vector_sample(vector, sample_fraction)
    if hnsw:
        vector_sample = augment_vectors(vector_sample, max_phi)

    if sq8_quantization or fp16_quantization:  # index requires training
        vs = vector_sample.numpy()
        logging.info(f'Training Quantizer with matrix of shape {vs.shape}')
        index.train(vs)
        del vs
    del vector_sample

    # logging.warning("tmp code")
    # import gc
    # del vector
    # gc.collect()
    # for idx in range(16):
    #     path = f"./data/embedding_and_faiss/PAQ_from_nq_ckpt/key_memory_dir/embeddings.{idx}.pt"
    #     vector = torch.load(path)
    #     if hnsw:
    #         vector_chunk = augment_vectors(vector, max_phi)

    # original code
    if hnsw:
        vector = augment_vectors(vector, max_phi)
    logging.info(f'Adding Vectors of shape {vector.shape}')
    index.add(vector.numpy())

    if output_path is not None:
        logger.info(f'Index Built, writing index to {output_path}')
        faiss.write_index(index, output_path)
        logger.info(f'Index dumped')
    else:
        logger.info("Built faiss-index.")
    return index


def parse_vectors_from_directory_chunks(embeddings_dir, half):
    assert os.path.isdir(embeddings_dir), \
        f"Vectors directory {embeddings_dir} doesnt exist, or is not a directory of pytorch vectors"
    paths = glob.glob(f"{embeddings_dir}/embeddings.*.pt")
    assert len(paths) > 0, "Files not found."
    paths_with_order = sorted([(int(os.path.basename(p).split('.')[1]), p) for p in paths], key=lambda x: x[0])
    paths = [po[1] for po in paths_with_order]
    for p in paths:
        print(p)
    for j, p in enumerate(paths):
        m = torch.load(p)
        # assert int(os.path.basename(p).split('.')[-3]) == j, (p, j)
        if half:
            m = m if m.dtype == torch.float16 else m.half()
        else:
            m = m if m.dtype == torch.float32 else m.float()
        yield m


def get_vector_sample_from_dir(cached_embeddings_path, sample_fraction, half=False):
    samples = []
    max_phi = -1
    N = 0
    vectors = parse_vectors_from_directory_chunks(cached_embeddings_path, half)
    for chunk in vectors:
        phis = (chunk ** 2).sum(1)
        max_phi = max(max_phi, phis.max())
        N += chunk.shape[0]
        if sample_fraction == 1.0:
            chunk_sample = chunk
        else:
            chunk_sample = chunk[random.sample(range(0, len(chunk)), int(len(chunk) * sample_fraction))]
        samples.append(chunk_sample)

    del vectors
    vector_sample = torch.cat(samples)
    return vector_sample, max_phi, N


def get_vector_from_key_chunks(key_chunks, half=False):
    samples = []
    max_phi = -1
    N = 0
    # vectors = parse_vectors_from_directory_chunks(key_chunks, half)
    for chunk in key_chunks:

        if half:
            chunk = chunk if chunk.dtype == torch.float16 else chunk.half()
        else:
            chunk = chunk if chunk.dtype == torch.float32 else chunk.float()

        phis = (chunk ** 2).sum(1)
        max_phi = max(max_phi, phis.max())
        N += chunk.shape[0]
        chunk_sample = chunk
        samples.append(chunk_sample)

    vector_sample = torch.cat(samples)
    return vector_sample, max_phi, N


def build_index_streaming_from_dir(cached_embeddings_path,
                                   output_path,
                                   hnsw=False,
                                   sq8_quantization=False,
                                   fp16_quantization=False,
                                   store_n=256,
                                   ef_search=32,
                                   ef_construction=80,
                                   sample_fraction=0.1,
                                   indexing_batch_size=5000000,
                                   ):
    logger.info("build index, read from directory.")
    first_chunk = torch.load(os.path.join(cached_embeddings_path, "embeddings.0.pt"))  # [batch_size, hidden_size]
    vector_size = first_chunk.shape[1]
    # load first chunk
    del first_chunk

    if not os.path.exists("./data/embedding_and_faiss/PAQ_from_nq_ckpt/trained_index.pkl"):

        if hnsw:
            if sq8_quantization:
                index = faiss.IndexHNSWSQ(vector_size + 1, faiss.ScalarQuantizer.QT_8bit, store_n)
            elif fp16_quantization:
                index = faiss.IndexHNSWSQ(vector_size + 1, faiss.ScalarQuantizer.QT_fp16, store_n)
            else:
                index = faiss.IndexHNSWFlat(vector_size + 1, store_n)

            index.hnsw.efSearch = ef_search
            index.hnsw.efConstruction = ef_construction
        else:
            if sq8_quantization:
                index = faiss.IndexScalarQuantizer(vector_size, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
            elif fp16_quantization:
                index = faiss.IndexScalarQuantizer(vector_size, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
            else:
                index = faiss.IndexIP(vector_size + 1, store_n)

        vector_sample, max_phi, N = get_vector_sample_from_dir(cached_embeddings_path, sample_fraction)

        print(max_phi, N)
        # exit()
        if hnsw:
            vector_sample = augment_vectors(vector_sample, max_phi)

        if sq8_quantization or fp16_quantization:  # index requires training
            vs = vector_sample.numpy()
            logging.info(f'Training Quantizer with matrix of shape {vs.shape}')
            index.train(vs)
            del vs
            pickle.dump({"index": index,
                         "max_phi": max_phi},
                        open("./data/embedding_and_faiss/PAQ_from_nq_ckpt/trained_index.pkl", 'wb'))
            exit()
        del vector_sample

    else:
        load_index_phi = pickle.load(open("./data/embedding_and_faiss/PAQ_from_nq_ckpt/trained_index.pkl", 'rb'))
        max_phi = load_index_phi["max_phi"]
        index = load_index_phi["index"]
        N = 64963526

    chunks_to_add = []
    added = 0
    for vector_chunk in parse_vectors_from_directory_chunks(cached_embeddings_path, half=False):
        if hnsw:
            vector_chunk = augment_vectors(vector_chunk, max_phi)

        chunks_to_add.append(vector_chunk)

        if sum(c.shape[0] for c in chunks_to_add) > indexing_batch_size:
            to_add = torch.cat(chunks_to_add)
            logging.info(f'Adding Vectors {added} -> {added + to_add.shape[0]} of {N}')
            added += to_add.shape[0]
            chunks_to_add = []
            index.add(to_add.numpy())

    if len(chunks_to_add) > 0:
        to_add = torch.cat(chunks_to_add).numpy()
        index.add(to_add)
        logging.info(f'Adding Vectors {added} -> {added + to_add.shape[0]} of {N}')

    logger.info(f'Index Built, writing index to {output_path}')
    faiss.write_index(index, output_path)
    logger.info(f'Index dumped')
    return index


def build_index_streaming_from_key_chunks(key_chunks,
                                          output_path,
                                          hnsw=False,
                                          sq8_quantization=False,
                                          fp16_quantization=False,
                                          store_n=256,
                                          ef_search=32,
                                          ef_construction=80,
                                          sample_fraction=0.1,
                                          indexing_batch_size=5000000,
                                          ):
    logger.info("build index, read from directory.")
    # first_chunk = torch.load(os.path.join(cached_embeddings_path, "0.key.pt"))  # [batch_size, hidden_size]
    vector_size = key_chunks[0].shape[1]

    if hnsw:
        if sq8_quantization:
            index = faiss.IndexHNSWSQ(vector_size + 1, faiss.ScalarQuantizer.QT_8bit, store_n)
        elif fp16_quantization:
            index = faiss.IndexHNSWSQ(vector_size + 1, faiss.ScalarQuantizer.QT_fp16, store_n)
        else:
            index = faiss.IndexHNSWFlat(vector_size + 1, store_n)

        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
    else:
        if sq8_quantization:
            index = faiss.IndexScalarQuantizer(vector_size, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        elif fp16_quantization:
            index = faiss.IndexScalarQuantizer(vector_size, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
        else:
            index = faiss.IndexFlatIP(vector_size + 1, store_n)

    vector_sample, max_phi, N = get_vector_from_key_chunks(key_chunks)
    if hnsw:
        vector_sample = augment_vectors(vector_sample, max_phi)

    if sq8_quantization or fp16_quantization:  # index requires training
        vs = vector_sample.numpy()
        logging.info(f'Training Quantizer with matrix of shape {vs.shape}')
        index.train(vs)
        del vs
    del vector_sample

    chunks_to_add = []
    added = 0
    for vector_chunk in key_chunks:
        if hnsw:
            vector_chunk = augment_vectors(vector_chunk, max_phi)

        chunks_to_add.append(vector_chunk)

        if sum(c.shape[0] for c in chunks_to_add) > indexing_batch_size:
            logging.info(f'Adding Vectors {added} -> {added + to_add.shape[0]} of {N}')
            to_add = torch.cat(chunks_to_add)
            chunks_to_add = []
            index.add(to_add)
            added += 1
            faiss.write_index(index, output_path)  # save intermediate index

    if len(chunks_to_add) > 0:
        to_add = torch.cat(chunks_to_add).float().numpy()
        index.add(to_add)
        logging.info(f'Adding Vectors {added} -> {added + to_add.shape[0]} of {N}')

    if output_path is not None:
        logger.info(f'Index Built, writing index to {output_path}')
        faiss.write_index(index, output_path)
        logger.info(f'Index dumped')
    else:
        logger.info(f'Index Built.')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Build a FAISS index from precomputed vector files from embed.py. "
                                     "Provides functionality to build either flat indexes (slow but exact)"
                                     " or HNSW indexes (much faster, but approximate). "
                                     "Optional application of 8bit or 16bit quantization is also available."
                                     " Many more indexes are possible with Faiss, consult the Faiss repository here"
                                     " if you want to build more advanced indexes.")
    parser.add_argument('--embeddings_dir', type=str, help='path to directory containing vectors to build index from')
    parser.add_argument('--output_path', type=str, help='path to write results to')
    parser.add_argument('--hnsw', action='store_true', help='Build an HNSW index rather than Flat')
    parser.add_argument('--SQ8', action='store_true', help='use SQ8 quantization on index to save memory')
    parser.add_argument('--fp16', action='store_true', help='use fp16 quantization on index to save memory')
    parser.add_argument('--store_n', type=int, default=32, help='hnsw store_n parameter')
    parser.add_argument('--ef_construction', type=int, default=128, help='hnsw ef_construction parameter')
    parser.add_argument('--ef_search', type=int, default=128, help='hnsw ef_search parameter')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='If memory is limited, specify a fraction (0.0->1.0) of the '
                             'data to sample for training the quantizer')
    parser.add_argument('--indexing_batch_size', type=int, default=None,
                        help='If memory is limited, specify the approximate number '
                             'of vectors to add to the index at once')
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()
    logging.info(f"Current process's PID: {os.getpid()}")
    set_num_threads = 10240
    faiss.omp_set_num_threads(set_num_threads)
    logging.info(f"FAISS build info -- set threads {set_num_threads}")
    logging.info(f"FAISS build info -- max threads {faiss.omp_get_max_threads()}")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    assert not (args.SQ8 and args.fp16), 'cant use both sq8 and fp16 Quantization'
    assert not os.path.exists(args.output_path), "Faiss index with name specificed in --output_path already exists"
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    args.indexing_batch_size = 10000000000000 if args.indexing_batch_size is None else args.indexing_batch_size
    assert 0 < args.sample_fraction <= 1.0

    build_start = time.perf_counter()

    if os.path.isdir(args.embeddings_dir):
        build_index_streaming_from_dir(
            args.embeddings_dir,
            args.output_path,
            args.hnsw,
            sq8_quantization=args.SQ8,
            fp16_quantization=args.fp16,
            store_n=args.store_n,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            sample_fraction=args.sample_fraction,
            indexing_batch_size=args.indexing_batch_size,
        )
    else:
        build_index_streaming(
            args.embeddings_dir,
            args.output_path,
            hnsw=args.hnsw,
            sq8_quantization=args.SQ8,
            fp16_quantization=args.fp16,
            store_n=args.store_n,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            sample_fraction=args.sample_fraction,
            indexing_batch_size=args.indexing_batch_size,
        )

    logging.info(f"building index cost {build_start - time.perf_counter():.5f} seconds")
