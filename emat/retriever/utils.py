import glob
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


def torch_mips(index, query_batch, top_k):
    sims = torch.matmul(query_batch, index.t())
    return sims.topk(top_k)


def flat_index_mips(index, query_batch, top_k):
    return index.search(query_batch.numpy(), top_k)


def aux_dim_index_mips(index, query_batch, top_k):
    # querying faiss indexes for MIPS using a euclidean distance index, used with hnsw
    aux_dim = query_batch.new(query_batch.shape[0]).fill_(0)
    aux_query_batch = torch.cat([query_batch, aux_dim.unsqueeze(-1)], -1)
    return index.search(aux_query_batch.numpy(), top_k)


def get_mips_function(index):
    if type(index) == torch.Tensor:
        return torch_mips
    elif 'hnsw' in str(type(index)).lower():
        return aux_dim_index_mips
    else:
        return flat_index_mips


def get_vectors_file_paths_in_vector_directory(embeddings_dir):
    paths = glob.glob(os.path.abspath(embeddings_dir) + '/*')
    np = len(paths)
    template = '.'.join(paths[0].split('.')[:-1])
    return [template + f'.{j}' for j in range(np)]


def parse_vectors_from_directory_chunks(embeddings_dir, half):
    paths = get_vectors_file_paths_in_vector_directory(embeddings_dir)
    for j, p in enumerate(paths):
        logger.info(f'Loading vectors from {p} ({j + 1} / {len(paths)})')
        m = torch.load(p)
        assert int(p.split('.')[-1]) == j, (p, j)

        if half:
            m = m if m.dtype == torch.float16 else m.half()
        else:
            m = m if m.dtype == torch.float32 else m.float()
        yield m


def parse_vectors_from_directory_fast(embeddings_dir):
    ms = []
    for m in parse_vectors_from_directory_chunks(embeddings_dir):
        ms.append(m)

    out = torch.cat(ms)
    logger.info(f'loaded index of shape {out.shape}')
    return out


def parse_vectors_from_directory_memory_friendly(embeddings_dir, size=None):
    paths = get_vectors_file_paths_in_vector_directory(embeddings_dir)
    if size is None:
        size = 0
        for j, p in enumerate(paths):
            logger.info(f'Loading vectors from {p} ({j + 1} / {len(paths)}) to find total num vectors')
            m = torch.load(p)
            size += m.shape[0]

    out = None
    offset = 0
    for j, p in enumerate(paths):
        logger.info(f'Loading vectors from {p} ({j + 1} / {len(paths)})')
        m = torch.load(p)

        assert int(p.split('.')[-1]) == j, (p, j)
        if out is None:
            out = torch.zeros(size, m.shape[1])
        out[offset: offset + m.shape[0]] = m
        offset += m.shape[0]
    assert offset == size
    logger.info(f'loaded index of shape {out.shape}')

    return out


def parse_vectors_from_directory(fi, memory_friendly=False, size=None, as_chunks=False, half=False):
    assert os.path.isdir(fi), f"Vectors directory {fi} doesnt exist, or is not a directory of pytorch vectors"
    if as_chunks:
        return parse_vectors_from_directory_chunks(fi, half)

    if memory_friendly:
        out = parse_vectors_from_directory_memory_friendly(fi, size=size)
    else:
        out = parse_vectors_from_directory_fast(fi)

    if half:
        out = out if out.dtype == torch.float16 else out.half()
    else:
        out = out if out.dtype == torch.float32 else out.float()

    return out


def parse_vectors_from_file(fi, half=False):
    assert os.path.isfile(fi), f"{fi}"
    logger.info(f'Loading vectors from {fi}')
    out = torch.load(fi)
    logger.info(f'loaded vectors of shape {out.shape}')

    if half:
        out = out if out.dtype == torch.float16 else out.half()
    else:
        out = out if out.dtype == torch.float32 else out.float()

    return out


def mips(index, queries, top_k, n_queries_to_parallelize=256):
    # t = time.time()
    all_top_indices = None
    all_top_scores = None

    _mips = get_mips_function(index)

    for mb in range(0, len(queries), n_queries_to_parallelize):
        query_batch = queries[mb:mb + n_queries_to_parallelize].float()
        scores, top_indices = _mips(index, query_batch, top_k)

        all_top_indices = top_indices if all_top_indices is None else np.concatenate([all_top_indices, top_indices])
        all_top_scores = scores if all_top_scores is None else np.concatenate([all_top_scores, scores])

        # delta = time.time() - t
        # logger.info(
        #     f'{len(all_top_indices)}/ {len(queries)} queries searched in {delta:04f} '
        #     f'seconds ({len(all_top_indices) / delta} per second)')

    assert len(all_top_indices) == len(queries)

    # delta = time.time() - t
    # logger.info(f'Index searched in {delta:04f} seconds ({len(queries) / delta} per second)')
    return all_top_indices, all_top_scores
