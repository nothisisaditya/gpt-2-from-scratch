import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset

shard_size = 100_000_000
remote = "sample-10BT"
local = "edu_fineweb10B"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name=remote, split="train")
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']


def tokenize(doc):
    tokens = [eot]
    return np.array(tokens.extend(enc.encode_ordinary(doc["text"]))).astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == '__main__':
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0

        for tokens in pool.imap(tokenize, fineweb, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
