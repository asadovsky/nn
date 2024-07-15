"""Downloads and tokenizes the FineWeb-Edu dataset.
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import multiprocessing
import os
from collections.abc import Iterable
from typing import cast

import datasets
import numpy as np
import tiktoken
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "FineWeb-Edu-10B")
TOKS_PER_SHARD = int(1e8)  # 100M
ENC = tiktoken.get_encoding("gpt2")


def tokenize(doc: dict) -> np.ndarray:
    """Tokenizes a doc to a NumPy array of uint16 tokens."""
    toks = [ENC.eot_token]  # <|endoftext|> token
    toks.extend(ENC.encode_ordinary(doc["text"]))
    toks_np = np.array(toks)
    assert (toks_np < np.iinfo(np.uint16).max).all()
    return toks_np.astype(np.uint16)


def mk_pbar(shard_idx: int) -> tqdm:
    return tqdm(total=TOKS_PER_SHARD, unit="tok", desc=f"Writing shard {shard_idx}")


def get_filename(shard_idx: int) -> str:
    split = "val" if shard_idx == 0 else "train"
    return os.path.join(OUTPUT_DIR, f"FineWeb-Edu-10B_{split}_{shard_idx:06d}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train"
    )

    num_toks = 0
    shard_idx = 0
    shard_toks_np = np.empty((TOKS_PER_SHARD,), dtype=np.uint16)
    pbar = mk_pbar(shard_idx)

    with multiprocessing.Pool(max(1, (os.cpu_count() or 1) // 2)) as pool:
        for toks in pool.imap(tokenize, cast(Iterable[dict], dataset), chunksize=16):
            if num_toks + len(toks) < TOKS_PER_SHARD:
                # Add to the current shard.
                shard_toks_np[num_toks : num_toks + len(toks)] = toks
                num_toks += len(toks)
                pbar.update(len(toks))
            else:
                # Fill the current shard and write it out.
                num_to_fill = TOKS_PER_SHARD - num_toks
                shard_toks_np[num_toks:] = toks[:num_to_fill]
                pbar.update(num_to_fill)
                pbar.close()
                np.save(get_filename(shard_idx), shard_toks_np)
                # Start a new shard.
                num_toks = len(toks) - num_to_fill
                shard_idx += 1
                shard_toks_np[:num_toks] = toks[num_to_fill:]
                pbar = mk_pbar(shard_idx)
                pbar.update(num_toks)

        # Write the last shard.
        if num_toks > 0:
            pbar.update(TOKS_PER_SHARD - num_toks)
            np.save(get_filename(shard_idx), shard_toks_np[:num_toks])


if __name__ == "__main__":
    main()
