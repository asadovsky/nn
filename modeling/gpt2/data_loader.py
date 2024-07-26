import os

import numpy as np
import torch


class GPTDataLoader:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        ddp_rank: int,
        ddp_world_size: int,
        split: str,
    ) -> None:
        self._batch_size: int = batch_size
        self._seq_len: int = seq_len
        self._ddp_rank: int = ddp_rank
        self._ddp_world_size: int = ddp_world_size
        assert split in {"train", "val", "test"}
        data_dir = os.path.join(os.getcwd(), "resources", "FineWeb-Edu-10B")
        shards = os.listdir(data_dir)
        shards = [x for x in shards if f"_{split}_" in x]
        shards = [os.path.join(data_dir, x) for x in sorted(shards)]
        assert len(shards) > 0
        self._shards: list[str] = shards
        self._shard_idx: int = 0
        self._toks: torch.Tensor = self._load_toks()
        self._tok_idx: int = self._batch_size * self._seq_len * self._ddp_rank

    def __iter__(self) -> "GPTDataLoader":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self._batch_size, self._seq_len
        buf = self._toks[self._tok_idx : self._tok_idx + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets
        self._tok_idx += B * T * self._ddp_world_size
        # Advance to next shard if needed.
        if self._tok_idx + B * T * self._ddp_world_size >= len(self._toks):
            self._shard_idx = (self._shard_idx + 1) % len(self._shards)
            self._toks = self._load_toks()
            self._tok_idx = B * T * self._ddp_rank
        return x, y

    def _load_toks(self):
        return torch.tensor(np.load(self._shards[self._shard_idx]), dtype=torch.long)

    def state_dict(self) -> dict:
        return {"shard_idx": self._shard_idx, "tok_idx": self._tok_idx}

    def load_state_dict(self, state: dict) -> None:
        self._shard_idx = state["shard_idx"]
        self._toks = self._load_toks()
        self._tok_idx = state["tok_idx"]
