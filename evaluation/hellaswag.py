"""HellaSwag eval.

https://github.com/rowanz/hellaswag
"""

import json
from collections.abc import Callable, Iterator

import tiktoken
import torch
from torch.nn import functional as F

_DATA_DIR = "resources/hellaswag"
_DATA_FILENAMES = {
    "train": f"{_DATA_DIR}/hellaswag_train.jsonl",
    "val": f"{_DATA_DIR}/hellaswag_val.jsonl",
    "test": f"{_DATA_DIR}/hellaswag_test.jsonl",
}
_ENC = tiktoken.get_encoding("gpt2")


def _read_examples(split: str) -> Iterator[dict]:
    with open(_DATA_FILENAMES[split]) as f:
        for line in f:
            yield json.loads(line)


def _render_example(example: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Renders a given example.

    Args:
        example: The HellaSwag example.

    Returns:
        A tuple (toks, mask, label):
        - `toks` is a 4xN tensor of ctx and endings
        - `mask` is a 4xN tensor with value 1 for endings
        - `label` is the index of the correct ending
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_toks_len = len(_ENC.encode(ctx))
    tok_rows = []
    mask_rows = []
    for ending in endings:
        toks = _ENC.encode(ctx + " " + ending)
        tok_rows.append(_ENC.encode(ctx + " " + ending))
        mask_rows.append([0] * ctx_toks_len + [1] * (len(toks) - ctx_toks_len))

    max_row_len = max(len(row) for row in tok_rows)
    toks = torch.zeros((4, max_row_len), dtype=torch.long)
    mask = torch.zeros((4, max_row_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        toks[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return toks, mask, label


def _get_most_likely_row(
    toks: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor
) -> int:
    # Compute the autoregressive loss at all positions.
    shift_toks = (toks[..., 1:]).contiguous()
    shift_logits = (logits[..., :-1, :]).contiguous()
    flat_shift_toks = shift_toks.view(-1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_toks, reduction="none")
    shift_losses = shift_losses.view(toks.size(0), -1)
    # Get the mean loss for each ending (where mask is 1).
    shift_mask = (mask[..., 1:]).contiguous()
    mean_loss = (shift_losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
    # Pick the ending with the lowest loss.
    return int(mean_loss.argmin().item())


def run(model_fn: Callable, split: str, device: str) -> tuple[int, int]:
    num_correct, num_total = 0, 0
    for example in _read_examples(split):
        toks, mask, label = _render_example(example)
        toks = toks.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            logits = model_fn(toks)
            pred = _get_most_likely_row(toks, mask, logits)
        if pred == label:
            num_correct += 1
        num_total += 1
        # print(f"{num_correct / num_total:.4f} ({num_correct}/{num_total})")
    return num_correct, num_total
