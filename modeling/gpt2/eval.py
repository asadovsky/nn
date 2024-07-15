"""Evaluates a given model."""

import argparse
from typing import cast

import torch
from torch import nn
from transformers import GPT2LMHeadModel

from evaluation import hellaswag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="gpt2")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    model = cast(nn.Module, GPT2LMHeadModel.from_pretrained(args.model_name))
    model.to(args.device)
    if args.device != "mps":
        model = torch.compile(model)
    num_correct, num_total = hellaswag.run(
        lambda tokens: model(tokens).logits, "val", args.device
    )
    print(f"{num_correct / num_total:.4f} ({num_correct}/{num_total})")


if __name__ == "__main__":
    main()
