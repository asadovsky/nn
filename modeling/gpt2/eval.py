"""Evaluates a given model."""

import argparse
from typing import cast

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from evaluation import hellaswag
from modeling import device_util
from modeling.gpt2.model import GPT


# TODO: Support trained model checkpoints.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="gpt2")
    parser.add_argument("-d", "--device", type=str)
    args = parser.parse_args()
    model_name, device = args.model_name, device_util.get_device(args.device)

    torch.set_float32_matmul_precision("high")
    if False:
        model = cast(nn.Module, GPT2LMHeadModel.from_pretrained(model_name))
        model_logits_fn = lambda inputs: model(inputs).logits  # noqa: E731
    else:
        model = GPT.from_pretrained(model_name)
        model_logits_fn = lambda inputs: model(inputs)[0]  # noqa: E731
    model.to(device)
    if device != "mps":
        model = torch.compile(model)
    num_correct, num_total = hellaswag.run(model_logits_fn, "val", device)
    print(f"{num_correct / num_total:.4f} ({num_correct}/{num_total})")


if __name__ == "__main__":
    main()
