"""Evaluates a given model."""

import argparse
from typing import cast

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from evaluation import hellaswag
from modeling import device_util
from modeling.gpt2.model import GPT

USE_HF_MODEL = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=str)
    parser.add_argument("-m", "--model_name", type=str, default="gpt2")
    parser.add_argument("-d", "--device", type=str)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    device, device_type = device_util.get_device(args.device)

    if USE_HF_MODEL:
        model = cast(nn.Module, GPT2LMHeadModel.from_pretrained(args.model_name))
        model_logits_fn = lambda inputs: model(inputs).logits  # noqa: E731
    else:
        if args.ckpt:
            ckpt = torch.load(args.ckpt)
            model = GPT(ckpt["model_cfg"])
            model.load_state_dict(ckpt["model_sd"])
        else:
            model = GPT.from_pretrained(args.model_name)

        def model_logits_fn(inputs: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type, dtype=torch.bfloat16):
                return model(inputs)[0]

    model.eval()
    model.to(device)
    if device != "mps":
        model = torch.compile(model)

    num_correct, num_total = hellaswag.run(model_logits_fn, "val", device)
    print(f"{num_correct / num_total:.4f} ({num_correct}/{num_total})")


if __name__ == "__main__":
    main()
