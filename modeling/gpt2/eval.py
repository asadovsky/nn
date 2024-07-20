"""Evaluates a given model."""

import argparse
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from evaluation import hellaswag
from modeling import device_util
from modeling.gpt2.model import GPT

torch.set_float32_matmul_precision("high")


@dataclass(slots=True)
class Config:
    use_hf_model: bool = False
    ckpt: str = ""
    test_run: bool = False
    model_name: str = "gpt2"
    device: str = ""


def run(cfg: Config) -> None:
    device, device_type = device_util.get_device(cfg.device)

    if cfg.use_hf_model:
        model = cast(nn.Module, GPT2LMHeadModel.from_pretrained(cfg.model_name))
        model_logits_fn = lambda inputs: model(inputs).logits  # noqa: E731
    else:
        if cfg.ckpt:
            ckpt = torch.load(cfg.ckpt)
            model = GPT(ckpt["model_cfg"])
            model.load_state_dict(ckpt["model_sd"])
        else:
            model = GPT.from_pretrained(cfg.model_name)

        def model_logits_fn(inputs: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type, dtype=torch.bfloat16):
                return model(inputs)[0]

    model.eval()
    model.to(device)
    if device != "mps":
        model = torch.compile(model)

    num_correct, num_total = hellaswag.run(model_logits_fn, "val", device, cfg.test_run)
    print(f"{num_correct / num_total:.4f} ({num_correct}/{num_total})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=str)
    parser.add_argument("-t", "--test-run", action="store_true")
    parser.add_argument("-m", "--model-name", type=str, default="gpt2")
    parser.add_argument("-d", "--device", type=str)
    args = parser.parse_args()

    cfg = Config(
        ckpt=args.ckpt,
        test_run=args.test_run,
        model_name=args.model_name,
        device=args.device,
    )
    run(cfg)


if __name__ == "__main__":
    main()
