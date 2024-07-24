"""Evaluates a given model."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import numpy as np
import torch
import torch.nn as nn
from jax import numpy as jnp
from transformers import GPT2LMHeadModel

from evaluation import hellaswag
from modeling import device_util
from modeling.gpt2 import model_jax
from modeling.gpt2.model import GPT, GPTConfig

torch.set_float32_matmul_precision("high")


@dataclass(slots=True)
class Config:
    ckpt: str = ""
    test_run: bool = False
    model_name: str = "gpt2"
    use_hf: bool = False
    use_jax: bool = True
    device: str = ""


def get_torch_model(cfg: Config, device: str) -> Callable:
    if cfg.use_hf:
        model = cast(nn.Module, GPT2LMHeadModel.from_pretrained(cfg.model_name))
    elif cfg.ckpt:
        ckpt = torch.load(cfg.ckpt, weights_only=True)
        model = GPT(GPTConfig(**ckpt["model_cfg"]))
        model.load_state_dict(ckpt["model_sd"])
    else:
        model = GPT.from_pretrained(cfg.model_name)
    model.eval()
    model.to(device)
    if device != "mps":
        model = torch.compile(model)
    return model


def run(cfg: Config) -> None:
    assert not (cfg.use_hf and cfg.use_jax)
    device, device_type = device_util.get_device(cfg.device)

    if cfg.use_jax:
        model, params = model_jax.GPT.from_pretrained(cfg.model_name)

        def model_logits_fn(inputs: torch.Tensor) -> torch.Tensor:
            inputs_jax = jnp.array(inputs.cpu().numpy())
            outputs_jax = jax.jit(model.apply)(params, inputs_jax)
            return torch.tensor(np.asarray(outputs_jax[0])).to(device)

    else:
        model = get_torch_model(cfg, device)

        def model_logits_fn(inputs: torch.Tensor) -> torch.Tensor:
            if cfg.use_hf:
                return model(inputs).logits
            else:
                with torch.autocast(device_type, dtype=torch.bfloat16):
                    return model(inputs)[0]

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
