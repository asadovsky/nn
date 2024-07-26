import unittest
from collections.abc import Callable
from contextlib import nullcontext

import jax
import numpy as np
import tiktoken
import torch
from jax import numpy as jnp

from modeling.gpt2 import model_jax, model_torch

ENC = tiktoken.get_encoding("gpt2")
TOKS = np.array(ENC.encode("the only thing we have to fear is fear itself"))
INPUTS = TOKS[:-1].reshape((1, -1))
TARGETS = TOKS[1:].reshape((1, -1))


def mk_logits_loss_fn_jax(bfloat16: bool) -> Callable:
    model, params = model_jax.GPT.from_pretrained("gpt2")
    model_apply = jax.jit(model.apply)

    def logits_loss_fn(
        inputs: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        with jax.default_matmul_precision("bfloat16") if bfloat16 else nullcontext():
            logits, loss = model_apply(params, jnp.array(inputs), jnp.array(targets))
        return np.asarray(logits), np.asarray(loss)

    return logits_loss_fn


def mk_logits_loss_fn_torch(bfloat16: bool) -> Callable:
    model = model_torch.GPT.from_pretrained("gpt2")
    model.eval()

    def logits_loss_fn(
        inputs: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        with (
            torch.no_grad(),
            torch.autocast("cpu", dtype=torch.bfloat16) if bfloat16 else nullcontext(),
        ):
            logits, loss = model(torch.tensor(inputs), torch.tensor(targets))
        return logits.to(torch.float32).numpy(), loss.to(torch.float32).numpy()

    return logits_loss_fn


class EquivalenceTest(unittest.TestCase):
    def test_float32(self) -> None:
        logits_jax, loss_jax = mk_logits_loss_fn_jax(False)(INPUTS, TARGETS)
        logits_torch, loss_torch = mk_logits_loss_fn_torch(False)(INPUTS, TARGETS)
        np.testing.assert_allclose(logits_jax, logits_torch, rtol=1e-5)
        np.testing.assert_allclose(loss_jax, loss_torch, rtol=1e-5)

    def test_bfloat16(self) -> None:
        logits_jax, loss_jax = mk_logits_loss_fn_jax(True)(INPUTS, TARGETS)
        logits_torch, loss_torch = mk_logits_loss_fn_torch(True)(INPUTS, TARGETS)
        np.testing.assert_allclose(logits_jax, logits_torch, rtol=0.025)
        np.testing.assert_allclose(loss_jax, loss_torch, rtol=0.025)


if __name__ == "__main__":
    unittest.main()
