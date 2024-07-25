import unittest
from collections.abc import Callable

import jax
import numpy as np
import tiktoken
import torch
from jax import numpy as jnp

from modeling.gpt2 import model_jax, model_torch


def mk_logits_loss_fn_jax() -> Callable:
    model, params = model_jax.GPT.from_pretrained("gpt2")
    model_apply = jax.jit(model.apply)

    def logits_loss_fn(
        inputs: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        with jax.default_matmul_precision("bfloat16"):
            logits, loss = model_apply(params, jnp.array(inputs), jnp.array(targets))
        return np.asarray(logits), np.asarray(loss)

    return logits_loss_fn


def mk_logits_loss_fn_torch() -> Callable:
    model = model_torch.GPT.from_pretrained("gpt2")
    model.eval()

    def logits_loss_fn(
        inputs: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            logits, loss = model(torch.tensor(inputs), torch.tensor(targets))
        return (logits.to(torch.float32).numpy(), loss.to(torch.float32).numpy())

    return logits_loss_fn


class EquivalenceTest(unittest.TestCase):
    def test_run(self) -> None:
        enc = tiktoken.get_encoding("gpt2")
        toks = np.array(enc.encode("the only thing we have to fear is fear itself"))
        inputs, targets = toks[:-1].reshape((1, -1)), toks[1:].reshape((1, -1))
        logits_jax, loss_jax = mk_logits_loss_fn_jax()(inputs, targets)
        logits_torch, loss_torch = mk_logits_loss_fn_torch()(inputs, targets)
        np.testing.assert_allclose(logits_jax, logits_torch, rtol=0.05)
        np.testing.assert_allclose(loss_jax, loss_torch, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
