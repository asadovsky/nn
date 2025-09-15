"""Linear regression."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import optax
import torch
from flax import linen as fnn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from torch import nn as tnn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

torch.manual_seed(0)

N = 1000
X = np.random.uniform(low=0, high=100, size=(N, 4))
Y = (X @ np.arange(X.shape[1])).reshape(-1, 1) + np.random.normal(size=(N, 1))


@dataclass(slots=True)
class Config:
    batch_size: int = N // 2
    learning_rate: float = 10.0
    max_steps: int = 500


def print_results(loss: float, w: np.ndarray, b: np.ndarray) -> None:
    print(f"{loss=:.6f} w={np.array_str(w.reshape(-1), precision=3)} b={b.item():.3f}")


class RepeatingDataLoader:
    """Iterates over a DataLoader ad infinitum."""

    def __init__(self, dl: DataLoader) -> None:
        self._dl: DataLoader = dl
        self._it: Iterator = iter(dl)

    def __iter__(self) -> "RepeatingDataLoader":
        return self

    def __next__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self._dl)
            return next(self._it)


class ModelWithLoss(tnn.Module):
    def __init__(self, model: Callable, loss: Callable) -> None:
        super().__init__()
        self._model = model
        self._loss = loss

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self._model(inputs)
        loss = None
        if targets is not None:
            loss = self._loss(outputs, targets)
        return outputs, loss


def train_torch(cfg: Config) -> None:
    """Trains using PyTorch."""
    model = ModelWithLoss(tnn.Linear(X.shape[1], 1), tnn.MSELoss())
    optimizer = torch.optim.Adam(  # pyright: ignore[reportPrivateImportUsage]
        model.parameters(), lr=cfg.learning_rate
    )
    dl = RepeatingDataLoader(
        DataLoader(
            TensorDataset(torch.Tensor(X), torch.Tensor(Y)),
            batch_size=cfg.batch_size,
        )
    )
    model.train()
    for _ in range(cfg.max_steps):
        optimizer.zero_grad()
        x, y = next(dl)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, loss = model(torch.Tensor(X), torch.Tensor(Y))
    w, b = (x.data.numpy() for x in model.parameters())
    print_results(loss, w, b)


class LinearRegression(fnn.Module):
    @fnn.compact
    def __call__(
        self, inputs: jax.Array, targets: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        outputs = fnn.Dense(1)(inputs)
        loss = None
        if targets is not None:
            loss = jnp.mean((outputs - targets) ** 2)
        return outputs, loss


def train_jax(cfg: Config) -> None:
    """Trains using JAX."""
    dl = RepeatingDataLoader(
        DataLoader(
            TensorDataset(torch.Tensor(X), torch.Tensor(Y)),
            batch_size=cfg.batch_size,
        )
    )

    def mk_train_state() -> TrainState:
        model = LinearRegression()
        params = model.init(
            jax.random.key(0), jnp.ones(X.shape[1]), jnp.ones(Y.shape[1])
        )
        tx = optax.adam(learning_rate=cfg.learning_rate)
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    state = mk_train_state()
    loss_and_grad_fn = jax.jit(
        jax.value_and_grad(lambda *args, **kwargs: state.apply_fn(*args, **kwargs)[1])
    )

    @jax.jit
    def train_step(state: TrainState, x: np.ndarray, y: np.ndarray) -> TrainState:
        _, grad = loss_and_grad_fn(state.params, x, y)
        return state.apply_gradients(grads=grad)

    for _ in range(cfg.max_steps):
        x, y = next(dl)
        state = train_step(state, x.data.numpy(), y.data.numpy())
    loss, _ = loss_and_grad_fn(state.params, X, Y)
    dense_params = state.params["params"]["Dense_0"]
    w, b = dense_params["kernel"], dense_params["bias"]
    print_results(loss, w, b)  # pyright: ignore
