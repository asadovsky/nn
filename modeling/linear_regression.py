"""Linear regression."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import jax
import keras
import numpy as np
import optax
import torch
from flax import linen as fnn
from jax import numpy as jnp
from keras.layers import Dense, Input
from keras.models import Sequential
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

keras.utils.set_random_seed(0)
torch.manual_seed(0)

N = 1000
X_NP = np.random.uniform(low=0, high=100, size=(N, 4)).astype(np.float32)
Y_NP = (X_NP @ np.arange(X_NP.shape[1])).reshape(-1, 1) + np.random.normal(
    size=(N, 1)
).astype(np.float32)


@dataclass(slots=True)
class Config:
    batch_size: int = N // 2
    learning_rate: float = 10.0
    max_steps: int = 500


def print_results(loss: float, w: np.ndarray, b: np.ndarray) -> None:
    print(f"{loss=:.6f} w={np.array_str(w.reshape(-1), precision=3)} b={b.item():.3f}")


def train_keras(cfg: Config) -> None:
    """Trains using Keras."""
    model = Sequential()
    model.add(Input(shape=(X_NP.shape[1],)))
    dense_layer = Dense(1)
    model.add(dense_layer)
    model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    )
    model.fit(
        X_NP,
        Y_NP,
        batch_size=cfg.batch_size,
        epochs=cfg.max_steps * cfg.batch_size // N,
        verbose=0,
    )
    loss = model.evaluate(X_NP, Y_NP, verbose=0)
    w, b = dense_layer.get_weights()
    print_results(loss, w, b)


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


class ModelWithLoss(nn.Module):
    def __init__(self, model: Callable, loss_fn: Callable) -> None:
        super().__init__()
        self._model = model
        self._loss_fn = loss_fn

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._model(inputs)
        loss = self._loss_fn(outputs, targets)
        return outputs, loss


def train_torch(cfg: Config) -> None:
    """Trains using PyTorch."""
    model = ModelWithLoss(nn.Linear(X_NP.shape[1], 1), nn.MSELoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    dl = RepeatingDataLoader(
        DataLoader(
            TensorDataset(torch.Tensor(X_NP), torch.Tensor(Y_NP)),
            batch_size=cfg.batch_size,
        )
    )
    for _ in range(cfg.max_steps):
        optimizer.zero_grad()
        x, y = next(dl)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, loss = model(torch.Tensor(X_NP), torch.Tensor(Y_NP))
    w, b = (x.data.numpy() for x in model.parameters())
    print_results(loss, w, b)


def train_jax(cfg: Config) -> None:
    """Trains using JAX."""
    model = fnn.Dense(1)
    params = model.init(jax.random.key(0), jnp.empty((1, X_NP.shape[1])))
    tx = optax.adam(learning_rate=cfg.learning_rate)
    opt_state = tx.init(params)
    dl = RepeatingDataLoader(
        DataLoader(
            TensorDataset(torch.Tensor(X_NP), torch.Tensor(Y_NP)),
            batch_size=cfg.batch_size,
        )
    )

    @jax.jit
    def mse(params: dict, inputs: jax.Array, targets: jax.Array) -> jax.Array:
        outputs = model.apply(params, inputs)
        return jnp.mean((outputs - targets) ** 2)

    loss_and_grad = jax.jit(jax.value_and_grad(mse))

    @jax.jit
    def train_step(
        params: dict, opt_state: optax.OptState, x: np.ndarray, y: np.ndarray
    ) -> tuple[dict, optax.OptState]:
        _, grad = loss_and_grad(params, x, y)
        updates, opt_state = tx.update(grad, opt_state)
        params = cast(dict, optax.apply_updates(params, updates))
        return params, opt_state

    for _ in range(cfg.max_steps):
        x, y = next(dl)
        params, opt_state = train_step(
            params, opt_state, x.data.numpy(), y.data.numpy()
        )
    loss, _ = loss_and_grad(params, X_NP, Y_NP)
    params = cast(dict, params)
    w, b = params["params"]["kernel"], params["params"]["bias"]
    print_results(loss, w, b)
