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
from flax.training.train_state import TrainState
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
    model.add(Dense(1))
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
    w, b = model.layers[0].get_weights()
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
    model.train()
    for _ in range(cfg.max_steps):
        optimizer.zero_grad()
        x, y = next(dl)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, loss = model(torch.Tensor(X_NP), torch.Tensor(Y_NP))
    w, b = (x.data.numpy() for x in model.parameters())
    print_results(loss, w, b)


def train_jax(cfg: Config) -> None:
    """Trains using JAX."""
    dl = RepeatingDataLoader(
        DataLoader(
            TensorDataset(torch.Tensor(X_NP), torch.Tensor(Y_NP)),
            batch_size=cfg.batch_size,
        )
    )

    def mk_train_state() -> TrainState:
        model = fnn.Dense(1)
        params = model.init(jax.random.key(0), jnp.ones((X_NP.shape[1],)))
        tx = optax.adam(learning_rate=cfg.learning_rate)
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    state = mk_train_state()

    @jax.jit
    def mse(params: dict, inputs: jax.Array, targets: jax.Array) -> jax.Array:
        outputs = state.apply_fn(params, inputs)
        return jnp.mean((outputs - targets) ** 2)

    loss_and_grad_fn = jax.jit(jax.value_and_grad(mse))

    @jax.jit
    def train_step(state: TrainState, x: np.ndarray, y: np.ndarray) -> TrainState:
        _, grad = loss_and_grad_fn(state.params, x, y)
        return state.apply_gradients(grads=grad)

    for _ in range(cfg.max_steps):
        x, y = next(dl)
        state = train_step(state, x.data.numpy(), y.data.numpy())
    loss, _ = loss_and_grad_fn(state.params, X_NP, Y_NP)
    params = cast(dict, state.params)
    w, b = params["params"]["kernel"], params["params"]["bias"]
    print_results(loss, w, b)
