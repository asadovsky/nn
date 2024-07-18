"""Linear regression."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import keras
import numpy as np
import torch
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
    """Configuration."""

    batch_size: int = N // 2
    learning_rate: float = 10.0
    max_steps: int = 500
    log_steps: int = 100


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
    w_np, b_np = dense_layer.get_weights()
    print(
        f"{loss=:.6f} "
        f"w={np.array_str(w_np.reshape(-1), precision=3)} b={b_np.item():.3f}"
    )


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


def mk_repeating_data_loader(cfg: Config) -> RepeatingDataLoader:
    return RepeatingDataLoader(
        DataLoader(
            TensorDataset(torch.Tensor(X_NP), torch.Tensor(Y_NP)),
            batch_size=cfg.batch_size,
        )
    )


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
    step = 0
    for x, y in mk_repeating_data_loader(cfg):
        if step == cfg.max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        step += 1
    with torch.no_grad():
        _, loss = model(torch.Tensor(X_NP), torch.Tensor(Y_NP))
    w_np, b_np = (x.data.numpy() for x in model.parameters())
    print(
        f"{loss=:.6f} "
        f"w={np.array_str(w_np.reshape(-1), precision=3)} b={b_np.item():.3f}"
    )
