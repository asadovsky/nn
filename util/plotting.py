"""Plotting libraries."""

import matplotlib.pyplot as plt
from keras.callbacks import History


def plot_history(history: History, filepath: str | None = None) -> None:
    """Plots Keras model.fit history."""
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, "b", label="Training accuracy")
    plt.plot(x, val_acc, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label="Training loss")
    plt.plot(x, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.close()
