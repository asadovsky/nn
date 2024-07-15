"""Utilities for GloVe embeddings."""

import keras
import numpy as np


def _glove_filepath(dim: int, pruned: bool = False) -> str:
    pruned_str = "pruned." if pruned else ""
    return f"resources/glove/glove.6B.{dim}d.{pruned_str}txt"


def read_glove(dim: int, pruned: bool = False) -> dict[str, np.ndarray]:
    """Reads GloVe word vectors from disk."""
    filepath = _glove_filepath(dim, pruned=pruned)
    print(f"Reading GloVe word vectors from {filepath}")
    word2vec = {}
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            word2vec[parts[0]] = np.asarray(parts[1:], dtype="float32")
    print(f"Read {len(word2vec)} word vectors")
    assert dim == len(next(iter(word2vec.values())))
    return word2vec


def _write_pruned_glove(dim: int, word2vec: dict[str, np.ndarray]) -> None:
    """Writes pruned GloVe word vectors to disk."""
    filepath = _glove_filepath(dim, pruned=True)
    print(f"Writing GloVe word vectors to {filepath}")
    with open(filepath, "w") as f:
        for word, vec in word2vec.items():
            f.write(" ".join([word] + [str(v) for v in vec.tolist()]) + "\n")
    print(f"Wrote {len(word2vec)} word vectors")


def prune_glove(dim: int, words_to_keep: list[str]) -> None:
    """Prunes GloVe word vectors."""
    word2vec = read_glove(dim, pruned=False)
    pruned_word2vec = {}
    for word in words_to_keep:
        vec = word2vec.get(word)
        if vec is not None:
            pruned_word2vec[word] = vec
    _write_pruned_glove(dim, pruned_word2vec)


def mk_embedding_matrix(
    id2word: list[str], word2vec: dict[str, np.ndarray], initializer: str
) -> np.ndarray:
    """Returns an embedding matrix for the given words.

    Args:
        id2word: List of words (map of id to word) to include in the matrix.
        word2vec: Map of word to embedding.
        initializer: Name of initializer to use, e.g. "uniform".
    """
    input_dim = len(id2word)
    output_dim = len(next(iter(word2vec.values())))
    init_fn = keras.initializers.get(initializer)
    assert isinstance(init_fn, keras.initializers.Initializer)
    res = np.zeros((input_dim, output_dim))
    for i, word in enumerate(id2word):
        vec = word2vec.get(word)
        if vec is None:
            # Generate random embeddings for all words missing from word2vec,
            # including UNK.
            res[i] = init_fn((output_dim,)).numpy()
        else:
            res[i] = vec
    return res
