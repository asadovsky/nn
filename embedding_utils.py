"""Utilities for GloVe embeddings."""

from __future__ import print_function

import numpy as np
import tensorflow as tf


def _glove_filepath(dim, pruned=False):
  return "data/glove/glove.6B.{}d.{}txt".format(
      dim, "pruned." if pruned else "")


def read_glove(dim, pruned=False):
  """Reads GloVe word vectors from disk."""
  filepath = _glove_filepath(dim, pruned=pruned)
  print("Reading GloVe word vectors from {}".format(filepath))
  word2vec = {}
  with open(filepath) as f:
    for line in f:
      parts = line.split()
      word2vec[parts[0]] = np.asarray(parts[1:], dtype="float32")
  print("Read {} word vectors".format(len(word2vec)))
  assert dim == len(next(iter(word2vec.values())))
  return word2vec


def _write_pruned_glove(dim, word2vec):
  """Writes pruned GloVe word vectors to disk."""
  filepath = _glove_filepath(dim, pruned=True)
  print("Writing GloVe word vectors to {}".format(filepath))
  with open(filepath, "w") as f:
    for word, vec in word2vec.items():
      f.write(" ".join([word] + [str(v) for v in vec.tolist()]) + "\n")
  print("Wrote {} word vectors".format(len(word2vec)))


def prune_glove(dim, words_to_keep):
  """Prunes GloVe word vectors."""
  word2vec = read_glove(dim, pruned=False)
  pruned_word2vec = {}
  for word in words_to_keep:
    vec = word2vec.get(word)
    if vec is not None:
      pruned_word2vec[word] = vec
  _write_pruned_glove(dim, pruned_word2vec)


def make_embedding_matrix(id2word, word2vec, hp):
  """Returns an embedding matrix for the given words.

  Args:
    id2word: List of words (map of id to word) to include in the matrix.
    word2vec: Map of word to embedding.
  """
  input_dim = len(id2word)
  output_dim = len(next(iter(word2vec.values())))
  initializer = tf.initializers.get(hp.word_emb_initializer)
  res = np.zeros((input_dim, output_dim))
  for i, word in enumerate(id2word):
    vec = word2vec.get(word)
    if vec is None:
      # Generate random embeddings for all words missing from word2vec,
      # including UNK.
      res[i] = initializer([output_dim]).numpy()
    else:
      res[i] = vec
  return res
