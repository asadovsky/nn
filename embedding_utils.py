"""GloVe helpers."""

from __future__ import print_function

import numpy as np
from tensorflow.keras.layers import Embedding

_GLOVE_PATH = 'data/glove/glove.6B.100d.txt'


def _load_glove():
  """Loads GloVe word vectors."""
  word2vec = dict()
  print('Loading GloVe word vectors from {}'.format(_GLOVE_PATH))
  with open(_GLOVE_PATH) as f:
    for line in f:
      parts = line.split()
      word2vec[parts[0]] = np.asarray(parts[1:], dtype='float32')
  print('Loaded {} word vectors'.format(len(word2vec)))
  return word2vec


def _embedding_matrix(id2word, word2vec):
  """Makes an embedding matrix for the given words."""
  input_dim = len(id2word)
  output_dim = len(next(iter(word2vec.values())))
  embedding_matrix = np.zeros((input_dim, output_dim))
  for i, word in enumerate(id2word):
    vec = word2vec.get(word)
    if vec is not None:
      embedding_matrix[i] = vec
  return embedding_matrix


def rand_embedding(id2word, input_length):
  """Returns an Embedding with random word vectors."""
  input_dim = len(id2word)
  output_dim = 8
  # TODO: Set mask_zero=True.
  return Embedding(input_dim, output_dim, input_length=input_length)


def glove_embedding(id2word, input_length, trainable):
  """Returns an Embedding with GloVe word vectors."""
  word2vec = _load_glove()
  input_dim = len(id2word)
  output_dim = len(next(iter(word2vec.values())))
  embedding_matrix = _embedding_matrix(id2word, word2vec)
  # TODO: Set mask_zero=True.
  return Embedding(input_dim, output_dim, weights=[embedding_matrix],
                   input_length=input_length, trainable=trainable)
