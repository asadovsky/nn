"""GloVe helpers."""

from __future__ import print_function

import numpy as np

_GLOVE_PATH = 'data/glove/glove.6B.100d.txt'
_GLOVE_DIM = 100


def load_word2vec():
  """Loads GloVe word vectors."""
  word2vec = dict()
  with open(_GLOVE_PATH) as f:
    for line in f:
      parts = line.split()
      word2vec[parts[0]] = np.asarray(parts[1:], dtype='float32')
  print('Loaded {} word vectors'.format(len(word2vec)))
  return word2vec


def make_embedding_matrix(word2id, word2vec):
  """Makes an embedding matrix for the given words."""
  vocab_size = len(word2id) + 1
  embedding_matrix = np.zeros((vocab_size, _GLOVE_DIM))
  for word, i in word2id.items():
    vec = word2vec.get(word)
    if vec is not None:
      embedding_matrix[i] = vec
  return embedding_matrix
