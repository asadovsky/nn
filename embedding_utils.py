"""GloVe helpers."""

from __future__ import print_function

import numpy as np
from tensorflow.keras.layers import Embedding


def _load_glove(output_dim):
  """Loads GloVe word vectors."""
  filepath = 'data/glove/glove.6B.{}d.txt'.format(output_dim)
  word2vec = dict()
  print('Loading GloVe word vectors from {}'.format(filepath))
  with open(filepath) as f:
    for line in f:
      parts = line.split()
      word2vec[parts[0]] = np.asarray(parts[1:], dtype='float32')
  print('Loaded {} word vectors'.format(len(word2vec)))
  assert output_dim == len(next(iter(word2vec.values())))
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


def rand_embedding(id2word, output_dim, mask_zero=False, input_length=None,
                   trainable=False):
  """Returns an Embedding with random word vectors."""
  input_dim = len(id2word)
  return Embedding(input_dim, output_dim, mask_zero=mask_zero,
                   input_length=input_length, trainable=trainable)


# TODO: Maybe generate random embeddings for training set words that don't have
# GloVe embeddings.
def glove_embedding(id2word, output_dim, mask_zero=False, input_length=None,
                    trainable=False):
  """Returns an Embedding with GloVe word vectors."""
  word2vec = _load_glove(output_dim)
  input_dim = len(id2word)
  embedding_matrix = _embedding_matrix(id2word, word2vec)
  return Embedding(input_dim, output_dim, weights=[embedding_matrix],
                   mask_zero=mask_zero, input_length=input_length,
                   trainable=trainable)
