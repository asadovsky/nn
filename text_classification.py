"""Text classification examples."""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import glove_utils

np.random.seed(0)
tf.set_random_seed(0)


# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
DOCS = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
NUM_CLASSES = len(set(LABELS))
PAD_LEN = 4


def make_rand_embedding(word2id, input_length):
  """Returns an Embedding with random word vectors."""
  input_dim = len(word2id) + 1
  output_dim = 8
  return Embedding(input_dim, output_dim, input_length=input_length)


def make_glove_embedding(word2id, input_length):
  """Returns an Embedding with GloVe word vectors."""
  word2vec = glove_utils.load_word2vec()
  input_dim = len(word2id) + 1
  output_dim = len(word2vec.itervalues().next())
  embedding_matrix = glove_utils.make_embedding_matrix(word2id, word2vec)
  return Embedding(input_dim, output_dim, weights=[embedding_matrix],
                   input_length=input_length, trainable=False)


def train_model(use_glove, is_categorical):
  """Trains a model."""
  t = Tokenizer()
  t.fit_on_texts(DOCS)
  encoded_docs = t.texts_to_sequences(DOCS)
  padded_docs = pad_sequences(encoded_docs, maxlen=PAD_LEN, padding='post')

  embedding = None
  if use_glove:
    embedding = make_glove_embedding(t.word_index, PAD_LEN)
  else:
    embedding = make_rand_embedding(t.word_index, PAD_LEN)

  labels = LABELS
  loss = 'binary_crossentropy'
  if is_categorical:
    labels = to_categorical(LABELS, NUM_CLASSES)
    loss = 'categorical_crossentropy'

  model = Sequential()
  model.add(embedding)
  model.add(Flatten())

  if is_categorical:
    model.add(Dense(NUM_CLASSES, activation='softmax'))
  else:
    model.add(Dense(1, activation='sigmoid'))

  model.compile(loss=loss, optimizer='adam', metrics=['acc'])
  print(model.summary())
  model.fit(padded_docs, labels, epochs=50, verbose=0)
  loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
  print('loss={} accuracy={}'.format(loss, accuracy * 100))
