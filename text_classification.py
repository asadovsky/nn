"""Text classification examples."""

from __future__ import print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import embedding_utils

np.random.seed(0)
tf.set_random_seed(0)

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
INPUTS = ['Well done!',
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
PAD_LEN = 4

HParams = namedtuple('HParams', ['use_glove', 'categorical'])

DEFAULT_HPARAMS = HParams(use_glove=False, categorical=True)


def train_model(hp):
  """Trains a model."""
  t = Tokenizer()
  t.fit_on_texts(INPUTS)
  encoded_inputs = t.texts_to_sequences(INPUTS)
  padded_inputs = pad_sequences(encoded_inputs, maxlen=PAD_LEN, padding='post')

  embedding = None
  if hp.use_glove:
    embedding = embedding_utils.make_glove_embedding(t.word_index, PAD_LEN,
                                                     False)
  else:
    embedding = embedding_utils.make_rand_embedding(t.word_index, PAD_LEN)

  num_classes = len(set(LABELS))
  labels = LABELS
  loss = 'binary_crossentropy'
  if hp.categorical:
    labels = to_categorical(LABELS, num_classes)
    loss = 'categorical_crossentropy'

  model = Sequential()
  model.add(embedding)
  model.add(Flatten())
  if hp.categorical:
    model.add(Dense(num_classes, activation='softmax'))
  else:
    model.add(Dense(1, activation='sigmoid'))

  model.compile(loss=loss, optimizer='adam', metrics=['acc'])
  print(model.summary())
  model.fit(padded_inputs, labels, epochs=50, verbose=0)
  loss, acc = model.evaluate(padded_inputs, labels, verbose=0)
  print('loss={} accuracy={}'.format(loss, acc))
