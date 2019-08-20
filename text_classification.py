"""Text classification examples."""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import embedding_utils

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
PAD_LEN = 4


def train_model(use_glove, is_categorical):
  """Trains a model."""
  t = Tokenizer()
  t.fit_on_texts(DOCS)
  encoded_docs = t.texts_to_sequences(DOCS)
  padded_docs = pad_sequences(encoded_docs, maxlen=PAD_LEN, padding='post')

  embedding = None
  if use_glove:
    embedding = embedding_utils.make_glove_embedding(t.word_index, PAD_LEN)
  else:
    embedding = embedding_utils.make_rand_embedding(t.word_index, PAD_LEN)

  num_classes = len(set(LABELS))
  labels = LABELS
  loss = 'binary_crossentropy'
  if is_categorical:
    labels = to_categorical(LABELS, num_classes)
    loss = 'categorical_crossentropy'

  model = Sequential()
  model.add(embedding)
  model.add(Flatten())
  if is_categorical:
    model.add(Dense(num_classes, activation='softmax'))
  else:
    model.add(Dense(1, activation='sigmoid'))

  model.compile(loss=loss, optimizer='adam', metrics=['acc'])
  print(model.summary())
  model.fit(padded_docs, labels, epochs=50, verbose=1)
  loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
  print('loss={} accuracy={}'.format(loss, accuracy * 100))
