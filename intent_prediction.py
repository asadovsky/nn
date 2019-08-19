"""ATIS intent prediction."""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from load_atis_yvchen import Dataset
import embedding_utils

np.random.seed(0)
tf.set_random_seed(0)

PAD_LEN = 32


def load_train():
  return Dataset('data/atis/atis-2.train.w-intent.iob')


def train_model(use_glove):
  """Trains a model."""
  d = load_train()
  padded_utts = pad_sequences(d.word_id_lists, maxlen=PAD_LEN, padding='post',
                              value=d.word2id['<pad>'])

  embedding = None
  if use_glove:
    embedding = embedding_utils.make_glove_embedding(d.word2id, PAD_LEN)
  else:
    embedding = embedding_utils.make_rand_embedding(d.word2id, PAD_LEN)

  num_classes = len(d.intent_ids)
  labels = to_categorical(d.intent_ids, num_classes)
  loss = 'categorical_crossentropy'

  model = Sequential()
  model.add(embedding)
  model.add(Flatten())
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=loss, optimizer='adam', metrics=['acc'])
  print(model.summary())
  model.fit(padded_utts, labels, epochs=50, verbose=0)
  loss, accuracy = model.evaluate(padded_utts, labels, verbose=0)
  print('loss={} accuracy={}'.format(loss, accuracy * 100))
