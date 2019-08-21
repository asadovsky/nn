"""ATIS intent prediction."""

from __future__ import print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from load_atis_yvchen import Dataset
import embedding_utils

np.random.seed(0)
tf.set_random_seed(0)

PAD_LEN = 32
TRAIN_FILENAME = 'data/atis/atis.train.w-intent.iob'
TEST_FILENAME = 'data/atis/atis.test.w-intent.iob'


HParams = namedtuple('HParams', ['use_glove', 'max_pool'])


def train_model(hparams):
  """Trains a model."""
  d = Dataset(TRAIN_FILENAME)
  padded_utts = pad_sequences(d.word_id_lists, maxlen=PAD_LEN, padding='post',
                              value=d.word2id['<pad>'])

  embedding = None
  if hparams.use_glove:
    embedding = embedding_utils.make_glove_embedding(d.word2id, PAD_LEN)
  else:
    embedding = embedding_utils.make_rand_embedding(d.word2id, PAD_LEN)

  num_classes = len(d.intent2id)
  labels = to_categorical(d.intent_ids, num_classes)
  loss = 'categorical_crossentropy'

  model = Sequential()
  model.add(embedding)
  if hparams.max_pool:
    model.add(GlobalMaxPool1D())
  else:
    model.add(Flatten())
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=loss, optimizer='adam', metrics=['acc'])
  print(model.summary())
  model.fit(padded_utts, labels, epochs=50, verbose=1)
  loss, accuracy = model.evaluate(padded_utts, labels, verbose=0)
  print('loss={} accuracy={}'.format(loss, accuracy * 100))
