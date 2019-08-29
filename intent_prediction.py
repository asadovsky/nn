"""ATIS intent prediction."""

from __future__ import print_function
import datetime
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import atis_yvchen
import embedding_utils
import plotting

np.random.seed(0)
tf.set_random_seed(0)

TRAIN_FILENAME = 'data/atis/atis.train.w-intent.iob'
TEST_FILENAME = 'data/atis/atis.test.w-intent.iob'
PAD_LEN = 32

HParams = namedtuple('HParams', [
    'embedding',  # glove, rand
    'arch',        # max_pool, concat
    'optimizer'  # adam
])

DEFAULT_HPARAMS = HParams(embedding='glove', arch='concat', optimizer='adam')


def _scalars_log_dir():
  return 'logs/scalars/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def get_inputs_and_labels(d):
  inputs = pad_sequences(d.word_id_lists, maxlen=PAD_LEN, padding='post',
                         value=d.word2id[atis_yvchen.PAD])
  labels = to_categorical(d.intent_ids, len(d.intent2id))
  return inputs, labels


def train_model(hparams):
  """Trains a model."""
  d_train = atis_yvchen.Dataset(TRAIN_FILENAME)
  d_test = atis_yvchen.Dataset(TEST_FILENAME, train_dataset=d_train)

  # TODO: Include GloVe embeddings for words that only occur in the test set.
  # Note, these won't be fine-tuned during training. Also, maybe generate random
  # embeddings for training set words that don't have GloVe embeddings.
  embedding = None
  if hparams.embedding == 'glove':
    embedding = embedding_utils.make_glove_embedding(d_train.word2id, PAD_LEN,
                                                     True)
  elif hparams.embedding == 'rand':
    embedding = embedding_utils.make_rand_embedding(d_train.word2id, PAD_LEN)
  else:
    assert False, hparams.embedding

  num_classes = len(d_train.intent2id)
  loss = 'categorical_crossentropy'

  model = Sequential()
  model.add(embedding)

  if hparams.arch == 'concat':
    model.add(Flatten())
  elif hparams.arch == 'max_pool':
    model.add(GlobalMaxPool1D())
  else:
    assert False, hparams.arch

  model.add(Dense(num_classes, activation='softmax'))

  optimizer = None
  if hparams.optimizer != 'adam':
    assert False, hparams.optimizer
  optimizer = hparams.optimizer

  model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
  print(model.summary())

  x_train, y_train = get_inputs_and_labels(d_train)
  x_test, y_test = get_inputs_and_labels(d_test)
  history = model.fit(x_train, y_train, epochs=50, verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=[TensorBoard(log_dir=_scalars_log_dir())])
  plotting.plot_history(history)

  loss_train, acc_train = model.evaluate(x_train, y_train, verbose=0)
  loss_test, acc_test = model.evaluate(x_test, y_test, verbose=0)
  print('train: loss={} accuracy={}'.format(loss_train, acc_train))
  print('test: loss={} accuracy={}'.format(loss_test, acc_test))
