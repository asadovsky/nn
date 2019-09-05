"""ATIS intent prediction."""

from __future__ import print_function
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Flatten, GlobalMaxPool1D, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import atis_yvchen
import embedding_utils
from params import Params
import plotting

np.random.seed(0)
tf.set_random_seed(0)

TRAIN_FILENAME = 'data/atis/atis.train.w-intent.iob'
TEST_FILENAME = 'data/atis/atis.test.w-intent.iob'


def hparams_seq():
  """Returns hyperparams for sequence processing."""
  p = Params()
  p.define('mode', 'seq',
           'Sequence tagging or classification. Options: seq, cls.')
  p.define('pad_len', 32,
           'Maximum sequence length.')
  p.define('embedding', 'glove',
           'Embedding type. Options: glove, rand.')
  p.define('seq_arch', 'bilstm',
           'Architecture for sequence processing. Used for both sequence'
           ' tagging and classification. Options: none, lstm, bilstm.')
  p.define('cls_arch', None,
           'Architecture for classification. Used only for classification.'
           ' Options: flatten, max_pool.')
  p.define('hidden_dim', 50,
           'Size of hidden sequence processing layer.')
  p.define('dropout_rate', 0,
           'Dropout rate. If 0, we disable dropout.')
  p.define('optimizer', 'adam',
           'The optimizer to use. Options: adam.')
  return p


def hparams_cls():
  p = hparams_seq()
  p.mode = 'cls'
  p.seq_arch = 'none'
  p.cls_arch = 'flatten'
  return p


def _scalars_log_dir():
  return 'logs/scalars/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def get_inputs_and_labels(d, hp):
  """Returns model inputs and labels, i.e. x and y."""
  inputs = pad_sequences(d.word_id_lists, maxlen=hp.pad_len, padding='post',
                         value=d.word2id[atis_yvchen.PAD])
  labels = None
  if hp.mode == 'seq':
    padded_tag_id_lists = pad_sequences(
        d.tag_id_lists, maxlen=hp.pad_len, padding='post',
        value=d.tag2id[atis_yvchen.PAD])
    # One-hot encoding.
    labels = np.array([to_categorical(tag_ids, len(d.tag2id))
                       for tag_ids in padded_tag_id_lists])
  elif hp.mode == 'cls':
    labels = to_categorical(d.intent_ids, len(d.intent2id))
  else:
    assert False, hp.mode
  return inputs, labels


def build_model(d, hp):
  """Builds a model."""
  model = Sequential()

  # TODO:
  # - Include GloVe embeddings for words that only occur in the test set. Note,
  #   these won't be fine-tuned during training.
  # - Maybe generate random embeddings for training set words that don't have
  #   GloVe embeddings.
  embedding = None
  if hp.embedding == 'glove':
    embedding = embedding_utils.glove_embedding(d.id2word, hp.pad_len, True)
  elif hp.embedding == 'rand':
    embedding = embedding_utils.rand_embedding(d.id2word, hp.pad_len)
  else:
    assert False, hp.embedding

  model.add(embedding)

  if hp.seq_arch == 'none':
    pass
  elif hp.seq_arch.endswith('lstm'):
    layer = LSTM(hp.hidden_dim, return_sequences=True)
    if hp.seq_arch == 'bilstm':
      layer = Bidirectional(layer)
    else:
      assert hp.seq_arch == 'lstm'
    model.add(layer)
  else:
    assert False, hp.arch

  if hp.dropout_rate > 0:
    model.add(Dropout(hp.dropout_rate))

  if hp.mode == 'seq':
    model.add(TimeDistributed(Dense(len(d.tag2id), activation='softmax')))
  elif hp.mode == 'cls':
    if hp.cls_arch == 'flatten':
      model.add(Flatten())
    elif hp.cls_arch == 'max_pool':
      model.add(GlobalMaxPool1D())
    else:
      assert False, hp.cls_arch
    model.add(Dense(len(d.intent2id), activation='softmax'))
  else:
    assert False, hp.mode

  optimizer = None
  if hp.optimizer != 'adam':
    assert False, hp.optimizer
  optimizer = hp.optimizer

  model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                metrics=['acc'])
  return model


def train_model(hp):
  """Trains a model."""
  d_train = atis_yvchen.Dataset(TRAIN_FILENAME)
  d_test = atis_yvchen.Dataset(TEST_FILENAME, train_dataset=d_train)

  model = build_model(d_train, hp)
  print(model.summary())

  x_train, y_train = get_inputs_and_labels(d_train, hp)
  x_test, y_test = get_inputs_and_labels(d_test, hp)
  history = model.fit(x_train, y_train, epochs=50, verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=[TensorBoard(log_dir=_scalars_log_dir())])
  plotting.plot_history(history)

  loss_train, acc_train = model.evaluate(x_train, y_train, verbose=0)
  loss_test, acc_test = model.evaluate(x_test, y_test, verbose=0)
  print('train: loss={} accuracy={}'.format(loss_train, acc_train))
  print('test: loss={} accuracy={}'.format(loss_test, acc_test))
