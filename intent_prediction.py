"""ATIS intent prediction."""

from __future__ import print_function
import datetime
import os

import numpy as np
from seqeval.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
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


def hparams_seq(**kwargs):
  """Returns hyperparams for sequence processing."""
  p = Params()
  p.define('id', None,
           'String identifier for this run.')
  p.define('mode', 'seq',
           'Sequence tagging or classification. Options: seq, cls.')
  p.define('pad_len', 32,
           'Maximum sequence length.')
  p.define('padding', 'post',
           'Options: pre, post.')
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
  p.define('dropout_rate', 0.2,
           'Dropout rate. If 0, we disable dropout.')
  p.define('optimizer', 'adam',
           'The optimizer to use. Options: adam.')
  p.define('epochs', 50,
           'Number of epochs to train.')
  p.set(**kwargs)
  return p


def hparams_cls(**kwargs):
  p = hparams_seq()
  p.mode = 'cls'
  p.seq_arch = 'none'
  p.cls_arch = 'flatten'
  p.set(**kwargs)
  return p


def inputs_and_labels(d, hp):
  """Returns model inputs and labels, i.e. x and y."""
  inputs = pad_sequences(d.word_id_seqs, maxlen=hp.pad_len, padding=hp.padding,
                         value=d.word2id[atis_yvchen.PAD])
  labels = None
  if hp.mode == 'seq':
    padded_tag_id_seqs = pad_sequences(
        d.tag_id_seqs, maxlen=hp.pad_len, padding=hp.padding,
        value=d.tag2id[atis_yvchen.PAD])
    # One-hot encoding.
    labels = np.array([to_categorical(tag_ids, len(d.tag2id))
                       for tag_ids in padded_tag_id_seqs])
  elif hp.mode == 'cls':
    labels = to_categorical(d.intent_ids, len(d.intent2id))
  else:
    assert False, hp.mode
  return inputs, labels


def _get_iob_seqs(tag_id_seqs, d):
  return [[d.id2tag[i] for i in seq] for seq in tag_id_seqs]


def true_iob_seqs(d, hp):
  """Returns (possibly truncated) IOB sequences for the given dataset."""
  return [seq[:hp.pad_len] for seq in _get_iob_seqs(d.tag_id_seqs, d)]


def pred_iob_seqs(y, d):
  """Returns IOB sequences for the given predictions on the given dataset."""
  # Predict the most likely tag at each sequence position.
  # TODO: Use Viterbi algorithm when decoding IOB tag predictions.
  # https://www.tensorflow.org/api_docs/python/tf/contrib/crf/crf_decode
  y = np.argmax(y, axis=2)
  # Remove padding.
  tag_id_seqs = [seq[:len(d.tag_id_seqs[i])] for i, seq in enumerate(y)]
  return _get_iob_seqs(tag_id_seqs, d)


def build_model(d, hp):
  """Builds a model."""
  model = Sequential()

  # TODO:
  # - Include GloVe embeddings for words that only occur in the test set. Note,
  #   these won't be fine-tuned during training.
  # - Maybe generate random embeddings for training set words that don't have
  #   GloVe embeddings.
  mask_zero = hp.mode == 'seq'
  embedding = None
  if hp.embedding == 'glove':
    embedding = embedding_utils.glove_embedding(
        d.id2word, mask_zero=mask_zero, input_length=hp.pad_len, trainable=True)
  elif hp.embedding == 'rand':
    embedding = embedding_utils.rand_embedding(
        d.id2word, mask_zero=mask_zero, input_length=hp.pad_len)
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


def train_model(model, d_train, d_test, hp):
  """Trains a model."""
  x_train, y_train = inputs_and_labels(d_train, hp)
  x_test, y_test = inputs_and_labels(d_test, hp)
  callbacks = [
      ModelCheckpoint(
          'checkpoints/' + hp.id + '/{epoch:03d}-{val_loss:.4f}.hdf5'),
      TensorBoard(log_dir=('logs/' + hp.id))
  ]
  history = model.fit(x_train, y_train, epochs=hp.epochs, verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks)
  return x_train, y_train, x_test, y_test, history


def evaluate_model(prefix, model, d, x, y, hp):
  """Evaluates a model."""
  loss, acc = model.evaluate(x, y, verbose=0)
  print('{} loss={} accuracy={}'.format(prefix, loss, acc))
  if hp.mode == 'seq':
    y_true, y_pred = true_iob_seqs(d, hp), pred_iob_seqs(model.predict(x), d)
    print('{} seq.accuracy={} seq.f1={}'.format(
        prefix, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)))


def train_and_evaluate_model(hp):
  """Trains and evaluates a model."""
  hp.set(id=datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  os.makedirs('checkpoints/' + hp.id, exist_ok=True)
  os.makedirs('logs/' + hp.id, exist_ok=True)
  d_train = atis_yvchen.Dataset(TRAIN_FILENAME)
  d_test = atis_yvchen.Dataset(TEST_FILENAME, train_dataset=d_train)
  model = build_model(d_train, hp)
  print(model.summary())
  x_train, y_train, x_test, y_test, history = train_model(
      model, d_train, d_test, hp)
  plotting.plot_history(history)
  evaluate_model('train', model, d_train, x_train, y_train, hp)
  evaluate_model('test', model, d_test, x_test, y_test, hp)


def grid_search():
  for dropout_rate in [0, 0.1, 0.2, 0.5]:
    print('\n' * 5)
    print('=' * 40)
    print('dropout_rate={}'.format(dropout_rate))
    print('=' * 40)
    train_and_evaluate_model(hparams_seq(dropout_rate=dropout_rate))
