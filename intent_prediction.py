"""ATIS intent prediction."""

from __future__ import print_function
from collections import OrderedDict
import datetime
import itertools
import os

import numpy as np
from seqeval.metrics import accuracy_score, f1_score
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GlobalAvgPool1D, GlobalMaxPool1D, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from atis_yvchen import dataset_iter, TEST_FILENAME, TRAIN_FILENAME
from dataset import Dataset
from decoding_utils import iob_transition_params
import embedding_utils
from params import Params
import plotting
from vocab import PAD, Vocab

np.random.seed(0)
tf.random.set_seed(0)


def hparams_seq(**kwargs):
  """Returns hyperparams for sequence processing."""
  p = Params()
  p.define("run_id", None,
           "String identifier for this run.")
  p.define("mode", "seq",
           "Sequence tagging or classification. Options: seq, cls.")
  p.define("pad_len", 32,
           "Maximum sequence length.")
  p.define("padding", "post",
           "Options: pre, post.")
  p.define("truncating", "post",
           "Options: pre, post.")
  p.define("emb_type", "glove",
           "Embedding type. Options: glove, rand.")
  p.define("emb_output_dim", 50,
           "Size of the embedding.")
  p.define("emb_initializer", "uniform",
           "Embedding initializer.")
  p.define("train_emb", True,
           "Whether to update embeddings during training.")
  p.define("use_viterbi_decoding", True,
           "Whether to use Viterbi (vs. independent) decoding of IOB tags.")
  p.define("seq_arch", "bilstm",
           "Architecture for sequence processing. Used for both sequence"
           " tagging and classification. Options: none, lstm, bilstm.")
  p.define("cls_arch", None,
           "Architecture for classification. Used only for classification."
           " Options: avg_pool, max_pool.")
  p.define("hidden_dim", 50,
           "Size of hidden sequence processing layer.")
  p.define("dropout_rate", 0.2,
           "Dropout rate. If 0, we disable dropout.")
  p.define("optimizer", "adam",
           "The optimizer to use. Options: adam.")
  p.define("epochs", 50,
           "Number of epochs to train.")
  p.set(**kwargs)
  return p


def hparams_cls(**kwargs):
  p = hparams_seq()
  p.mode = "cls"
  p.seq_arch = "bilstm"
  p.cls_arch = "avg_pool"
  p.set(**kwargs)
  return p


def _run_dir(run_id):
  return os.path.join("runs", run_id)


def _hyperparams_filepath(run_id):
  return os.path.join(_run_dir(run_id), "hyperparams.txt")


def _record_hyperparams(hp):
  os.makedirs(_run_dir(hp.run_id), exist_ok=True)
  with open(_hyperparams_filepath(hp.run_id), "w") as f:
    f.write(str(hp))


def _plot_history_filepath(hp):
  return os.path.join(_run_dir(hp.run_id), "history.png")


def _checkpoints_dir(run_id):
  return os.path.join(_run_dir(run_id), "checkpoints")


def _logs_dir(run_id):
  return os.path.join(_run_dir(run_id), "logs")


def _inputs_and_labels(d, hp):
  """Returns model inputs and labels, i.e. x and y."""
  inputs = pad_sequences(d.word_id_seqs, maxlen=hp.pad_len,
                         padding=hp.padding, truncating=hp.truncating,
                         value=d.word2id[PAD])
  labels = None
  if hp.mode == "seq":
    padded_tag_id_seqs = pad_sequences(
        d.tag_id_seqs, maxlen=hp.pad_len,
        padding=hp.padding, truncating=hp.truncating,
        value=d.tag2id[PAD])
    # One-hot encoding.
    labels = np.array([to_categorical(tag_ids, len(d.tag2id))
                       for tag_ids in padded_tag_id_seqs])
  elif hp.mode == "cls":
    labels = to_categorical(d.intent_ids, len(d.intent2id))
  else:
    assert False, hp.mode
  return inputs, labels


def _get_iob_seqs(tag_id_seqs, d):
  return [[d.id2tag[i] for i in seq] for seq in tag_id_seqs]


def _true_iob_seqs(d, hp):
  """Returns (possibly truncated) IOB sequences for the given dataset."""
  return [seq[:hp.pad_len] for seq in _get_iob_seqs(d.tag_id_seqs, d)]


def _pred_iob_seqs(y, d, hp):
  """Returns IOB sequences for the given predictions on the given dataset."""
  # Predict the most likely tag at each sequence position.
  if hp.use_viterbi_decoding:
    seq_lengths = [min(len(seq), hp.pad_len) for seq in d.tag_id_seqs]
    y, _ = tfa.text.crf_decode(
        tf.constant(y), iob_transition_params(d.id2tag), np.array(seq_lengths))
    y = y.numpy()
  else:
    # Independent decoding.
    y = np.argmax(y, axis=-1)
  # Remove padding.
  tag_id_seqs = [seq[:len(d.tag_id_seqs[i])] for i, seq in enumerate(y)]
  return _get_iob_seqs(tag_id_seqs, d)


def build_model(d, word2vec, hp):
  """Builds a model."""
  model = Sequential()

  # TODO: Add character BiLSTM as in https://arxiv.org/abs/1805.01052.
  # TODO: Make it so UNK appears in the training set, e.g. by replacing rare
  # words with UNK or adding some form of dropout.
  emb_matrix = embedding_utils.make_embedding_matrix(d.id2word, word2vec, hp)
  emb = Embedding(
      len(d.id2word), hp.emb_output_dim, weights=[emb_matrix],
      mask_zero=True, input_length=hp.pad_len, trainable=hp.train_emb)
  model.add(emb)

  if hp.seq_arch == "none":
    pass
  elif hp.seq_arch.endswith("lstm"):
    layer = LSTM(hp.hidden_dim, return_sequences=True)
    if hp.seq_arch == "bilstm":
      layer = Bidirectional(layer)
    else:
      assert hp.seq_arch == "lstm"
    model.add(layer)
  else:
    assert False, hp.arch

  if hp.dropout_rate > 0:
    model.add(Dropout(hp.dropout_rate))

  if hp.mode == "seq":
    # TODO: Add CRF layer.
    model.add(TimeDistributed(Dense(len(d.tag2id), activation="softmax")))
  elif hp.mode == "cls":
    if hp.cls_arch == "avg_pool":
      model.add(GlobalAvgPool1D())
    elif hp.cls_arch == "max_pool":
      model.add(GlobalMaxPool1D())
    else:
      assert False, hp.cls_arch
    # GlobalMaxPool1D does not support masking, but for some reason training
    # still succeeds.
    assert model.layers[-1].supports_masking
    model.add(Dense(len(d.intent2id), activation="softmax"))
  else:
    assert False, hp.mode

  optimizer = None
  if hp.optimizer != "adam":
    assert False, hp.optimizer
  optimizer = hp.optimizer

  model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                metrics=["acc"])
  return model


def train_model(model, d_train, d_test, hp):
  """Trains a model."""
  x_train, y_train = _inputs_and_labels(d_train, hp)
  x_test, y_test = _inputs_and_labels(d_test, hp)
  callbacks = [
      ModelCheckpoint(os.path.join(_checkpoints_dir(hp.run_id),
                                   "{epoch:03d}-{val_loss:.4f}.hdf5")),
      TensorBoard(log_dir=_logs_dir(hp.run_id))
  ]
  history = model.fit(x_train, y_train, epochs=hp.epochs, verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=callbacks)
  return x_train, y_train, x_test, y_test, history


def evaluate_model(prefix, model, d, x, y, hp):
  """Evaluates a model."""
  # TODO: Use sklearn.metrics.classification_report and
  # seqeval.metrics.classification_report.
  loss, acc = model.evaluate(x, y, verbose=0)
  print("{} loss={:.4f} accuracy={:.4f}".format(prefix, loss, acc))
  if hp.mode == "seq":
    y_true, y_pred = (_true_iob_seqs(d, hp),
                      _pred_iob_seqs(model.predict(x), d, hp))
    print("{} seq.accuracy={:.4f} seq.f1={:.4f}".format(
        prefix, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)))


def train_and_evaluate_model(hp):
  """Trains and evaluates a model."""
  hp.set(run_id=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  _record_hyperparams(hp)
  for path in [_checkpoints_dir(hp.run_id), _logs_dir(hp.run_id)]:
    os.makedirs(path, exist_ok=True)

  word2vec = {}
  if hp.emb_type == "glove":
    word2vec = embedding_utils.read_glove(hp.emb_output_dim, pruned=True)
  elif hp.emb_type != "rand":
    assert False, hp.emb_type

  v = Vocab()
  v.add_dataset(dataset_iter(TRAIN_FILENAME))
  v.add_dataset(dataset_iter(TEST_FILENAME))
  v.add_words(word2vec.keys())

  d_train = Dataset(v, dataset_iter(TRAIN_FILENAME))
  d_test = Dataset(v, dataset_iter(TEST_FILENAME))

  model = build_model(d_train, word2vec, hp)
  print(model.summary())

  x_train, y_train, x_test, y_test, history = train_model(
      model, d_train, d_test, hp)

  plotting.plot_history(history, filepath=_plot_history_filepath(hp))
  evaluate_model("train", model, d_train, x_train, y_train, hp)
  evaluate_model("test", model, d_test, x_test, y_test, hp)


def grid_search(hp=None):
  """Runs grid search."""
  if hp is None:
    hp = hparams_seq()
  grid = OrderedDict([
      ("emb_type", ["glove"]),
      ("emb_initializer", ["uniform"]),
      ("train_emb", [True]),
      ("use_viterbi_decoding", [True]),
      ("dropout_rate", [0.2])
  ])
  for values in itertools.product(*grid.values()):
    params = dict(zip(grid.keys(), values))
    print("\n" * 5)
    print("=" * 40)
    print(params)
    print("=" * 40)
    hp.set(**params)
    train_and_evaluate_model(hp)
