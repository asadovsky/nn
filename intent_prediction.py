"""ATIS intent prediction."""

from __future__ import print_function
from collections import OrderedDict
import datetime
import itertools
import os

import numpy as np
from seqeval import metrics as seq_metrics
from sklearn import metrics as cls_metrics
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Bidirectional, concatenate, Dense, Dropout, Embedding, Input, GlobalAvgPool1D, GlobalMaxPool1D, LSTM, TimeDistributed
from tensorflow.keras.models import Model
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


def hparams_emb(**kwargs):
  """Returns hyperparams for an embedding layer."""
  p = Params()
  p.define("pretrained", "none",
           "Pretrained embedding type. Options: none, glove.")
  p.define("trainable", True,
           "Whether to update embeddings during training.")
  p.define("dim", 50,
           "Embedding size.")
  p.define("initializer", "uniform",
           "Embedding initializer.")
  p.set(**kwargs)
  return p


def hparams_model(**kwargs):
  """Returns hyperparams for a model."""
  p = Params()
  p.define("run_id", None,
           "String identifier for this run.")
  p.define("mode", "seq",
           "Sequence tagging or classification. Options: seq, cls.")
  p.define("max_len_words", 32,
           "Maximum sequence length in words.")
  p.define("max_len_chars", 8,
           "Maximum word length in chars. Used for char encoding.")
  p.define("padding", "post",
           "Options: pre, post.")
  p.define("truncating", "post",
           "Options: pre, post.")
  p.define("word_emb", hparams_emb(pretrained="glove"),
           "Word embedding params.")
  p.define("char_emb", hparams_emb(dim=10),
           "Char embedding params, or None to disable char encoding.")
  p.define("char_enc_dim", 20,
           "Size of char encoding layer.")
  p.define("seq_arch", "bilstm",
           "Architecture for sequence processing. Used for both sequence"
           " tagging and classification. Options: none, lstm, bilstm.")
  p.define("cls_arch", "avg_pool",
           "Architecture for classification. Used only for classification."
           " Options: avg_pool, max_pool.")
  p.define("hidden_dim", 50,
           "Size of hidden sequence processing layer.")
  # TODO: Experiment with dropout at different layers, e.g. recurrent_dropout
  # for LSTM layers and SpatialDropout1D for embeddings.
  p.define("dropout_rate", 0.2,
           "Dropout rate. If 0, we disable dropout.")
  p.define("drop_rare_words_freq", 0,
           "Drop training set words whose frequency is <= this value.")
  p.define("optimizer", "adam",
           "Model optimizer.")
  p.define("epochs", 50,
           "Number of epochs to train.")
  p.define("use_viterbi_decoding", True,
           "Whether to use Viterbi (vs. independent) decoding of IOB tags.")
  p.set(**kwargs)
  return p


def hparams_seq(**kwargs):
  p = hparams_model()
  p.char_emb = None
  p.set(**kwargs)
  return p


def hparams_cls(**kwargs):
  p = hparams_model()
  p.mode = "cls"
  p.char_emb = None
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


def _x_and_y(d, hp):
  """Returns model inputs and labels, i.e. x and y."""
  # <int>[num_examples, max_len_words]
  word_x = pad_sequences(
      d.word_id_seqs, maxlen=hp.max_len_words, value=d.word2id[PAD],
      padding=hp.padding, truncating=hp.truncating)
  x = [word_x]

  if hp.char_emb is not None:
    # Pad examples so that each one has hp.max_len_words words. Note, ideally
    # we'd use pad_sequences here but it doesn't support padding with arrays.
    examples = [example[:hp.max_len_words] +
                [[] for _ in range(hp.max_len_words - len(example))]
                for example in d.char_id_seqs]
    # <int>[num_examples, max_len_words, max_len_chars]
    char_x = np.stack([pad_sequences(
        words, maxlen=hp.max_len_chars, value=d.char2id[PAD],
        padding=hp.padding, truncating=hp.truncating) for words in examples])
    x = [word_x, char_x]

  if hp.mode == "seq":
    padded_tag_id_seqs = pad_sequences(
        d.tag_id_seqs, maxlen=hp.max_len_words,
        padding=hp.padding, truncating=hp.truncating,
        value=d.tag2id[PAD])
    # One-hot encoding.
    y = np.array([to_categorical(tag_ids, len(d.tag2id))
                  for tag_ids in padded_tag_id_seqs])
  elif hp.mode == "cls":
    y = to_categorical(d.intent_ids, len(d.intent2id))
  else:
    assert False, hp.mode

  return x, y


def _get_iob_seqs(tag_id_seqs, d):
  return [[d.id2tag[i] for i in seq] for seq in tag_id_seqs]


def _true_iob_seqs(d, hp):
  """Returns (possibly truncated) IOB sequences for the given dataset."""
  return [seq[:hp.max_len_words] for seq in _get_iob_seqs(d.tag_id_seqs, d)]


def _pred_iob_seqs(y, d, hp):
  """Returns IOB sequences for the given predictions on the given dataset."""
  # Predict the most likely tag at each sequence position.
  if hp.use_viterbi_decoding:
    seq_lengths = [min(len(seq), hp.max_len_words) for seq in d.tag_id_seqs]
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
  word_x = Input(shape=[hp.max_len_words])
  word_emb_mat = embedding_utils.make_embedding_matrix(
      d.id2word, word2vec, hp.word_emb.initializer)
  word_emb = Embedding(
      len(d.id2word), hp.word_emb.dim, weights=[word_emb_mat],
      mask_zero=True, input_length=hp.max_len_words,
      trainable=hp.word_emb.trainable)(word_x)
  x = [word_x]
  y = word_emb

  if hp.char_emb is not None:
    char_x = Input(shape=[hp.max_len_words, hp.max_len_chars])
    x = [word_x, char_x]
    char_emb = TimeDistributed(Embedding(
        len(d.id2char), hp.char_emb.dim,
        mask_zero=True, input_length=hp.max_len_chars,
        trainable=hp.char_emb.trainable))(char_x)
    char_enc = TimeDistributed(Bidirectional(LSTM(hp.char_enc_dim)))(char_emb)
    y = concatenate([word_emb, char_enc])

  if hp.seq_arch == "none":
    pass
  elif hp.seq_arch.endswith("lstm"):
    layer = LSTM(hp.hidden_dim, return_sequences=True)
    if hp.seq_arch == "bilstm":
      layer = Bidirectional(layer)
    else:
      assert hp.seq_arch == "lstm"
    y = layer(y)
  else:
    assert False, hp.arch

  if hp.dropout_rate > 0:
    y = Dropout(hp.dropout_rate)(y)

  if hp.mode == "seq":
    # TODO: Add CRF layer.
    y = TimeDistributed(Dense(len(d.tag2id), activation="softmax"))(y)
  elif hp.mode == "cls":
    layer = None
    if hp.cls_arch == "avg_pool":
      layer = GlobalAvgPool1D()
    elif hp.cls_arch == "max_pool":
      layer = GlobalMaxPool1D()
    else:
      assert False, hp.cls_arch
    # https://github.com/tensorflow/tensorflow/issues/33260
    assert layer.supports_masking
    y = layer(y)
    y = Dense(len(d.intent2id), activation="softmax")(y)
  else:
    assert False, hp.mode

  model = Model(x, y)
  model.compile(loss="categorical_crossentropy", optimizer=hp.optimizer,
                metrics=["acc"])
  return model


def train_model(model, d_train, d_test, hp):
  """Trains a model."""
  x_train, y_train = _x_and_y(d_train, hp)
  x_test, y_test = _x_and_y(d_test, hp)
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
  loss, acc = model.evaluate(x, y, verbose=0)
  print("{} model loss={:.4f} acc={:.4f}".format(prefix, loss, acc))

  if hp.mode == "seq":
    m = seq_metrics
    y_true, y_pred = (_true_iob_seqs(d, hp),
                      _pred_iob_seqs(model.predict(x), d, hp))
    report = m.classification_report(y_true, y_pred)
  elif hp.mode == "cls":
    m = cls_metrics
    y_true, y_pred = d.intent_ids, np.argmax(y, axis=-1)
    report = m.classification_report(y_true, y_pred,
                                     labels=range(len(d.id2intent)),
                                     target_names=d.id2intent)
  else:
    assert False, hp.mode

  print("{} {} acc={:.4f} f1={:.4f}".format(
      prefix, hp.mode, m.accuracy_score(y_true, y_pred),
      m.f1_score(y_true, y_pred, average="micro")))
  print(report)


def train_and_evaluate_model(hp):
  """Trains and evaluates a model."""
  hp.set(run_id=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  _record_hyperparams(hp)
  for path in [_checkpoints_dir(hp.run_id), _logs_dir(hp.run_id)]:
    os.makedirs(path, exist_ok=True)

  if hp.word_emb.pretrained == "none":
    word2vec = {}
  elif hp.word_emb.pretrained == "glove":
    word2vec = embedding_utils.read_glove(hp.word_emb.dim, pruned=True)
  else:
    assert False, hp.word_emb.pretrained

  v = Vocab()
  v.add_dataset(dataset_iter(TRAIN_FILENAME))
  v.add_dataset(dataset_iter(TEST_FILENAME))
  v.add_words(word2vec.keys())

  d_train = Dataset(v, dataset_iter(TRAIN_FILENAME))
  d_test = Dataset(v, dataset_iter(TEST_FILENAME))

  # Drop rare words so that UNK appears in the training set.
  d_train.drop_rare_words(hp.drop_rare_words_freq)

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
      ("word_emb.pretrained", ["glove"]),
      ("word_emb.trainable", [True]),
      ("word_emb.initializer", ["uniform"]),
      ("dropout_rate", [0.2]),
      ("use_viterbi_decoding", [True])
  ])
  for values in itertools.product(*grid.values()):
    params = dict(zip(grid.keys(), values))
    print("\n" * 5)
    print("=" * 40)
    print(params)
    print("=" * 40)
    hp.set(**params)
    train_and_evaluate_model(hp)
