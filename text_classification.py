"""Text classification examples."""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, GlobalAvgPool1D, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

np.random.seed(0)
tf.random.set_seed(0)

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
INPUTS = ["Well done!",
          "Good work",
          "Great effort",
          "nice work",
          "Excellent!",
          "Weak",
          "Poor effort!",
          "not good",
          "poor work",
          "Could have done better."]
LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

PAD = "<pad>"
UNK = "<unk>"

DEFAULT_HPARAMS = {
    "pad_len": 4,
    "emb_output_dim": 8,
    # Options: flatten, avg_pool, max_pool.
    "arch": "avg_pool",
    "categorical": True
}


def _rand_embedding(id2word, output_dim, mask_zero=False, input_length=None):
  """Returns an Embedding with random word vectors."""
  input_dim = len(id2word)
  return Embedding(input_dim, output_dim, mask_zero=mask_zero,
                   input_length=input_length)


def train_and_evaluate_model(hp):
  """Trains and evaluates a model."""
  t = Tokenizer(oov_token=UNK)
  t.fit_on_texts(INPUTS)
  id2word = [PAD] + list(t.index_word.values())
  word2id = {v: i for i, v in enumerate(id2word)}
  encoded_inputs = t.texts_to_sequences(INPUTS)
  padded_inputs = pad_sequences(encoded_inputs, maxlen=hp["pad_len"],
                                padding="post", truncating="post",
                                value=word2id[PAD])

  num_classes = len(set(LABELS))
  labels = np.array(LABELS)
  loss = "binary_crossentropy"
  if hp["categorical"]:
    labels = to_categorical(LABELS, num_classes)
    loss = "categorical_crossentropy"

  model = Sequential()

  embedding = _rand_embedding(id2word, hp["emb_output_dim"], mask_zero=True,
                              input_length=hp["pad_len"])
  model.add(embedding)

  if hp["arch"] == "flatten":
    model.add(Flatten())
  elif hp["arch"] == "avg_pool":
    model.add(GlobalAvgPool1D())
  elif hp["arch"] == "max_pool":
    model.add(GlobalMaxPool1D())
  else:
    assert False, hp["arch"]

  if hp["categorical"]:
    model.add(Dense(num_classes, activation="softmax"))
  else:
    model.add(Dense(1, activation="sigmoid"))

  model.compile(loss=loss, optimizer="adam", metrics=["acc"])
  print(model.summary())
  model.fit(padded_inputs, labels, epochs=50, verbose=0)
  loss, acc = model.evaluate(padded_inputs, labels, verbose=0)
  print("loss={:.4f} accuracy={:.4f}".format(loss, acc))
