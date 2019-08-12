"""Text classification examples."""

from __future__ import print_function

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

GLOVE_PATH = 'data/glove/glove.6B.100d.txt'
GLOVE_DIM = 100

# Source: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
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


def make_rand_embedding(tokenizer):
  """Returns an Embedding with random word vectors."""
  vocab_size = len(tokenizer.word_index) + 1
  return Embedding(vocab_size, 8, input_length=PAD_LEN)


def make_glove_embedding(tokenizer):
  """Returns an Embedding with GloVe word vectors."""
  vocab_size = len(tokenizer.word_index) + 1
  word2vec = dict()
  with open(GLOVE_PATH) as f:
    for line in f:
      parts = line.split()
      word2vec[parts[0]] = np.asarray(parts[1:], dtype='float32')
  print('Loaded %d word vectors' % len(word2vec))
  embedding_matrix = np.zeros((vocab_size, GLOVE_DIM))
  for word, i in tokenizer.word_index.items():
    vec = word2vec.get(word)
    if vec is not None:
      embedding_matrix[i] = vec
  return Embedding(vocab_size, GLOVE_DIM, weights=[embedding_matrix],
                   input_length=PAD_LEN, trainable=False)


def train_simple(use_glove):
  """Trains a simple model."""
  t = Tokenizer()
  t.fit_on_texts(DOCS)
  encoded_docs = t.texts_to_sequences(DOCS)
  padded_docs = pad_sequences(encoded_docs, maxlen=PAD_LEN, padding='post')

  embedding = None
  if use_glove:
    embedding = make_glove_embedding(t)
  else:
    embedding = make_rand_embedding(t)

  model = Sequential()
  model.add(embedding)
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
  print(model.summary())
  model.fit(padded_docs, LABELS, epochs=50, verbose=0)
  loss, accuracy = model.evaluate(padded_docs, LABELS, verbose=0)
  print('loss=%f accuracy=%f' % (loss, accuracy * 100))
