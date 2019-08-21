"""Utilities for loading mesnilgr's ATIS data."""

from __future__ import print_function
import cPickle


def load_fold(fold):
  """Returns (train, valid, test, maps) for the given fold."""
  assert fold in range(5)
  with open('./data/atis/atis.fold%d.pkl' % fold, mode='rb') as f:
    return cPickle.load(f)


def print_fold(fold):
  """Prints ATIS data for the given fold."""
  train, _, test, m = load_fold(fold)
  w2id, ne2id, la2id = m['words2idx'], m['tables2idx'], m['labels2idx']

  id2w = dict((v, k) for k, v in w2id.iteritems())
  id2ne = dict((v, k) for k, v in ne2id.iteritems())
  id2la = dict((v, k) for k, v in la2id.iteritems())

  train_w, train_ne, train_la = train
  test_w, test_ne, test_la = test  # pylint: disable=unused-variable

  col = 30
  for w, ne, la in zip(train_w, train_ne, train_la):
    print('WORD'.rjust(col), 'TABLE'.rjust(col), 'LABEL'.rjust(col))
    for w_i, ne_i, la_i in zip(w, ne, la):
      print('{} {} {}'.format(id2w[w_i].rjust(col),
                              id2ne[ne_i].rjust(col),
                              id2la[la_i].rjust(col)))
    print('\n' + '*' * (col * 3 + 2) + '\n')
