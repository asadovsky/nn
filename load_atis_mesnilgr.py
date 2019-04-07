"""Utilities for loading ATIS data."""

import cPickle


def load_fold(fold):
  """Returns (train, valid, test, maps) for a given fold."""
  assert fold in range(5)
  with open('./data/atis/atis.fold%d.pkl' % fold, mode='rb') as f:
    return cPickle.load(f)


def print_fold(fold):
  """Prints ATIS data for the given fold."""
  train, _, test, m = load_fold(fold)
  w2idx, ne2idx, la2idx = m['words2idx'], m['tables2idx'], m['labels2idx']

  idx2w = dict((v, k) for k, v in w2idx.iteritems())
  idx2ne = dict((v, k) for k, v in ne2idx.iteritems())
  idx2la = dict((v, k) for k, v in la2idx.iteritems())

  train_w, train_ne, train_la = train
  test_w, test_ne, test_la = test  # pylint: disable=unused-variable

  col = 30
  for w, ne, la in zip(train_w, train_ne, train_la):
    print 'WORD'.rjust(col), 'TABLE'.rjust(col), 'LABEL'.rjust(col)
    for w_i, ne_i, la_i in zip(w, ne, la):
      print '%s %s %s' % (idx2w[w_i].rjust(col),
                          idx2ne[ne_i].rjust(col),
                          idx2la[la_i].rjust(col))
    print '\n' + '*' * (col * 3 + 2) + '\n'
