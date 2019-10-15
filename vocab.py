"""Vocab representation."""

PAD = "<pad>"
UNK = "<unk>"


def _proc_tokens(tokens, token2id):
  for token in tokens:
    if token not in token2id:
      token2id[token] = len(token2id)


def _make_id2token(token2id):
  id2token = [""] * len(token2id)
  for token, i in token2id.items():
    id2token[i] = token
  return id2token


class Vocab:
  """A vocab."""

  def __init__(self):
    self.word2id = {PAD: 0, UNK: 1}
    self.char2id = {PAD: 0, UNK: 1}
    self.tag2id = {PAD: 0, UNK: 1}
    self.intent2id = {UNK: 0}

    self.id2word = []
    self.id2char = []
    self.id2tag = []
    self.id2intent = []

  def add_dataset(self, dataset_iter):
    """Adds words, chars, tags, and intents from the given dataset."""
    for words, tags, intent in dataset_iter:
      _proc_tokens(words, self.word2id)
      for word in words:
        _proc_tokens(list(word), self.char2id)
      _proc_tokens(tags, self.tag2id)
      _proc_tokens([intent], self.intent2id)

    self.id2word = _make_id2token(self.word2id)
    self.id2char = _make_id2token(self.char2id)
    self.id2tag = _make_id2token(self.tag2id)
    self.id2intent = _make_id2token(self.intent2id)

  def add_words(self, words):
    """Adds words from the given list."""
    _proc_tokens(words, self.word2id)

    self.id2word = _make_id2token(self.word2id)
