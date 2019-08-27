"""Utilities for loading yvchen's ATIS data."""

# Reference:
# https://github.com/yvchen/JointSLU/blob/master/program/wordSlotDataSet.py

PAD = '<pad>'
UNK = '<unk>'


def _proc_tokens(tokens, token2id, update_token2id):
  token_ids = []
  for token in tokens:
    if token not in token2id:
      if update_token2id:
        token2id[token] = len(token2id)
      else:
        token = UNK
    token_ids.append(token2id[token])
  return token_ids


def _make_id2token(token2id):
  id2token = [''] * len(token2id)
  for token, i in iter(token2id.items()):
    id2token[i] = token
  return id2token


class Dataset:  # pylint: disable=too-few-public-methods
  """An ATIS dataset."""
  def __init__(self, filename, train_dataset=None):
    self.word_id_lists = []  # word lists (utterances), one per example
    self.tag_id_lists = []   # IOB tag lists, one per example
    self.intent_ids = []     # intents, one per example

    is_train = train_dataset is None
    if is_train:
      self.word2id = {PAD: 0, UNK: 1}
      self.tag2id = {UNK: 0}
      self.intent2id = {UNK: 0}
    else:
      self.word2id = train_dataset.word2id
      self.tag2id = train_dataset.tag2id
      self.intent2id = train_dataset.intent2id

    with open(filename) as f:
      for line in f:
        parts = line.split('\t')
        assert len(parts) == 2
        words = parts[0].strip().split()
        tags = parts[1].strip().split()
        assert len(words) == len(tags)
        self.word_id_lists.append(
            _proc_tokens(words, self.word2id, is_train))
        self.tag_id_lists.append(
            _proc_tokens(tags[:-1], self.tag2id, is_train))
        self.intent_ids.append(
            _proc_tokens([tags[-1]], self.intent2id, is_train)[0])

    self.id2word = _make_id2token(self.word2id)
    self.id2tag = _make_id2token(self.tag2id)
    self.id2intent = _make_id2token(self.intent2id)
