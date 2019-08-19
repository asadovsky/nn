"""Utilities for loading yvchen's ATIS data."""

# Reference:
# https://github.com/yvchen/JointSLU/blob/master/program/wordSlotDataSet.py


def _process_tokens(tokens, token2id, id2token):
  token_ids = []
  for token in tokens:
    if token not in token2id:
      token2id[token] = len(id2token)
      id2token.append(token)
    token_ids.append(token2id[token])
  return token_ids


class Dataset(object):  # pylint: disable=too-few-public-methods
  """An ATIS dataset."""
  # TODO: Add support for taking word and tag maps as input.
  def __init__(self, filename):
    self.word_id_lists = []  # word lists (utterances), one per example
    self.tag_id_lists = []   # IOB tag lists, one per example
    self.intent_ids = []     # intents, one per example

    # Reserve index 0 for padding, 1 for unknown words/tokens.
    self.word2id = {'<pad>': 0, '<unk>': 1}
    self.tag2id = {'<pad>': 0, '<unk>': 1}
    self.intent2id = dict()
    self.id2word = ['<pad>', '<unk>']
    self.id2tag = ['<pad>', '<unk>']
    self.id2intent = []

    with open(filename, mode='rb') as f:
      for line in f:
        parts = line.split('\t')
        assert len(parts) == 2
        words = parts[0].strip().split()
        tags = parts[1].strip().split()
        assert len(words) == len(tags)
        self.word_id_lists.append(
            _process_tokens(words, self.word2id, self.id2word))
        self.tag_id_lists.append(
            _process_tokens(tags[:-1], self.tag2id, self.id2tag))
        self.intent_ids.append(
            _process_tokens([tags[-1]], self.intent2id, self.id2intent)[0])
