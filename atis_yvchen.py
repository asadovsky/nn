"""Utilities for loading yvchen's ATIS data."""

# Reference:
# https://github.com/yvchen/JointSLU/blob/master/program/wordSlotDataSet.py

PAD = "<pad>"
UNK = "<unk>"


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
  id2token = [""] * len(token2id)
  for token, i in iter(token2id.items()):
    id2token[i] = token
  return id2token


class Dataset:  # pylint: disable=too-few-public-methods
  """An ATIS dataset."""

  def __init__(self, filename, vocab_dataset=None):
    self.word_id_seqs = []  # word seqs (utterances), one per example
    self.tag_id_seqs = []   # IOB tag seqs, one per example
    self.intent_ids = []    # intents, one per example

    extend_vocab = vocab_dataset is None
    if extend_vocab:
      self.word2id = {PAD: 0, UNK: 1}
      self.tag2id = {PAD: 0, UNK: 1}
      self.intent2id = {UNK: 0}
    else:
      self.word2id = vocab_dataset.word2id
      self.tag2id = vocab_dataset.tag2id
      self.intent2id = vocab_dataset.intent2id

    self._process_file(filename, extend_vocab, True)

  def extend_vocab(self, filename):
    """Adds examples from `filename` to vocab."""
    self._process_file(filename, True, False)

  def _process_file(self, filename, extend_vocab, extend_data):
    """Adds examples from `filename` to vocab and/or data."""
    with open(filename) as f:
      for line in f:
        parts = line.split("\t")
        assert len(parts) == 2
        words = parts[0].strip().split()
        tags = parts[1].strip().split()
        assert len(words) == len(tags)
        word_id_seq = _proc_tokens(words, self.word2id, extend_vocab)
        # Replace the intent name with "O" in the tag seq so that the word and
        # tag seqs have the same length.
        # TODO: Add an option to keep the intent name as the final tag, which
        # corresponds to the final word ("EOS"), so we can jointly predict the
        # tags and intent as done in the yvchen paper.
        tag_id_seq = _proc_tokens(tags[:-1] + ["O"], self.tag2id, extend_vocab)
        intent_id = _proc_tokens([tags[-1]], self.intent2id, extend_vocab)[0]
        if extend_data:
          self.word_id_seqs.append(word_id_seq)
          self.tag_id_seqs.append(tag_id_seq)
          self.intent_ids.append(intent_id)

    self.id2word = _make_id2token(self.word2id)
    self.id2tag = _make_id2token(self.tag2id)
    self.id2intent = _make_id2token(self.intent2id)
