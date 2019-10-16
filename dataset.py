"""Dataset representation."""

from collections import Counter

from vocab import UNK


def _proc_tokens(tokens, token2id):
  token_ids = []
  for token in tokens:
    if token not in token2id:
      token = UNK
    token_ids.append(token2id[token])
  return token_ids


class Dataset:
  """A dataset."""

  def __init__(self, vocab, dataset_iter):
    self._vocab = vocab
    self.word_id_seqs = []  # word seqs, one per example
    self.char_id_seqs = []  # char seqs, one per word per example
    self.tag_id_seqs = []   # IOB tag seqs, one per example
    self.intent_ids = []    # intents, one per example

    for words, tags, intent in dataset_iter:
      self.word_id_seqs.append(_proc_tokens(words, vocab.word2id))
      self.char_id_seqs.append(
          [_proc_tokens(list(word), vocab.char2id) for word in words])
      self.tag_id_seqs.append(_proc_tokens(tags, vocab.tag2id))
      self.intent_ids.append(_proc_tokens([intent], vocab.intent2id)[0])

  def drop_rare_words(self, max_freq_to_drop):
    """Drops words whose frequency is <= `max_freq_to_drop`."""
    if max_freq_to_drop == 0:
      return
    word_ids = sum(self.word_id_seqs, [])
    to_drop = set()
    for word_id, count in Counter(word_ids).items():
      if count / len(word_ids) <= max_freq_to_drop:
        to_drop.add(word_id)
    print("Dropping {} words".format(len(to_drop)))
    unk_word_id = self.word2id[UNK]
    for seq in self.word_id_seqs:
      for i, v in enumerate(seq):
        if v in to_drop:
          seq[i] = unk_word_id

  @property
  def word2id(self):
    return self._vocab.word2id

  @property
  def char2id(self):
    return self._vocab.char2id

  @property
  def tag2id(self):
    return self._vocab.tag2id

  @property
  def intent2id(self):
    return self._vocab.intent2id

  @property
  def id2word(self):
    return self._vocab.id2word

  @property
  def id2char(self):
    return self._vocab.id2char

  @property
  def id2tag(self):
    return self._vocab.id2tag

  @property
  def id2intent(self):
    return self._vocab.id2intent
