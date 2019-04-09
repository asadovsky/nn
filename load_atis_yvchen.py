"""Utilities for loading yvchen's ATIS data."""

# Reference:
# https://github.com/yvchen/JointSLU/blob/master/program/wordSlotDataSet.py


def process_tokens(tokens, token2id, id2token):
  token_ids = []
  for token in tokens:
    if token not in token2id:
      token2id[token] = len(id2token)
      id2token.append(token)
    token_ids.append(token2id[token])
  return token_ids


# TODO: Add support for taking word and tag maps as input.
def load(filename):
  """Returns dataset dict."""
  word_id_lists = []  # word lists (utterances), one per example
  tag_id_lists = []   # IOB tag lists, one per example

  # Reserve index 0 for padding, 1 for unknown words/tokens.
  word2id = {'<pad>': 0, '<unk>': 1}
  tag2id = {'<pad>': 0, '<unk>': 1}
  id2word = ['<pad>', '<unk>']
  id2tag = ['<pad>', '<unk>']

  with open(filename, mode='rb') as f:
    for line in f:
      parts = line.split('\t')
      assert len(parts) == 2
      words = parts[0].strip().split()
      tags = parts[1].strip().split()
      assert len(words) == len(tags)
      word_ids = process_tokens(words, word2id, id2word)
      tag_ids = process_tokens(tags, tag2id, id2tag)
      word_id_lists.append(word_ids)
      tag_id_lists.append(tag_ids)

  return {'word_id_lists': word_id_lists,
          'tag_id_lists': tag_id_lists,
          'word2id': word2id,
          'tag2id': tag2id,
          'id2word': id2word,
          'id2tag': id2tag}
