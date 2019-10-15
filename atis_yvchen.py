"""Utilities for reading yvchen's ATIS data."""

TRAIN_FILENAME = "data/atis/atis.train.w-intent.iob"
TEST_FILENAME = "data/atis/atis.test.w-intent.iob"


def dataset_iter(filename):
  """Returns a (words, tags, intent) iterator for the given file."""
  with open(filename) as f:
    for line in f:
      parts = line.split("\t")
      assert len(parts) == 2
      words = parts[0].strip().split()
      tags = parts[1].strip().split()
      assert len(words) == len(tags)
      # The final tag (for the final word, "EOS") is the intent name; replace it
      # with "O".
      intent = tags[-1]
      tags[-1] = "O"
      yield words, tags, intent
