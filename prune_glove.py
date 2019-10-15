"""Generates pruned GloVe embedding files."""

from atis_yvchen import dataset_iter, TEST_FILENAME, TRAIN_FILENAME
import embedding_utils
from vocab import Vocab


def main():
  v = Vocab()
  v.add_dataset(dataset_iter(TRAIN_FILENAME))
  v.add_dataset(dataset_iter(TEST_FILENAME))
  for dim in [50, 100, 200, 300]:
    embedding_utils.prune_glove(dim, v.id2word)


if __name__ == "__main__":
  main()
