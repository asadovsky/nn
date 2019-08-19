"""ATIS intent prediction."""

from load_atis_yvchen import Dataset


def load_train():
  return Dataset('data/atis/atis-2.train.w-intent.iob')
