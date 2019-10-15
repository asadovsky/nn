"""Tests for intent_prediction module."""

import unittest

from intent_prediction import hparams_cls, hparams_seq, train_and_evaluate_model


class TestIntentPrediction(unittest.TestCase):
  """Tests for intent_prediction module."""

  @staticmethod
  def test_cls():
    """Tests intent prediction (classification)."""
    train_and_evaluate_model(hparams_cls(epochs=3))

  @staticmethod
  def test_seq():
    """Tests slot prediction (sequence tagging)."""
    train_and_evaluate_model(hparams_seq(epochs=3))


if __name__ == "__main__":
  unittest.main()
