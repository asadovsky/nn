"""Tests for the linear_regression module."""

import unittest

from modeling.linear_regression import (
    ModelConfig,
    train_keras,
    train_torch,
)


class LinearRegressionTest(unittest.TestCase):
    def test_train_keras(self) -> None:
        train_keras(ModelConfig())

    def test_train_torch(self) -> None:
        train_torch(ModelConfig())


if __name__ == "__main__":
    unittest.main()
