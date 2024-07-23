"""Tests for the linear_regression module."""

import unittest

from modeling.linear_regression import Config, train_jax, train_keras, train_torch


class LinearRegressionTest(unittest.TestCase):
    def test_train_keras(self) -> None:
        train_keras(Config())

    def test_train_torch(self) -> None:
        train_torch(Config())

    def test_train_jax(self) -> None:
        train_jax(Config())


if __name__ == "__main__":
    unittest.main()
