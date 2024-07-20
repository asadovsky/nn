"""Tests for the train module."""

import unittest

from modeling.gpt2.train import Config, run


class TrainTest(unittest.TestCase):
    def test_run(self) -> None:
        run(Config(test_run=True))

    def test_ckpt(self) -> None:
        pass  # FIXME


if __name__ == "__main__":
    unittest.main()
