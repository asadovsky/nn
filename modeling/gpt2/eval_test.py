"""Tests for the eval module."""

import unittest

from modeling.gpt2.eval import Config, run


class EvalTest(unittest.TestCase):
    def test_run(self) -> None:
        run(Config(test_run=True))


if __name__ == "__main__":
    unittest.main()
