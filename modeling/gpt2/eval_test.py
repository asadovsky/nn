"""Tests for the eval module."""

import os
import unittest
from tempfile import TemporaryDirectory

from modeling.gpt2 import train_torch
from modeling.gpt2.eval import Config, run


class EvalTest(unittest.TestCase):
    def test_run(self) -> None:
        run(Config(test_run=True))

    def test_ckpt(self) -> None:
        with TemporaryDirectory() as dir0:
            # Train on CPU to exercise torch.compile.
            train_torch.run(
                train_torch.Config(test_run=True, run_dir=dir0, device="cpu")
            )
            run(Config(ckpt=os.path.join(dir0, "model_000009.pt"), test_run=True))


if __name__ == "__main__":
    unittest.main()
