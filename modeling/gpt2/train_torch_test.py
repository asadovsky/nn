"""Tests for the train module."""

import os
import unittest
from tempfile import TemporaryDirectory

import torch

from modeling.gpt2.train_torch import Config, run


class TrainTest(unittest.TestCase):
    def test_run(self) -> None:
        with TemporaryDirectory() as dir0:
            run(Config(test_run=True, run_dir=dir0))
            self.assertTrue(os.path.exists(os.path.join(dir0, "log.txt")))
            for step in range(10):
                self.assertEqual(
                    os.path.exists(os.path.join(dir0, f"model_{step:06d}.pt")),
                    step in {4, 8, 9},
                )

    def test_ckpt(self) -> None:
        """Checks that starting from a checkpoint yields an identical result."""
        with TemporaryDirectory() as dir0, TemporaryDirectory() as dir1:
            # Train on CPU to exercise torch.compile.
            run(Config(test_run=True, run_dir=dir0, device="cpu"))
            run(
                Config(
                    ckpt=os.path.join(dir0, "model_000004.pt"),
                    test_run=True,
                    run_dir=dir1,
                    device="cpu",
                )
            )
            self.assertFalse(os.path.exists(os.path.join(dir1, "model_000004.pt")))
            ckpt0, ckpt1 = (
                torch.load(os.path.join(x, "model_000009.pt"), weights_only=True)
                for x in [dir0, dir1]
            )
            self.assertEqual(ckpt0["val_loss"], ckpt1["val_loss"])


if __name__ == "__main__":
    unittest.main()
