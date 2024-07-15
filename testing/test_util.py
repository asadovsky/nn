"""Defines testing utilities."""

import os
import unittest
from collections.abc import Callable


def slow_test(func: Callable) -> Callable:
    return unittest.skipUnless(
        os.environ.get("RUN_SLOW_TESTS", False), "skipping slow tests"
    )(func)
