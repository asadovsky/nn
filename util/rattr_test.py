"""Tests the rattr module."""

import dataclasses
import unittest

from util.rattr import rgetattr, rsetattr, rsetattrs, rsetattrs_from_str


@dataclasses.dataclass(slots=True)
class Foo:
    a: int = 0
    b: str = "b"
    c: bool = False


@dataclasses.dataclass(slots=True)
class Bar:
    a: Foo = dataclasses.field(default_factory=Foo)
    b: int = 2


class RattrTest(unittest.TestCase):
    def test_rgetattr(self) -> None:
        bar = Bar()
        self.assertEqual(rgetattr(bar, "b"), 2)
        self.assertEqual(rgetattr(bar, "a"), Foo())
        self.assertEqual(rgetattr(bar, "a.a"), 0)
        with self.assertRaises(AttributeError):
            rgetattr(bar, "")
        with self.assertRaises(AttributeError):
            rgetattr(bar, "z")
        with self.assertRaises(AttributeError):
            rgetattr(bar, "a.z")
        with self.assertRaises(AttributeError):
            rgetattr(bar, "z.a")

    def test_rgetattr_with_default(self) -> None:
        bar = Bar()
        self.assertEqual(rgetattr(bar, "b", 0), 2)
        self.assertEqual(rgetattr(bar, "a", 0), Foo())
        self.assertEqual(rgetattr(bar, "a.a", 1), 0)
        self.assertEqual(rgetattr(bar, "", 0), 0)
        self.assertEqual(rgetattr(bar, "z", 0), 0)
        self.assertEqual(rgetattr(bar, "a.z", 0), 0)
        self.assertEqual(rgetattr(bar, "z.a", 0), 0)

    def test_rsetattr(self) -> None:
        bar = Bar()
        rsetattr(bar, "b", 3)
        self.assertEqual(rgetattr(bar, "b"), 3)
        rsetattr(bar, "a.a", 4)
        self.assertEqual(rgetattr(bar, "a.a"), 4)
        rsetattr(bar, "a", Foo())
        self.assertEqual(rgetattr(bar, "a.a"), 0)
        with self.assertRaises(AttributeError):
            rsetattr(bar, "", 0)
        with self.assertRaises(AttributeError):
            rsetattr(bar, "z", 0)
        with self.assertRaises(AttributeError):
            rsetattr(bar, "a.z", 0)
        with self.assertRaises(AttributeError):
            rsetattr(bar, "z.a", 0)

    def test_rsetattrs(self) -> None:
        bar = Bar()
        rsetattrs(bar, [("b", 3), (("a.a"), 4)])
        self.assertEqual(rgetattr(bar, "b"), 3)
        self.assertEqual(rgetattr(bar, "a.a"), 4)

    def test_rsetattrs_from_str(self) -> None:
        bar = Bar()
        rsetattrs_from_str(bar, "b=3, a.b='a b', a.c=True")
        self.assertEqual(rgetattr(bar, "b"), 3)
        self.assertEqual(rgetattr(bar, "a.b"), "a b")
        rsetattrs_from_str(bar, "")
        with self.assertRaises(ValueError):
            rsetattrs_from_str(bar, "b=1.5")
        with self.assertRaises(ValueError):
            rsetattrs_from_str(bar, "a.b=baz")


if __name__ == "__main__":
    unittest.main()
