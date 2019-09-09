"""Tests for Params class."""

import unittest

from params import Params


class TestParams(unittest.TestCase):
  """Tests for Params class."""

  def test_eq_ne(self):
    """Tests equality methods."""
    a = Params()
    b = Params()
    self.assertEqual(a, b)
    a.define('x', 0, '')
    self.assertNotEqual(a, b)
    b.define('x', 0, '')
    self.assertEqual(a, b)
    a.x = 1
    self.assertNotEqual(a, b)
    a.x = 0
    self.assertEqual(a, b)
    c = Params()
    c.define('y', 0, '')
    self.assertNotEqual(b, c)

  def test_get_set_has(self):
    """Tests get, set, and has methods."""
    p = Params()
    self.assertFalse(p.has('x'))
    self.assertRaises(AttributeError, lambda: p.x)
    self.assertRaises(AttributeError, lambda: p.get('x'))
    p.define('x', 0, '')
    self.assertTrue(p.has('x'))
    self.assertEqual(p.x, 0)
    self.assertEqual(p.get('x'), 0)
    p.x = 1
    self.assertTrue(p.has('x'))
    self.assertEqual(p.x, 1)
    self.assertEqual(p.get('x'), 1)
    p.set(x=2)
    self.assertTrue(p.has('x'))
    self.assertEqual(p.x, 2)
    self.assertEqual(p.get('x'), 2)

  def test_description(self):
    """Tests description method."""
    p = Params()
    p.define('x', 0, 'x desc')
    p.define('y', 1, '')
    self.assertEqual(p.description('x'), 'x desc')
    self.assertEqual(p.description('y'), '')
    self.assertRaises(AttributeError, lambda: p.description('z'))

  def test_double_define(self):
    """Tests that defining the same parameter twice fails."""
    p = Params()
    p.define('x', 0, '')
    self.assertRaises(AttributeError, lambda: p.define('x', 0, ''))

  def test_nested(self):
    """Tests for nested Params."""
    a = Params()
    a.define('x', 0, '')
    self.assertEqual(str(a), '{\n  x: 0\n}')
    b = Params()
    a.define('y', b, '')
    b.define('z', 1, '')
    self.assertEqual(str(a), '{\n  x: 0\n  y: {\n    z: 1\n  }\n}')
    self.assertRaises(AttributeError, lambda: a.get('w'))
    self.assertEqual(a.get('x'), 0)
    self.assertEqual(type(a.get('y')), Params)
    self.assertRaises(AttributeError, lambda: a.get('y.w'))
    self.assertEqual(a.get('y.z'), 1)
    self.assertRaises(AttributeError, lambda: a.get('y.z.w'))


if __name__ == '__main__':
  unittest.main()
