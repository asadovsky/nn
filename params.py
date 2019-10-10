"""A simplified version of tensorflow/lingvo's Params class."""

import pickle
import re
import six


class _Param:
  """A parameter."""

  def __init__(self, default_value, description):
    self._value = default_value
    self._description = description

  def __eq__(self, other):
    # pylint: disable=protected-access
    return self._value == other._value

  def __ne__(self, other):
    return not self == other

  def to_string(self, depth):
    if isinstance(self._value, Params):
      # pylint: disable=protected-access
      return self._value._to_string(depth)
    if isinstance(self._value, six.string_types):
      return '"%s"' % str(self._value)
    return str(self._value)

  def set(self, value):
    self._value = value

  def get(self):
    return self._value

  def description(self):
    return self._description


class Params:
  """A data structure for parameters."""

  def __init__(self):
    self._params = {}

  def __getattr__(self, name):
    if name == '_params':
      return self.__dict__[name]
    try:
      return self._params[name].get()
    except KeyError:
      raise AttributeError(name) from None

  def __setattr__(self, name, value):
    if name == '_params':
      self.__dict__[name] = value
    else:
      try:
        self._params[name].set(value)
      except KeyError:
        raise AttributeError(name) from None

  def __dir__(self):
    return self._params.keys()

  def __eq__(self, other):
    # pylint: disable=protected-access
    return isinstance(other, Params) and self._params == other._params

  def __ne__(self, other):
    return not self == other

  def _to_string(self, depth):
    param_strs = ['%s%s: %s' % ('  ' * (depth + 1), k, v.to_string(depth + 1))
                  for k, v in six.iteritems(self._params)]
    return '{\n%s\n%s}' % ('\n'.join(sorted(param_strs)), '  ' * depth)

  def __str__(self):
    return self._to_string(0)

  def __repr__(self):
    return self._to_string(0)

  def define(self, name, default_value, description):
    """Defines a parameter.

    Args:
      name: Parameter name. Must start with a lowercase letter and may only
          contain lowercase letters, numbers, and underscores.
      default_value: Default value for this parameter.
      description: String description of this parameter.

    Raises:
      AttributeError: If parameter `name` is already defined.
    """
    assert (name is not None and
            isinstance(name, six.string_types) and
            re.match('^[a-z][a-z0-9_]*$', name) is not None)
    if name in self._params:
      raise AttributeError('Parameter is already defined: %s' % name)
    self._params[name] = _Param(default_value, description)

  def _get_nested(self, name):
    """Returns the specified parameter."""
    parts = name.split('.')
    curr = self
    for i, part in enumerate(parts):
      try:
        if not isinstance(curr, Params):
          raise KeyError
        param = curr._params[part]  # pylint: disable=protected-access
        if i == len(parts) - 1:
          return param
        curr = param.get()
      except KeyError:
        raise AttributeError('.'.join(parts[:i+1])) from None
    return None

  def get(self, name):
    """Returns the value of the specified parameter."""
    return self._get_nested(name).get()

  def set(self, **kwargs):
    """Sets the specified parameters."""
    for name, value in six.iteritems(kwargs):
      self._get_nested(name).set(value)

  def has(self, name):
    """Returns True if the specified parameter exists."""
    try:
      self._get_nested(name)
      return True
    except AttributeError:
      return False

  def description(self, name):
    """Returns the description of the specified parameter."""
    return self._get_nested(name).description()

  def dumps(self):
    """Serializes this object to bytes."""
    return pickle.dumps(self)

  @staticmethod
  def loads(s):
    """Deserializes a Params object from bytes."""
    return pickle.loads(s)
