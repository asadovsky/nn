"""Utilities for reading and writing nested attributes."""

import ast
from operator import attrgetter
from typing import Any


def rgetattr(obj: object, path: str, *default: Any) -> Any:
    """Like getattr, but supports nested attributes."""
    try:
        return attrgetter(path)(obj)
    except AttributeError:
        if default:
            return default[0]
        raise


def rsetattr(obj: object, path: str, value: Any) -> None:
    """Like setattr, but supports nested attributes."""
    parts = path.rsplit(".", 1)
    if len(parts) > 1:
        obj, path = rgetattr(obj, parts[0]), parts[1]
    setattr(obj, path, value)


def rsetattrs(obj: object, attrs: list[tuple[str, Any]]) -> None:
    """Sets the given attributes."""
    for path, value in attrs:
        rsetattr(obj, path, value)


def rsetattrs_from_str(obj: object, attrs_str: str) -> None:
    """Sets the given comma-separated path=value attributes."""
    if not attrs_str:
        return
    attrs = []
    for path, value_str in [x.strip().split("=") for x in attrs_str.split(",")]:
        try:
            value = ast.literal_eval(value_str)
        except ValueError:
            raise ValueError(f"{path}: malformed literal {value_str}") from None
        typ = type(rgetattr(obj, path))
        if not isinstance(value, typ):
            raise ValueError(f"{path}: expected {typ}, got {type(value)}") from None
        attrs.append((path, value))
    rsetattrs(obj, attrs)
