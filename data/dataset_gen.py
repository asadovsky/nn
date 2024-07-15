"""Defines DatasetGen."""

from collections.abc import Generator

# Generates (words, tags, intent) tuples.
DatasetGen = Generator[tuple[list[str], list[str], str], None, None]
