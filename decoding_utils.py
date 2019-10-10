"""Utilities for decoding the output of a model."""

from typing import Any, List, Text, Tuple, Union

import numpy as np
import tensorflow as tf


def _split_iob_tag(tag: Text) -> Tuple[Text, Union[Text, None]]:
  """Handles tags <pad>, <unk>, O, B-foo, and I-foo."""
  if tag[:2] not in ['B-', 'I-']:
    return tag, None
  return tuple(tag.split('-', maxsplit=2))


def iob_transition_params(tags: List[Text]) -> Any:
  """Returns transition matrix suitable for tf.contrib.crf.crf_decode."""
  mat = np.zeros((len(tags), len(tags)))
  for i, i_tag in enumerate(tags):
    i_type, i_name = _split_iob_tag(i_tag)
    for j, j_tag in enumerate(tags):
      j_type, j_name = _split_iob_tag(j_tag)
      if ((i_type == 'O' and j_type == 'I') or
          (i_type in ['B', 'I'] and j_type == 'I' and i_name != j_name)):
        mat[i][j] = -np.inf
  return tf.constant(mat, dtype=tf.float32)
