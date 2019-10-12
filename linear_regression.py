"""Linear regression examples."""

from __future__ import print_function

import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

X_VALS = (np.linspace(0, 50, 50) +
          np.random.uniform(-4, 4, 50)).astype(np.float32)
Y_VALS = (np.linspace(0, 50, 50) +
          np.random.uniform(-4, 4, 50)).astype(np.float32)
N = len(Y_VALS)

TRAIN_EPOCHS = 2000
LOG_EPOCHS = 100


def print_vars(estimator):
  print("*** Variables ***")
  for name in estimator.get_variable_names():
    print("  {}: {}".format(name, estimator.get_variable_value(name)))


def input_fn(mode):
  """Input function for estimators."""
  d = tf.data.Dataset.from_tensor_slices(({"x": X_VALS}, Y_VALS))
  d = d.batch(1)
  if mode == tf.estimator.ModeKeys.TRAIN:
    d = d.repeat()
  return d


def model_fn(features, labels, mode):
  """Model function for estimators."""
  x = tf.reshape(features["x"], [-1, 1])
  y = labels
  w = tf.Variable(tf.initializers.GlorotUniform()(shape=[1, 1]), name="w")
  b = tf.Variable(tf.zeros(shape=[]), name="b")
  trainable_variables = [w, b]
  pred = tf.reshape(tf.matmul(x, w) + b, [-1])
  loss = tf.reduce_sum((pred - y) ** 2) / tf.cast(tf.size(pred), tf.float32)
  optimizer = tf.keras.optimizers.Adam()
  optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
  train_op = optimizer.get_updates(loss, trainable_variables)[0]
  return tf.estimator.EstimatorSpec(
      mode=mode, predictions=pred, loss=loss, train_op=train_op)


def train_custom_estimator():
  """Trains a custom estimator."""
  config = tf.estimator.RunConfig(log_step_count_steps=(LOG_EPOCHS * N))
  estimator = tf.estimator.Estimator(model_fn, config=config)
  train_spec = tf.estimator.TrainSpec(
      lambda: input_fn(tf.estimator.ModeKeys.TRAIN),
      max_steps=(TRAIN_EPOCHS * N))
  eval_spec = tf.estimator.EvalSpec(
      lambda: input_fn(tf.estimator.ModeKeys.EVAL))
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  print_vars(estimator)


def train_linear_regressor():
  """Trains a LinearRegressor estimator."""
  config = tf.estimator.RunConfig(log_step_count_steps=(LOG_EPOCHS * N))
  estimator = tf.estimator.LinearRegressor(
      [tf.feature_column.numeric_column("x")], config=config)
  train_spec = tf.estimator.TrainSpec(
      lambda: input_fn(tf.estimator.ModeKeys.TRAIN),
      max_steps=(TRAIN_EPOCHS * N))
  eval_spec = tf.estimator.EvalSpec(
      lambda: input_fn(tf.estimator.ModeKeys.EVAL))
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  print_vars(estimator)


def train_ad_hoc():
  """Trains linear regression model using ad hoc training code."""
  w = tf.Variable(tf.initializers.GlorotUniform()(shape=[1, 1]), name="w")
  b = tf.Variable(tf.zeros(shape=[]), name="b")
  trainable_variables = [w, b]
  optimizer = tf.keras.optimizers.Adam()

  def _loss(x, y):
    pred = tf.reshape(tf.matmul(x, w) + b, [-1])
    return tf.reduce_sum((pred - y) ** 2) / tf.cast(tf.size(pred), tf.float32)

  def _loss_and_vars_str():
    return "loss={:.4f} w={} b={}".format(
        _loss(X_VALS.reshape([-1, 1]), Y_VALS), w.numpy(), b.numpy())

  @tf.function
  def _train_step(x, y):
    with tf.GradientTape() as tape:
      loss = _loss(x, y)
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

  for epoch in range(TRAIN_EPOCHS):
    if epoch % LOG_EPOCHS == 0:
      print("epoch={} {}".format(epoch, _loss_and_vars_str()))
    for x_val, y_val in zip(X_VALS, Y_VALS):
      _train_step(np.reshape(x_val, [-1, 1]), np.reshape(y_val, [-1]))

  print(_loss_and_vars_str())
