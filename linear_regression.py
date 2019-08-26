"""Linear regression examples."""

from __future__ import print_function

import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

X_VALS = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
Y_VALS = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
N = len(Y_VALS)

TRAIN_EPOCHS = 1000
LOG_EPOCHS = 100


def print_vars(estimator):
  print("*** Variables ***")
  for name in estimator.get_variable_names():
    print("  {}: {}".format(name, estimator.get_variable_value(name)))


def input_fn(mode):
  """Input function for estimators."""
  d = tf.data.Dataset.from_tensor_slices(
      ({"x": tf.to_float(X_VALS)}, tf.to_float(Y_VALS)))
  d = d.batch(1)
  if mode == tf.estimator.ModeKeys.TRAIN:
    d = d.repeat()
  return d


def model_fn(features, labels, mode):
  """Model function for estimators."""
  x = tf.reshape(features["x"], [-1, 1])
  y = labels
  w = tf.get_variable(
      "w", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable("b", shape=[], initializer=tf.zeros_initializer())
  pred = tf.reshape(tf.matmul(x, w) + b, [-1])
  loss = tf.reduce_sum((pred - y) ** 2) / tf.to_float(tf.size(pred))
  train_op = tf.train.AdamOptimizer().minimize(
      loss, global_step=tf.train.get_global_step())
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
  tf.reset_default_graph()
  x = tf.placeholder(tf.float32, shape=[None, 1])
  y = tf.placeholder(tf.float32, shape=[None])
  w = tf.get_variable(
      "w", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable("b", shape=[], initializer=tf.zeros_initializer())
  pred = tf.reshape(tf.matmul(x, w) + b, [-1])
  loss = tf.reduce_sum((pred - y) ** 2) / tf.to_float(tf.size(pred))
  train_op = tf.train.AdamOptimizer().minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    eval_feed_dict = {x: X_VALS.reshape([-1, 1]), y: Y_VALS}
    for epoch in range(TRAIN_EPOCHS):
      if epoch % LOG_EPOCHS == 0:
        print("epoch={} loss={} w={} b={}".format(
            epoch, *sess.run([loss, w, b], feed_dict=eval_feed_dict)))
      for (x_val, y_val) in zip(X_VALS, Y_VALS):
        sess.run(train_op,
                 feed_dict={x: np.reshape(x_val, [-1, 1]),
                            y: np.reshape(y_val, [-1])})
    print("loss={} w={} b={}".format(
        *sess.run([loss, w, b], feed_dict=eval_feed_dict)))
