"""Linear regression examples."""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

np.random.seed(0)
tf.set_random_seed(0)

X_VALS = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
Y_VALS = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
N = len(Y_VALS)

plt.scatter(X_VALS, Y_VALS)
plt.show()

TRAIN_EPOCHS = 1000
LOG_EPOCHS = 100


def print_vars(estimator):
  print("*** Variables ***")
  for name in estimator.get_variable_names():
    print("  {}: {}".format(name, estimator.get_variable_value(name)))


def input_fn(mode):
  d = tf.data.Dataset.from_tensor_slices(({"x": tf.to_float(X_VALS)}, tf.to_float(Y_VALS)))
  d = d.batch(1)
  if mode == tf.estimator.ModeKeys.TRAIN:
    d = d.repeat()
  return d


def model_fn(features, labels, mode):
  x = tf.reshape(features["x"], [-1, 1])
  y = labels
  w = tf.get_variable("w", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable("b", shape=[], initializer=tf.zeros_initializer())
  pred = tf.reshape(tf.matmul(x, w) + b, [-1])
  loss = tf.reduce_sum((pred - y) ** 2) / tf.to_float(tf.size(pred))
  train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, predictions=pred, loss=loss, train_op=train_op)


def train_custom_estimator():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    config = tf.estimator.RunConfig(log_step_count_steps=(LOG_EPOCHS * N))
    estimator = tf.estimator.Estimator(model_fn, config=config)
    estimator.train(lambda: input_fn(tf.estimator.ModeKeys.TRAIN), max_steps=(TRAIN_EPOCHS * N))
    print_vars(estimator)
    print(estimator.evaluate(lambda: input_fn(tf.estimator.ModeKeys.EVAL)))


def train_linear_regressor():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    config = tf.estimator.RunConfig(log_step_count_steps=(LOG_EPOCHS * N))
    estimator = tf.estimator.LinearRegressor([tf.feature_column.numeric_column("x")], config=config)
    estimator.train(lambda: input_fn(tf.estimator.ModeKeys.TRAIN), max_steps=(TRAIN_EPOCHS * N))
    print_vars(estimator)
    print(estimator.evaluate(lambda: input_fn(tf.estimator.ModeKeys.EVAL)))


def train_ad_hoc():
  """Trains linear regression model with ad hoc training code."""
  tf.reset_default_graph()
  x = tf.placeholder(tf.float32, shape=[None, 1])
  y = tf.placeholder(tf.float32, shape=[None])
  w = tf.get_variable("w", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable("b", shape=[], initializer=tf.zeros_initializer())
  pred = tf.reshape(tf.matmul(x, w) + b, [-1])
  loss = tf.reduce_sum((pred - y) ** 2) / tf.to_float(tf.size(pred))
  train_op = tf.train.AdamOptimizer().minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    compute_loss = lambda: sess.run(loss, feed_dict={x: X_VALS.reshape([-1, 1]), y: Y_VALS})
    for epoch in range(TRAIN_EPOCHS):
      if epoch % LOG_EPOCHS == 0:
        print("epoch={} loss={} w={} b={}".format(epoch, compute_loss(), sess.run(w), sess.run(b)))
      for (x_val, y_val) in zip(X_VALS, Y_VALS):
        sess.run(train_op, feed_dict={x: np.reshape(x_val, [-1, 1]), y: np.reshape(y_val, [-1])})
    print("loss={} w={} b={}".format(compute_loss(), sess.run(w), sess.run(b)))
