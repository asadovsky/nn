from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if False:
  tf.enable_eager_execution()

np.random.seed(0)
tf.set_random_seed(0)

x_vals = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
y_vals = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
n = len(y_vals)

if False:
  plt.scatter(x_vals, y_vals)
  plt.show()


def print_vars(estimator):
  print("*** Variables ***")
  for name in estimator.get_variable_names():
    print("  {}: {}".format(name, estimator.get_variable_value(name)))


def input_fn(mode):
  d = tf.data.Dataset.from_tensor_slices(({"x": tf.to_float(x_vals)}, tf.to_float(y_vals)))
  d = d.batch(1)
  if mode == tf.estimator.ModeKeys.TRAIN:
    d = d.repeat()
  return d


def model_fn(features, labels, mode, params):
  x = tf.reshape(features["x"], [-1, 1])
  y = labels
  W = tf.get_variable("W", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable("b", shape=[], initializer=tf.zeros_initializer())
  pred = tf.reshape(tf.matmul(x, W) + b, [-1])
  loss = tf.reduce_sum((pred - y) ** 2) / tf.to_float(tf.size(pred))
  train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, predictions=pred, loss=loss, train_op=train_op)


# Custom tf.Estimator.
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  config = tf.estimator.RunConfig(log_step_count_steps=(100 * n))
  estimator = tf.estimator.Estimator(model_fn, config=config, params={})
  estimator.train(lambda: input_fn(tf.estimator.ModeKeys.TRAIN), max_steps=(1000 * n))
  print_vars(estimator)
  print(estimator.evaluate(lambda: input_fn(tf.estimator.ModeKeys.EVAL)))


# LinearRegressor tf.Estimator.
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  config = tf.estimator.RunConfig(log_step_count_steps=(100 * n))
  estimator = tf.estimator.LinearRegressor([tf.feature_column.numeric_column("x")], config=config)
  estimator.train(lambda: input_fn(tf.estimator.ModeKeys.TRAIN), max_steps=(1000 * n))
  print_vars(estimator)
  print(estimator.evaluate(lambda: input_fn(tf.estimator.ModeKeys.EVAL)))


# Completely ad hoc.
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None])
W = tf.get_variable("W", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable("b", shape=[], initializer=tf.zeros_initializer())
pred = tf.reshape(tf.matmul(x, W) + b, [-1])
loss = tf.reduce_sum((pred - y) ** 2) / tf.to_float(tf.size(pred))
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  compute_loss = lambda: sess.run(loss, feed_dict={x: x_vals.reshape([-1, 1]), y: y_vals})
  for epoch in range(1000):
    if epoch % 100 == 0:
      print("epoch={} loss={} W={} b={}".format(epoch, compute_loss(), sess.run(W), sess.run(b)))
    for (x_val, y_val) in zip(x_vals, y_vals):
      sess.run(train_op, feed_dict={x: np.reshape(x_val, [-1, 1]), y: np.reshape(y_val, [-1])})
  print("loss={} W={} b={}".format(compute_loss(), sess.run(W), sess.run(b)))
