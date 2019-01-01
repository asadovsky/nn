import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

x_vals = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
y_vals = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)

plt.scatter(x_vals, y_vals)
plt.show()


def make_input_fn(mode):

  def input_fn():
    d = tf.data.Dataset.from_tensor_slices(({"x": tf.to_float(x_vals)}, tf.to_float(y_vals)))
    d = d.batch(1)
    if mode == tf.estimator.ModeKeys.TRAIN:
      d = d.repeat()
    return d

  return input_fn


def make_model_fn():

  def model_fn(features, labels, mode, params):
    x = tf.reshape(features["x"], [-1, 1])
    y = labels
    W = tf.get_variable("W", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=[], initializer=tf.zeros_initializer())
    pred = tf.reshape(tf.matmul(x, W) + b, [-1])
    loss = tf.reduce_sum((pred - y) ** 2) / tf.to_float(tf.size(pred))
    train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, predictions=pred, loss=loss, train_op=train_op)

  return model_fn


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  estimator = tf.estimator.Estimator(make_model_fn(), params={})
  estimator.train(make_input_fn(tf.estimator.ModeKeys.TRAIN), steps=1000)
  print("W=", estimator.get_variable_value("W"), "b=", estimator.get_variable_value("b"))
  # TODO: Evaluation returns the correct loss value (computed over the entire
  # dataset), but training fails to optimize the variable values. Could it be
  # that every training iteration resets the variable values?
  print(estimator.evaluate(make_input_fn(tf.estimator.ModeKeys.EVAL)))


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
  for epoch in range(1000):
    if epoch % 50 == 0:
      loss_val = sess.run(loss, feed_dict={x: x_vals.reshape([-1, 1]), y: y_vals})
      print("epoch=", epoch, "loss=", loss_val, "W=", sess.run(W), "b=", sess.run(b))
    for (x_val, y_val) in zip(x_vals, y_vals):
      sess.run(train_op, feed_dict={x: np.reshape(x_val, [-1, 1]), y: np.reshape(y_val, [-1])})
