import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

x_vals = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
y_vals = np.linspace(0, 50, 50) + np.random.uniform(-4, 4, 50)
n = len(y_vals)

plt.scatter(x_vals, y_vals)
plt.show()

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None])
W = tf.get_variable("W", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable("b", shape=[], initializer=tf.zeros_initializer())
y_pred = tf.matmul(x, W) + b
loss = tf.reduce_sum((y_pred - tf.reshape(y, [-1, 1])) ** 2) / n

train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(1000):
    if epoch % 50 == 0:
      loss_val = sess.run(loss, feed_dict={x: x_vals.reshape([n, 1]), y: y_vals})
      print("epoch=", epoch, "loss=", loss_val, "W=", sess.run(W), "b=", sess.run(b))
    for (x_val, y_val) in zip(x_vals, y_vals):
      sess.run(train_op, feed_dict={x: np.reshape(x_val, [1, 1]), y: np.reshape(y_val, [1])})
