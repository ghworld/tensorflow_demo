# coding:utf-8

import tensorflow as tf
import numpy as np

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

global_step = tf.Variable(0, name='global_step', trainable=False)
loss = tf.square(Y - tf.mul(X, w) - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
checkpoint_period = 100
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(1000):
        for (x, y) in zip(train_X, train_Y):
            sess.run(train_op, feed_dict={X: x, Y: y})

        if i % checkpoint_period == 0:
            saver.save(sess, "/home/mi/PYTHON_WORKSPACE/tensorflow_demo/model/linear_model.ckpt",
                       global_step=global_step)

    print(sess.run(w))
    print(sess.run(b))
