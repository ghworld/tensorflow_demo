# coding=utf-8

import tensorflow as tf
import numpy as np


# hello world!
def hello():
    hello_op = tf.constant('hello ,TensorFlow!')
    a = tf.constant(10)
    b = tf.constant(32)
    compute_op = tf.add(a, b)
    with tf.Session() as sess:
        print sess.run(hello_op)
        print sess.run(compute_op)
    sess.close()


# liner_regression
def liner_regression():
    # Prepare train data
    train_X = np.linspace(-1, 1, 100)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

    # Define the model
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    w = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="bias")
    loss = tf.square(Y - tf.mul(X, w) - b)  # 最小二剩
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # Create session to run
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        epoch = 1
        for i in range(10):
            for (x, y) in zip(train_X, train_Y):
                _, w_value, b_value = sess.run([train_op, w, b],
                                               feed_dict={X: x,
                                                          Y: y})
            print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
            epoch += 1


# global_step
def lineer_regression_global_step():
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

        epoch = 1
        for i in range(1000):
            for (x, y) in zip(train_X, train_Y):
                sess.run(train_op, feed_dict={X: x, Y: y})

            # print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
            # epoch += 1

            if i % checkpoint_period == 0:
                saver.save(sess, "./model/linear_model.ckpt",
                           global_step=global_step)

        print(sess.run(w))
        print(sess.run(b))


if __name__ == '__main__':
    hello()
    liner_regression()
    lineer_regression_global_step()
