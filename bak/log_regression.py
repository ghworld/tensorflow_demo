#!/usr/bin/env python
# coding=utf-8

import json
from datetime import datetime
import numpy as np
import tensorflow as tf



def log_regression():
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(X, w):
        return tf.matmul(X, w)
        # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    X = tf.placeholder("float", [None, 784])  # create symbolic variables
    Y = tf.placeholder("float", [None, 10])

    w = init_weights(
        [784, 10])  # like in linear regression, we need a shared variable weight matrix for logistic regression

    py_x = model(X, w)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(py_x, Y))  # compute mean cross entropy (softmax is applied internally)
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)  # construct optimizer
    predict_op = tf.argmax(py_x, 1)  # at predict time, evaluate the argmax of the logistic regression

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY}))


if __name__ == '__main__':
    log_regression()
