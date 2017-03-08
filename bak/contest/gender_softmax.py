#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np

IRIS_TRAINING = "/home/mi/PYTHON_WORKSPACE/temp/deep/iris_training.csv"
IRIS_TEST = "/home/mi/PYTHON_WORKSPACE/temp/deep/iris_test.csv"


def main():
    ""
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST, target_dtype=np.int,
                                                                   features_dtype=np.float)



    x = tf.placeholder(tf.float32, [None, 4])
    W = tf.Variable(tf.zeros([4, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 1])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for v, t in zip(training_set.data, training_set.target):
        _, W_value, b_value = sess.run([train_step, W, b], feed_dict={x: [v], y_: [[t]]})
        # print W_value, b_value

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for v, t in zip(test_set.data, test_set.target):
        _, W_value, b_value = sess.run([train_step, W, b], feed_dict={x: [v], y_: [[t]]})
        # print W_value, b_value
        sess.run(accuracy, feed_dict={x: [v], y_: [[t]]})


def test():
    ""
    x = [[1., 1.], [2., 2.]]
    x = 1.0
    y = 2.0

    arr = np.array([[31, 23, 4, 24, 27, 34],
                    [18, 3, 25, 0, 6, 35],
                    [28, 14, 33, 22, 20, 8],
                    [13, 30, 21, 19, 7, 9],
                    [16, 1, 26, 32, 2, 29],
                    [17, 12, 5, 11, 10, 15]])

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    print sess.run(tf.arg_max(arr,1))
    print sess.run(tf.rank(arr))
    print tf.rank(arr).eval()
    # print sess.run(tf.reduce_mean(x))
    # print sess.run(tf.div(x,y))

    # tf.reduce_mean(x, 0)
    # tf.reduce_mean(x, 1)


    # x=[[1.,1.],[2.,2.]]
    # with tf.Session() as sess:
    #     print tf.reduce_mean(x)


if __name__ == '__main__':
    main()
    # test()
