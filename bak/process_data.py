# /usr/bin/env python


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def data_process():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_dir", "./data/mnist/input_data", "the mnist data dir")

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print mnist.train.images[0], type(mnist.train.images[0]), len(mnist.train.images[0])


def print_tfrecords():
    ""
    input_filename = "/home/mi/GIT_WORKSPACE/deep_recommend_system" + "/data/cancer/cancer_train.csv.tfrecords"
    tfrecords = tf.python_io.tf_record_iterator(input_filename)

    # for serialized_example in tf.python_io.tf_record_iterator(input_filename):
    #     example=tf.train.Example()
    #     example.ParseFromString(serialized_example)
    #     label = example.features.feature["label"].float_list.value
    #     features=example.features.feature["features"].float_list.value
    #     print label,features


def np_write():
    import numpy as np
    outfile = "outfile"
    x = np.arange(10)
    np.save(outfile, x)


def np_read():
    import numpy as np
    outfile = "outfile.npy"
    data = np.load(outfile)


def test():
    ""
    x_input = tf.placeholder(tf.float32, [10, 10], name="input")
    print x_input.op


if __name__ == '__main__':
    # data_process()
    # print_tfrecords()
    # np_write()
    # np_read()
    ""
    test()
