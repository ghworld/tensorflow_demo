#!/usr/bin/env python
# coding=utf-8


from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = os.path.join(os.path.dirname(__file__), "./data/iris_training.csv")
IRIS_TEST = os.path.join(os.path.dirname(__file__), "./data/iris_test.csv")


def main():
    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    validation_metrics = {
        "accuracy": MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key="classes"),
        "recall": MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key="classes"),
        "precision": MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key="classes")
    }
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_set.data,
        test_set.target,
        every_n_steps=50,
        metrics=validation_metrics,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=200)

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir="/tmp/iris_model",
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    # Fit model.
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=2000,
                   monitors=[validation_monitor])

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(
        x=test_set.data, y=test_set.target)["accuracy"]
    print("Accuracy: {0:f}".format(accuracy_score))

    # # Classify two new flower samples.
    # new_samples = np.array(
    #     [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
    # y = list(classifier.predict(new_samples))
    # print("Predictions: {}".format(str(y)))


if __name__ == '__main__':
    main()
