import tensorflow as tf

from zoobot.estimators.estimator_funcs import four_layer_binary_classifier
from zoobot.estimators.run_estimator import run_estimator


def default_params():
    """
    Get typical parameters controlling the training/testing of an estimator
    Returns:
        (dict) typical parameters controlling the training/testing of an estimator
    """
    return dict(
        epochs=1000,
        batch_size=128,
        image_dim=64,
        max_train_batches=None,
        log_freq=25,
        log_dir='runs/default_run',
        train_batches=30,
        eval_batches=3
    )


def default_three_layer_architecture():
    """
    Get parameters for three-layer CNN architecture (excluding input dimension)
    Returns:
        (dict) parameters for three-layer CNN architecture (excluding input dimension)
    """
    return dict(
        padding='same',

        conv1_filters=32,
        conv1_kernel=5,

        conv1_activation=tf.nn.relu,

        pool1_size=2,
        pool1_strides=2,

        conv2_filters=64,
        conv2_kernel=5,
        conv2_activation=tf.nn.relu,

        pool2_size=2,
        pool2_strides=2,

        dense1_units=1064,
        dense1_dropout=0.4,
        dense1_activation=tf.nn.relu,

        learning_rate=0.001,
        # optimizer=tf.train.GradientDescentOptimizer,
        optimizer=tf.train.AdamOptimizer
    )


def default_four_layer_architecture():
    """
    Get parameters for four-layer CNN architecture (excluding input dimension)
    Returns:
        (dict) parameters for four-layer CNN architecture (excluding input dimension)
    """
    return dict(
        padding='same',

        conv1_filters=32,
        conv1_kernel=3,

        conv1_activation=tf.nn.relu,

        pool1_size=2,
        pool1_strides=2,

        conv2_filters=32,
        conv2_kernel=3,
        conv2_activation=tf.nn.relu,

        pool2_size=2,
        pool2_strides=2,

        conv3_filters=64,
        conv3_kernel=3,
        conv3_activation=tf.nn.relu,

        pool3_size=2,
        pool3_strides=2,

        dense1_units=1064,
        dense1_dropout=0.5,
        dense1_activation=tf.nn.relu,

        learning_rate=0.001,
        optimizer=tf.train.AdamOptimizer
    )


if __name__ == '__main__':
    params = default_params()
    params.update(default_four_layer_architecture())
    params['image_dim'] = 128
    params['log_dir'] = 'runs/chollet_128_triple'
    run_estimator(four_layer_binary_classifier, params)
