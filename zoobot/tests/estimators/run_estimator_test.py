import pytest

import os
import random

import numpy as np
import tensorflow as tf

from zoobot.tfrecord import create_tfrecord
from zoobot.estimators.estimator_params import default_four_layer_architecture, default_params
from zoobot.estimators import run_estimator, estimator_funcs, bayesian_estimator_funcs, dummy_image_estimator


@pytest.fixture(scope='module')
def true_image_values():
    return 3.


@pytest.fixture(scope='module')
def false_image_values():
    return -3.


@pytest.fixture(scope='module')
def example_data(size, true_image_values, false_image_values):
    n_true_examples = 100
    n_false_examples = 400

    true_images = [np.ones((size, size, 3), dtype=float) * true_image_values for n in range(n_true_examples)]
    false_images = [np.ones((size, size, 3), dtype=float) * false_image_values for n in range(n_false_examples)]
    true_labels = [1 for n in range(n_true_examples)]
    false_labels = [0 for n in range(n_false_examples)]

    true_data = list(zip(true_images, true_labels))
    false_data = list(zip(false_images, false_labels))
    all_data = true_data + false_data
    random.shuffle(all_data)
    return all_data


@pytest.fixture()
def tfrecord_dir(tmpdir):
    return tmpdir.mkdir('tfrecord_dir').strpath


@pytest.fixture()
def log_dir(tmpdir):
    return tmpdir.mkdir('log_dir').strpath  # also includes saved model


@pytest.fixture()
def tfrecord_train_loc(tfrecord_dir):
    return '{}/train.tfrecords'.format(tfrecord_dir)


@pytest.fixture()
def tfrecord_test_loc(tfrecord_dir):
    return '{}/test.tfrecords'.format(tfrecord_dir)


# TODO investigate how to share fixtures across test files?
@pytest.fixture()
def example_tfrecords(tfrecord_train_loc, tfrecord_test_loc, example_data):
    tfrecord_locs = [
        tfrecord_train_loc,
        tfrecord_test_loc
    ]
    for tfrecord_loc in tfrecord_locs:
        if os.path.exists(tfrecord_loc):
            os.remove(tfrecord_loc)
        writer = tf.python_io.TFRecordWriter(tfrecord_loc)

        for example in example_data:
            writer.write(create_tfrecord.serialize_image_example(matrix=example[0], label=example[1]))
        writer.close()


@pytest.fixture(params=[
    # 'dummy',
    # 'basic', 
    'bayesian_binary', 
    'bayesian_regression'
    ])
def model(request, size):
    # if request.param == 'dummy':
    #     return dummy_image_estimator.dummy_model_fn
    # if request.param == 'basic':
    #     return estimator_funcs.four_layer_binary_classifier  # TODO move elsewhere?
    if request.param == 'bayesian_binary':
        return bayesian_estimator_funcs.BayesianModel(image_dim=size, regression=False)
    if request.param == 'bayesian_regression':
        return bayesian_estimator_funcs.BayesianModel(image_dim=size, regression=True)


@pytest.fixture()
def run_config(size, log_dir):
    return run_estimator.RunEstimatorConfig(
        initial_size=size,
        final_size=size - 10,
        channels=3,
        label_col='',  # not sure about this
        epochs=2,
        min_epochs=2,
        log_dir=log_dir,
        warm_start=True
    )


@pytest.fixture()
def n_examples():
    return 128  # possible error restoring model if this is not exactly one batch?


@pytest.fixture()
def features(n_examples):
    # {'feature_name':array_of_values} format expected
    feature_shape = [n_examples, 28, 28, 1]
    return {
        'x': tf.constant(
            np.random.rand(*feature_shape), 
            shape=feature_shape, 
            dtype=tf.float32)
        }


@pytest.fixture(params=[
    'binary', 
    'continuous'
    ])
def labels(request, n_examples):
    if request.param == 'binary':
        return tf.constant(
            np.random.randint(low=0, high=2, size=n_examples),
            shape=[n_examples],
            dtype=tf.int32)
    if request.param == 'continuous':
        return tf.constant(
            np.random.uniform(size=n_examples),
            shape=[n_examples],
            dtype=tf.float32)


def test_run_experiment(
    run_config, 
    model, 
    features, 
    labels, 
    n_examples, 
    train_input_fn, 
    monkeypatch):

    # TODO need to test estimator with input functions!
    # mock both input functions
    def dummy_input(input_config=None):
        return train_input_fn(
            features=features,
            labels=labels,
            batch_size=n_examples)
    monkeypatch.setattr(run_estimator, 'train_input', dummy_input)
    monkeypatch.setattr(run_estimator, 'eval_input', dummy_input)
    monkeypatch.setattr(run_config, 'is_ready_to_train', lambda: True)

    def dummy_save(estimator, params, epoch_n, serving_input_receiver_fn):
        pass
    monkeypatch.setattr(run_estimator, 'save_model', dummy_save)  # does not test saving the model

    # no need to add input configs to run_config, they've been monkeypatched
    run_config.model = model

    run_estimator.run_estimator(run_config)
    assert os.path.exists(run_config.log_dir)
