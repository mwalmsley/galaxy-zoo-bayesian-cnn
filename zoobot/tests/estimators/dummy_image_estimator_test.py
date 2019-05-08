import pytest

import tensorflow as tf
import numpy as np

from zoobot.estimators import dummy_image_estimator


# Build estimator from model function
@pytest.fixture()
def estimator(batch_size, size):
    return tf.estimator.Estimator(
        model_fn=dummy_image_estimator.dummy_model_fn,
        params={
            'batch_size': batch_size,  # required to get dimensions correct,
            'image_dim': size}
    )

def test_training(estimator, train_input_fn, random_features, random_labels, batch_size):
    # modifies estimator inplace
    estimator.train(
        input_fn=lambda: train_input_fn(random_features, random_labels, batch_size),
        steps=2
    )


def test_predict(estimator, train_input_fn, eval_input_fn, random_features, random_labels, batch_size):
    test_training(estimator, train_input_fn, random_features, random_labels, batch_size)  # requires model with a training
    predictions = estimator.predict(  # returns a generator
        input_fn=lambda: eval_input_fn(random_features, None, batch_size))


def test_eval(estimator, train_input_fn, random_features, random_labels, batch_size):
    test_training(estimator, train_input_fn, random_features, random_labels, batch_size)
    eval_result = estimator.evaluate(
        input_fn=lambda: train_input_fn(random_features, random_labels, batch_size),
        steps=1)  # or it never ends!
