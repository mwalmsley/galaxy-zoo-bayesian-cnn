import pytest

import os

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from zoobot.estimators import bayesian_estimator_funcs, make_predictions
from zoobot.tests import TEST_FIGURE_DIR


# by default
# @pytest.fixture()
# def n_votes():
#     return 40

@pytest.fixture()
def single_label():
    return tf.constant(0.3, dtype=tf.float32)  # 30% vote fraction


@pytest.fixture()
def single_prediction():
    return tf.constant(0.5, dtype=tf.float32)


def test_binomial_loss_1D(single_label, single_prediction):

    neg_log_likelihood = bayesian_estimator_funcs.binomial_loss(single_label, single_prediction)

    with tf.Session() as sess:
        neg_log_likelihood = sess.run([neg_log_likelihood])


def test_binomial_loss_1D_plot():
        # verify that np and tf versions of binomial loss look good and agree

    labels = tf.placeholder(tf.float32, shape=())
    predictions = tf.placeholder(tf.float32, shape=())
    tf_neg_log_likelihood = bayesian_estimator_funcs.binomial_loss(labels, predictions)



    x_range = np.linspace(0, 1., num=100)
    y = 0.3

    tf_neg_likilihoods = []
    for x in x_range:
        with tf.Session() as sess:
            result = sess.run(
                [tf_neg_log_likelihood],
                feed_dict={
                    labels: y,
                    predictions: x}
                    )
            tf_neg_likilihoods.append(result)

    np_neg_likilihoods = - make_predictions.binomial_likelihood(y, x_range, total_votes=40)
    
    plt.plot(x_range, np_neg_likilihoods, label='np neg log likelihood')
    plt.plot(x_range, tf_neg_likilihoods, label='tf neg log likelihood')
    plt.axvline(y, linestyle='--', label='True vote fraction')
    plt.xlabel('Model prediction')
    plt.ylabel('Negative log likelihood')
    plt.ylim([0., 100.])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_FIGURE_DIR, 'binomial_loss.png'))
    