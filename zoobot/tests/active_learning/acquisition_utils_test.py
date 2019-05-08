import pytest

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.estimators import make_predictions
from zoobot.active_learning import acquisition_utils
from zoobot.tests import TEST_FIGURE_DIR


@pytest.fixture
def save_dir():
    save_dir = os.path.join(TEST_FIGURE_DIR, 'metrics')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    return save_dir

@pytest.fixture
def n_subjects():
    return 50

@pytest.fixture
def n_samples():
    return 20

@pytest.fixture
def samples(n_subjects, n_samples):
    return np.random.rand(n_subjects, n_samples)

@pytest.fixture
def total_votes(n_subjects):
    return [np.random.randint(low=1, high=41) for n in range(n_subjects)]


@pytest.fixture
def bin_probs_of_samples(n_subjects, n_samples, total_votes):  # to rename
    binomial_probs_of_subjects = []
    for subject_n in range(n_subjects):
        binomial_probs_of_samples = []
        for sample_n in range(n_samples):
            k_array = np.random.rand(total_votes[subject_n] + 1)
            k_array = k_array / k_array.sum()  # np.array, not list
            binomial_probs_of_samples.append(k_array)
        binomial_probs_of_subjects.append(binomial_probs_of_samples)
    return binomial_probs_of_subjects


@pytest.fixture
def probabilities():
    raw = np.random.rand(25)
    return raw / raw.sum()  # should sum to 0


def test_distribution_entropy(probabilities):
    dist_entropy = acquisition_utils.distribution_entropy(probabilities)
    assert isinstance(dist_entropy, float)
    assert dist_entropy > 0.


def test_binomial_entropy():
    n_draws = 10
    rho = 0.5
    entropy = acquisition_utils.binomial_entropy(rho, n_draws)
    assert entropy.shape == ()  # should be scalar


def test_binomial_entropy_vectorized():
    n_draws = 10  # not yet tested with varying n
    rho = [0.1, 0.5, 0.9]
    entropy = acquisition_utils.binomial_entropy(rho, n_draws)
    assert entropy.min() > 0
    assert len(entropy) == 3  # should be scalar
    assert entropy[1] == entropy.max()
    assert np.allclose(entropy[0], entropy[-1])


def test_get_mean_prediction(bin_probs_of_samples):
    mean_predictions = acquisition_utils.get_mean_k_predictions(bin_probs_of_samples)
    assert len(mean_predictions) == len(bin_probs_of_samples)
    for subject_n, subject in enumerate(mean_predictions):
        assert not np.allclose(subject, subject.mean())  # should vary by k for each subject
        # should have k mean predictions per subject
        assert len(subject) == len(bin_probs_of_samples[subject_n][0])  # 0th sample, not important


def test_expected_binomial_entropy(bin_probs_of_samples):
    entropy = acquisition_utils.expected_binomial_entropy(bin_probs_of_samples)
    assert entropy.ndim == 1
    assert entropy.min() > 0


def test_predictive_binomial_entropy(bin_probs_of_samples):
    predictive_entropy = acquisition_utils.predictive_binomial_entropy(bin_probs_of_samples)
    assert predictive_entropy.ndim == 1
    assert not np.allclose(predictive_entropy, acquisition_utils.expected_binomial_entropy(bin_probs_of_samples))
    assert not np.allclose(predictive_entropy, predictive_entropy.mean())



def test_predictive_and_expected_entropy_functional(samples):
    bin_probs = make_predictions.bin_prob_of_samples(samples, total_votes=[40 for n in range(len(samples))])
    predictive_entropy = acquisition_utils.predictive_binomial_entropy(bin_probs)
    expected_entropy = acquisition_utils.expected_binomial_entropy(bin_probs)
    mutual_info = predictive_entropy - expected_entropy
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    mean_prediction = samples.mean(axis=1)
    ax0.scatter(mean_prediction, predictive_entropy, label='Predictive')
    ax0.set_xlabel('Mean prediction')
    ax0.set_ylabel('Predictive Entropy')
    ax1.scatter(mean_prediction, expected_entropy, label='Expected')
    ax1.set_xlabel('Mean prediction')
    ax1.set_ylabel('Expected Entropy')
    ax2.scatter(mean_prediction, mutual_info, label='Mutual Info')
    ax2.set_xlabel('Mean prediction')
    ax2.set_ylabel('Mutual Information')
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'mutual_info.png'))

@pytest.mark.xfail
def test_binomial_entropy_plotted():
    raise NotImplementedError


def test_mutual_info_acquisition_func(monkeypatch, samples, total_votes):
    # dummy functional test, don't actually have any expected answers
    # rely on each component to work correctly
    mutual_info = acquisition_utils.mutual_info_acquisition_func(samples, expected_votes=total_votes)
    assert len(mutual_info) == len(samples)


def test_save_acquisition_examples(subjects, save_dir):
    subject_data = np.array([subject['matrix'] for subject in subjects * 5])
    acq_values = np.random.rand(len(subjects))
    acq_string = 'mock_acquisition'
    acquisition_utils.save_acquisition_examples(subject_data, acq_values, acq_string, save_dir)


@pytest.mark.xfail
def test_show_acquisitions_from_tfrecords():
    acquisition_utils.show_acquisitions_from_tfrecords()
