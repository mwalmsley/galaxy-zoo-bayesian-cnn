import pytest

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.tests import TEST_FIGURE_DIR
from zoobot.estimators import make_predictions

@pytest.fixture
def n_galaxies():
    return 30

@pytest.fixture
def mean_rho_predictions(n_galaxies):
    return np.random.rand(n_galaxies)

@pytest.fixture
def n_draws():
    return 10

@pytest.fixture
def n_samples():
    return 5

@pytest.fixture
def samples(n_galaxies, n_samples):
    return np.random.rand(n_galaxies, n_samples)


@pytest.fixture
def labels(n_galaxies):
    return np.random.randint(low=1, high=6, size=n_galaxies)


@pytest.fixture
def total_votes(n_galaxies):
    return np.random.randint(low=1, high=6, size=n_galaxies)


def test_load_predictor(predictor_model_loc):
    predictor = make_predictions.load_predictor(predictor_model_loc)
    assert callable(predictor)

def test_binomial_likelihood_1D_predictions(labels, mean_rho_predictions, total_votes):
    likelihoods = make_predictions.binomial_likelihood(labels, mean_rho_predictions, total_votes)
    assert likelihoods.shape == mean_rho_predictions.shape

# def test_binomial_likelihood_samples(labels, samples, total_votes):
#     likelihoods = make_predictions.binomial_likelihood(labels, samples, total_votes)
#     assert likelihoods.shape == samples.shape
    
def test_get_samples_of_subjects(predictor, size, channels):
    n_samples = 5
    n_subjects = 26
    images = np.random.rand(n_subjects, size, size, channels)
    samples = make_predictions.get_samples_of_images(predictor, images, n_samples)
    assert samples.shape == (n_subjects, n_samples)
    assert not np.allclose(samples[0, 0], samples [0, 1])
    assert not np.allclose(samples[0], samples[1])  # predictor is non-deterministic


def test_get_samples_of_many_subjects(predictor, size, channels):
    n_samples = 5
    n_subjects = 20000
    images = np.random.rand(n_subjects, size, size, channels)
    samples = make_predictions.get_samples_of_images(predictor, images, n_samples)
    assert predictor.call_count > n_samples
    assert samples.shape == (n_subjects, n_samples)
    assert not np.allclose(samples[0, 0], samples [0, 1])
    assert not np.allclose(samples[0], samples[1])  # predictor is NOT deterministic
    # TODO replace with a non-deterministic predictor


def test_binomial_prob_per_k(n_draws):
    sampled_rho = 0.5
    prob_per_k = make_predictions.binomial_prob_per_k(sampled_rho, n_draws)
    for n in range(int(n_draws/2) - 1):
        assert np.allclose(prob_per_k[n], prob_per_k[-1-n])
    assert prob_per_k[0] < prob_per_k[1] < prob_per_k[2]


def test_bin_prob_of_samples(samples, total_votes):
    probs = make_predictions.bin_prob_of_samples(samples, total_votes)
    # both len(probs) and len(samples) should be num. of galaxies
    assert len(probs) == len(samples)
    for gal_n, gal in enumerate(probs):
        # each gal should have a list of k's for every sample
        assert len(gal) == samples.shape[1]
        for sample in gal:
            # each list of k's should be as long as total_votes for that galaxy, +1 for edge 
            assert len(sample) == total_votes[gal_n] + 1


@pytest.mark.xfail
def test_view_samples():
    raise NotImplementedError
