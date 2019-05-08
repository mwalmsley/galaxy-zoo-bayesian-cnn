import pytest

import numpy as np

from zoobot.uncertainty import sample_statistics


@pytest.fixture()
def samples_mean():
    return 0.5


@pytest.fixture()
def samples_scale():
    return 0.1


@pytest.fixture(params=[5, 50, 1000])
def n_samples(request):
    return request.param


@pytest.fixture()
def samples(samples_mean, samples_scale):
    return np.random.normal(loc=samples_mean, scale=samples_scale, size=(50, 50)).mean(axis=1)


def test_samples_to_posterior(samples, samples_mean, samples_scale):
    posterior = sample_statistics.samples_to_posterior(samples)
    eval_points = np.linspace(0, 1)
    posterior_eval = posterior(eval_points)
    most_likely_eval_point = eval_points[np.argmax(posterior_eval)]
    assert (samples_mean - samples_scale) < most_likely_eval_point < (samples_mean + samples_scale)


def test_samples_to_interval(samples, samples_mean, samples_scale):
    interval = sample_statistics.samples_to_interval(samples, alpha=0.05)
    assert interval[0] < interval[1]
    assert interval[0] < samples_mean - samples_scale * 0.1
    assert interval[1] > samples_mean + samples_scale * 0.1
