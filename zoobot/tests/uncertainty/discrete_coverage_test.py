import pytest

import os

import scipy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.estimators import make_predictions
from zoobot.uncertainty import discrete_coverage
from zoobot.tests import TEST_FIGURE_DIR


@pytest.fixture()
def n_subjects():
    return 24

@pytest.fixture()
def n_samples():
    return 10

@pytest.fixture()
def n_draws():
    return 20

@pytest.fixture()
def true_p(n_subjects):
    return np.random.rand(n_subjects)  # per subject

@pytest.fixture()
def volunteer_votes(true_p, n_draws):
    return np.concatenate([scipy.stats.binom.rvs(p=true_p[subject_n], n=n_draws, size=1) for subject_n in range(len(true_p))])


@pytest.fixture()
def bin_prob_of_samples_by_k(n_subjects, n_samples, n_draws, true_p):
    """ of form [subject_n, sample_n, k] """
    bin_probs = np.zeros((n_subjects, n_samples, n_draws + 1))
    for subject_n in range(n_subjects):
        for sample_n in range(n_samples):
            for k in range(n_draws + 1):
                bin_probs[subject_n, sample_n, k] = scipy.stats.binom(p=true_p[subject_n], n=n_draws).pmf(k)
            assert np.allclose(bin_probs[subject_n, sample_n].sum(), 1.)  # must be one of these k
    return bin_probs


@pytest.fixture()
def coverage_df():
    return pd.DataFrame([
        {
            'max_state_error': 4,
            'prediction': 0.7,
            'observed': 1.
        },
        {
            'max_state_error': 4,
            'prediction': 0.7,
            'observed': 0.
        },
        {
            'max_state_error': 12,
            'prediction': 0.5,
            'observed': 0.
        },
        {
            'max_state_error': 12,
            'prediction': 0.,
            'observed': 0.
        },
        {
            'max_state_error': 12,
            'prediction': 1.,
            'observed': 0.
        },
    ])


@pytest.fixture()
def coverage_df_large(n_subjects, n_samples, n_draws):
    # only 1 sample
    rands = np.random.rand(n_subjects, n_samples, n_draws + 1)
    data = [{'max_state_error': n_k, 'prediction': rands[n_subj, n_samp, n_k] * 0.9, 'observed': np.around(rands[n_subj, n_samp, n_k])}
            for n_k in range(n_draws + 1)
            for n_samp in range(n_samples)
            for n_subj in range(n_subjects)
            ]
    return pd.DataFrame(data)

@pytest.fixture()
def reduced_df():
    return pd.DataFrame([
        {
            'max_state_error': 4,
            'prediction': 0.7,
            'observed': 0.6
        },
        {
            'max_state_error': 8,
            'prediction':0.8,
            'observed': 0.8
        },
        {
            'max_state_error': 10,
            'prediction':0.85,
            'observed': 0.85
        },
        {
            'max_state_error': 12,
            'prediction': 0.9,
            'observed': 0.95
        }
    ])


def test_plot_coverage_df(coverage_df):
    fig, ax = plt.subplots()
    save_loc = os.path.join(TEST_FIGURE_DIR, 'discrete_coverage_plot.png')
    discrete_coverage.plot_coverage_df(coverage_df, ax)
    fig.tight_layout()
    fig.savefig(save_loc)


def test_evaluate_discrete_coverage(volunteer_votes, bin_prob_of_samples_by_k):
    # if I'm clever, I can get error bars
    # df of form: [max +/- n states, mean observed frequency, mean probability prediction]
    coverage_df = discrete_coverage.evaluate_discrete_coverage(volunteer_votes, bin_prob_of_samples_by_k)
    # assert np.allclose(len(coverage_df), np.product(bin_prob_of_samples_by_k) * 10 * 2)  # 10 test errors, observed Y/N
    save_loc = os.path.join(TEST_FIGURE_DIR, 'discrete_coverage_evaluate.png')
    fig, ax = plt.subplots()
    discrete_coverage.plot_coverage_df(coverage_df, ax)
    fig.tight_layout()
    fig.savefig(save_loc)

def test_evaluate_discrete_coverage_bad_fractions(bin_prob_of_samples_by_k):  # TODO
    """Should raise an error if mistakenly called with vote fractions instead of labels"""
    vote_fracs = np.random.rand(len(bin_prob_of_samples_by_k))
    with pytest.raises(ValueError):
        discrete_coverage.evaluate_discrete_coverage(vote_fracs, bin_prob_of_samples_by_k)


def test_reduce_coverage_df(coverage_df):
    reduced_df = discrete_coverage.reduce_coverage_df(coverage_df)
    assert len(reduced_df) == 2
    first_row = reduced_df.iloc[0]
    assert first_row['max_state_error'] == 4
    assert np.allclose(first_row['prediction'], 0.7)
    assert np.allclose(first_row['observed'], .5)
    second_row = reduced_df.iloc[1]
    assert second_row['max_state_error'] == 12
    assert np.allclose(second_row['prediction'], 0.5)
    assert np.allclose(second_row['observed'], .0)


def test_calibrate_predictions(coverage_df_large):
    calibrated_df = discrete_coverage.calibrate_predictions(coverage_df_large)
    assert len(calibrated_df) < len(coverage_df_large)  # only calibration test set['Prediction', 'Observed']
    save_loc = os.path.join(TEST_FIGURE_DIR, 'discrete_coverage_calibrated.png')
    fig, ax = plt.subplots()
    discrete_coverage.plot_coverage_df(calibrated_df, ax=ax)
    fig.tight_layout()
    fig.savefig(save_loc)
    # should have increased predictions to compensate for offset
    assert calibrated_df['prediction_calibrated'].mean() > calibrated_df['prediction'].mean()
