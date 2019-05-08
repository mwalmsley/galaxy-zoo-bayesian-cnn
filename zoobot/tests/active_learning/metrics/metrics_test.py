import pytest

import os

import numpy as np

from zoobot.active_learning import metrics
from zoobot.tests import TEST_FIGURE_DIR

def test_save_iteration_state(iteration_dir, subjects, samples, acquisitions):
    metrics.save_iteration_state(iteration_dir, subjects, samples, acquisitions)
    assert os.path.isfile(os.path.join(iteration_dir, 'state.pickle'))


def test_load_iteration_state(subjects, samples, acquisitions, iteration_dir):
    # assumes saving works correctly 
    metrics.save_iteration_state(iteration_dir, subjects, samples, acquisitions)
    state = metrics.load_iteration_state(iteration_dir)
    assert isinstance(state, metrics.IterationState)
    # TODO should probably test with consistent id str etc

def test_model_init(state):
    example_metrics = metrics.Model(state, name='example')
    assert example_metrics.name == 'example'
     # should have been sorted by acq. value
    assert np.allclose(example_metrics.acquisitions, np.sort(example_metrics.acquisitions)[::-1])
    assert state.id_strs != example_metrics.id_strs 
    # TODO proper checks that everything is sorted by acquisition

def test_show_mutual_info_vs_predictions(model, save_dir):
    model.show_mutual_info_vs_predictions(save_dir)
    assert os.path.exists(os.path.join(save_dir, 'entropy_by_prediction.png'))

def test_acquisitions_vs_mean_prediction(model, save_dir):
    n_acquired = 20
    if model.acquisitions is None:
        with pytest.raises(ValueError):
            model.acquisitions_vs_mean_prediction(n_acquired, save_dir)
    else:
        model.acquisitions_vs_mean_prediction(n_acquired, save_dir)
        assert os.path.exists(os.path.join(save_dir, 'discrete_coverage.png'))
