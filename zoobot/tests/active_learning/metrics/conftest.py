import pytest

import os

import numpy as np
import pandas as pd

from zoobot.active_learning import metrics, simulated_metrics, simulation_timeline
from zoobot.tests import TEST_FIGURE_DIR


@pytest.fixture()
def save_dir():
    save_dir = os.path.join(TEST_FIGURE_DIR, 'metrics')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    return save_dir


@pytest.fixture()
def state(samples, acquisitions, id_strs):
    return metrics.IterationState(samples, acquisitions, id_strs)


@pytest.fixture()
def iteration_dir(tmpdir):
    return tmpdir.mkdir('iteration_dir').strpath


@pytest.fixture()
def model(state, request):
    return metrics.Model(state, name='example')


@pytest.fixture()
def full_catalog(id_strs):
    # tfrecord from conftest has id_str {'0', ..., '127'}
    data = [
        {
            'subject_id': id_str, 
            'smooth-or-featured_smooth_fraction': np.random.rand(),
            'ra': np.random.rand(),
            'dec': np.random.rand()} for id_str in id_strs]
    return pd.DataFrame(data)


@pytest.fixture()
def sim_model(model, full_catalog):
    return simulated_metrics.SimulatedModel(model, full_catalog)


@pytest.fixture()
def states(state):
    return [state for n in range(5)]


@pytest.fixture()
def n_acquired():
    return 24

@pytest.fixture()
def timeline(states, full_catalog, n_acquired):
    return simulation_timeline.Timeline(states, full_catalog, n_acquired, save_dir)

