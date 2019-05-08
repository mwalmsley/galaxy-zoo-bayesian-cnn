import pytest

import os

from zoobot.active_learning import simulation_timeline
from zoobot.tests import TEST_FIGURE_DIR


def test_simulated_models_over_time(states, full_catalog):
    sim_models = simulation_timeline.simulated_models_over_time(states, full_catalog)
    assert len(sim_models) == len(states)
    for iteration_n, model in enumerate(sim_models):
        assert model.model.name == 'iteration_{}'.format(iteration_n)


@pytest.fixture()
def save_dir(tmpdir):
    dir_loc = os.path.join(TEST_FIGURE_DIR, 'metrics', 'timeline_metrics')
    if not os.path.isdir(dir_loc):
        os.mkdir(dir_loc)
    return dir_loc

def test_show_model_attr_hist_by_iteration(timeline, save_dir):
    simulation_timeline.show_model_attr_hist_by_iteration(timeline, 'ra', timeline.n_acquired, save_dir)
