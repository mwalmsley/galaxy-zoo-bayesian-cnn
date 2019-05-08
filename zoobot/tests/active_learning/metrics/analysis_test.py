import pytest

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zoobot.tests import TEST_FIGURE_DIR, TEST_EXAMPLE_DIR
from zoobot.active_learning import analysis


@pytest.fixture()
def tfrecord_locs(tfrecord_matrix_ints_loc):
    return [tfrecord_matrix_ints_loc, tfrecord_matrix_ints_loc]


@pytest.fixture()
def tfrecord_index_loc(tfrecord_dir, tfrecord_locs):
    loc = os.path.join(tfrecord_dir, 'tfrecord_index.json')
    with open(loc, 'w') as f:
        json.dump(tfrecord_locs, f)
    return loc


def test_show_subjects_by_iteration(tfrecord_locs, size, channels):
    save_loc = os.path.join(TEST_FIGURE_DIR, 'subjects_in_shards.png')
    analysis.show_subjects_by_iteration(tfrecord_locs, 5, size, channels, save_loc)


# TODO temporary, will need one in TEST_EXAMPLES
@pytest.fixture()
def log_loc():
    return os.path.join(TEST_EXAMPLE_DIR, 'active_learning_example.log')


@pytest.fixture()
def eval_log_entry():
    return '2018-09-10 12:03:37,451 Saving dict for global step 841: acc/accuracy = 0.7994792, acc/mean_per_class_accuracy = 0.79981816, confusion/false_negatives = 35.0, confusion/false_positives = 42.0, confusion/true_negatives = 155.0, confusion/true_positives = 152.0, global_step = 841, loss = 0.40684244, pr/auc = 0.79981816, pr/precision = 0.78350514, pr/recall = 0.8128342, sanity/predictions_above_95% = 0.1015625, sanity/predictions_below_5% = 0.13802083'


@pytest.fixture()
def iteration_split_entry():
    return '2018-09-10 22:22:08,600 All epochs completed - finishing gracefully'


def test_is_iteration_split_entry(iteration_split_entry):
    assert analysis.is_iteration_split(iteration_split_entry)


def test_is_eval_log_entry(eval_log_entry):
    assert analysis.is_eval_log_entry(eval_log_entry)


def test_parse_eval_log_entry(eval_log_entry):
    data = analysis.parse_eval_log_entry(eval_log_entry)
    assert np.isclose(data['acc/accuracy'], 0.7994792)
    assert np.isclose(data['sanity/predictions_below_5%'], 0.13802083)


def test_get_metrics_from_log(log_loc):
    metrics = analysis.get_metrics_from_log(log_loc)
    metrics = metrics[metrics['acc/accuracy'] > 0]
    assert metrics['iteration'].min() == 0
    assert metrics['iteration'].max() > 0
