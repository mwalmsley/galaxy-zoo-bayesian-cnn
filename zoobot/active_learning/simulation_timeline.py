import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from zoobot.tfrecord import read_tfrecord
from zoobot.active_learning import metrics, simulated_metrics

ATTR_STRS = ['labels', 'ra', 'dec', 'petroth50', 'petrotheta', 'redshift', 'z', 'absolute_size', 'mag_g', 'mag_r', 'petroflux']

class Timeline():
    """
    Create and compare SimulatedModel over many iterations
    """
    def __init__(self, states, catalog, n_acquired, save_dir):
        self._models = simulated_models_over_time(states, catalog)
        self.n_acquired = n_acquired
        self.save_dir = save_dir

    # TODO use collections.abc to generalise this automatically?

    def __getitem__(self, key):
        return self._models[key]

    def __len__(self):
        return len(self._models)


    def save_model_histograms(self):
        for attr_str in ATTR_STRS:
            try:
                show_model_attr_hist_by_iteration(self, attr_str, self.n_acquired, self.save_dir)
            except KeyError:
                logging.warning('Key not found: {}'.format(attr_str))

    
    def save_acquistion_comparison(self):
        for attr_str in ATTR_STRS:
            for iteration_n, model in enumerate(self._models):
                save_dir = os.path.join(self.save_dir, 'iteration_{}'.format(iteration_n))
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                try:
                    try:
                        attr_values = getattr(model, attr_str)  # first, look in model attrs
                    except AttributeError:
                        attr_values = model.catalog[attr_str]  # else, look in catalog columns
                    metrics.acquisitions_vs_values(
                        model.model.acquisitions, 
                        attr_values, 
                        n_acquired=self.n_acquired, 
                        xlabel=attr_str, 
                        save_dir=save_dir)
                except KeyError:
                    logging.warning('{} not found'.format(attr_str))


def simulated_models_over_time(states, catalog):
    sim_models = []
    for iteration_n, state in enumerate(states):
        model = metrics.Model(state, name='iteration_{}'.format(iteration_n))
        simulated_model = simulated_metrics.SimulatedModel(model, catalog)
        sim_models.append(simulated_model)
    return sim_models


def read_id_strs_from_tfrecord(tfrecord_loc, max_subjects=1024):
    # useful for verifying the right subjects are in fact saved
    feature_spec = read_tfrecord.id_feature_spec()
    subjects = read_tfrecord.load_examples_from_tfrecord(tfrecord_loc, feature_spec, max_examples=max_subjects)
    id_strs = [subject['id_str'].decode('utf-8') for subject in subjects]
    assert len(set(id_strs)) == len(id_strs)
    return id_strs


def identify_catalog_subjects_history(tfrecord_locs, catalog):
    assert isinstance(tfrecord_locs, list)
    return [simulated_metrics.match_id_strs_to_catalog(read_id_strs_from_tfrecord(tfrecord_loc), catalog) for tfrecord_loc in tfrecord_locs]


def show_model_attr_hist_by_iteration(models, attr_str, n_acquired, save_dir):
    fig, axes = plt.subplots(nrows=len(models), sharex=True)
    for iteration_n, model in enumerate(models):
        try:
            attr_values = getattr(model, attr_str)  # first, look in model attrs
        except AttributeError:
            attr_values = model.catalog[attr_str]  # else, look in catalog columns
        axes[iteration_n].hist(attr_values[:n_acquired], density=True)
    axes[-1].set_xlabel(attr_str)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, attr_str + '_over_time.png'))
