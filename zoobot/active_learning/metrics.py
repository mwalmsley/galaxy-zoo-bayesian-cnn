import os
import logging
import json
import pickle
from collections import namedtuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import metrics
import tensorflow as tf

from zoobot.estimators import make_predictions, bayesian_estimator_funcs, input_utils


"""Useful for basic recording from iterations. Input to Model, should not be used elsewhere"""
IterationState = namedtuple('IterationState', ['samples', 'acquisitions', 'id_strs'])


def save_iteration_state(iteration_dir, subjects, samples, acquisitions):
    id_strs = [subject['id_str'] for subject in subjects]
    iteration_state = IterationState(samples, acquisitions, id_strs)
    with open(os.path.join(iteration_dir, 'state.pickle'), 'wb') as f:
        pickle.dump(iteration_state, f)


def load_iteration_state(iteration_dir):
    with open(os.path.join(iteration_dir, 'state.pickle'), 'rb') as f:
        return pickle.load(f)



class Model():
    """Get and plot basic model results, with no external info"""

    def __init__(self, state, name, bin_probs=None, ):
        # save samples, id_strs and acquisitions sorted by acq. value (descending), to avoid resorting later
        args_to_sort = np.argsort(state.acquisitions)[::-1]
        self.samples = state.samples[args_to_sort, :]
        self.id_strs = [state.id_strs[n] for n in args_to_sort]  # list, sort with listcomp
        self.acquisitions = [state.acquisitions[n] for n in args_to_sort]
        self.name = name

