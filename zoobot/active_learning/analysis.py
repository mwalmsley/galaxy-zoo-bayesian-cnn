import json
import ast
import itertools
import os
import argparse
import pickle

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')
sns.set()
from zoobot.estimators import input_utils
from zoobot.active_learning import metrics, simulation_timeline

from zoobot.tfrecord import read_tfrecord
from zoobot.tests import TEST_FIGURE_DIR


def get_metrics_from_log(log_loc):
    with open(log_loc, 'r') as f:
        content = f.readlines()
    log_entries = filter(lambda x: is_eval_log_entry(x) or is_iteration_split(x), content)

    iteration = 0
    eval_data = []
    for entry in log_entries:
        if is_iteration_split(entry):
            iteration += 1
        else:
            entry_data = parse_eval_log_entry(entry)
            entry_data['iteration'] = iteration
            eval_data.append(entry_data)
        
    metrics = pd.DataFrame(list(eval_data))
    return metrics[metrics['loss'] > 0]


def is_iteration_split(line):
    return 'All epochs completed - finishing gracefully' in line


def parse_eval_log_entry(line):
    """
    Extract step and metrics from eval log entries
    e.g. 'INFO:tensorflow:Saving dict for global step 1710: global_step = 1710, loss = 25.026402'

    Args:
        line (str): log entry evaluating model, with current global step and metric(s) (see above)

    Return:
        (dict): of form {'step': global step, 'some_metric': metric value, etc.}
    """
    # strip everything left of the colon, and interpret the rest as a literal expression
    colon_index = line.find('global step')
    step_str, data_str = line[colon_index + 12:].split(':')
    step = int(step_str)

    # split on commas, and remove commas
    key_value_strings = data_str.split(',')

    # split on equals, remove equals
    data = {}
    for string in key_value_strings:
        key, value = string.strip(',').split('=')
        data[key.strip()] = float(value.strip())
        data['step'] = step
    return data


def is_eval_log_entry(line):
    """Decide if line is a log entry evaluating the model, or not
    e.g. 'INFO:tensorflow:Saving dict for global step 1710: global_step = 1710, loss = 25.026402'
    
    Args:
        line (str): log entry which may be evaluating model (see above), or may not
    
    Returns:
        (bool): True if log entry is evaluating model, otherwise False
    """
    return 'Saving dict for global step' in line


def smooth_loss(metrics_list, frac=0.25):
    """Smooth out the loss of a model to remove stochastic noise and find typical performance.
    
    Args:
        metrics_list (list): of df, where each df is the metrics for an iteration (step = df row)
    
    Returns:
        (list): as metrics_list, but where each df has 'smooth_loss' column added with smoothed loss
    """
    smoothed_list = []
    for df in metrics_list:
        lowess = sm.nonparametric.lowess
        smoothed_metrics = lowess(
            df['loss'],
            df['step'],
            is_sorted=True, 
            frac=frac)  # controls how much smoothing
        df['smoothed_loss'] = smoothed_metrics[:, 1]
        smoothed_list.append(df)
    return smoothed_list


def plot_log_metrics(metrics_list, save_loc, title=None):
    """Display the loss (smoothed and unsmoothed) of a model over many active learning iterations
    
    Args:
        metrics_list (list): of df, where each df is the metrics for an iteration (step = df row)
        save_loc (str): path to save figure
        title (str, optional): Defaults to None. Title for figure (ideally descriptive)
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, sharey=True)
    for df in metrics_list:
        iteration = df['iteration'].iloc[0]
        ax1.plot(
            df['step'],
            df['smoothed_loss'],
            label='Iteration {}'.format(iteration)
        )

        ax2.plot(
            df['step'],
            df['loss'],
            label='Iteration {}'.format(iteration)
        )

    ax2.set_xlabel('Step')
    ax1.set_ylabel('Eval Loss')
    ax2.set_ylabel('Eval Loss')
    ax1.legend()
    ax2.legend()
    if title is not None:
        ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_loc)


def compare_loss_over_time(active_metrics, baseline_metrics, save_loc, title=None):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, sharey=True)
    runs = [active_metrics, baseline_metrics]
    styles = ['r', 'k--']
    names = ['active', 'baseline']
    for run_n in range(len(runs)):
        metrics_list = runs[run_n]
        style = styles[run_n]

        for df in metrics_list:
            ax1.plot(
                df['step'],
                df['smoothed_loss'],
                style         
            )
            

            ax2.plot(
                df['step'],
                df['loss'],
                style
            )

    ax1.legend(names)
    ax2.legend(names)
    ax2.set_xlabel('Step')
    ax1.set_ylabel('Smoothed Eval Loss')
    ax2.set_ylabel('Eval Loss')
    if title is not None:
        ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_loc)


def find_log(directory):
    return os.path.join(directory, list(filter(lambda x: '.log' in x, os.listdir(directory)))[0])  # returns as tuple of (dir, name)


def compare_metrics(all_metrics, save_loc, title=None):
    
    best_rows = []
    for df in all_metrics:
        df = df.reset_index()
        best_loss_idx = df['smoothed_loss'].idxmin()
        best_row = df.iloc[best_loss_idx]
        best_rows.append(best_row)

    metrics = pd.DataFrame(best_rows)


    fig, ax = plt.subplots()
    sns.barplot(
        data=metrics, 
        x='iteration', 
        y='smoothed_loss', 
        hue='name', 
        ax=ax,
        ci=80
    )
    # ax.set_ylim([0.75, 0.95])
    ax.set_ylabel('Final eval Loss')
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_loc)

def split_by_iter(df):
    return [df[df['iteration'] == iteration] for iteration in df['iteration'].unique()]


def get_smooth_metrics_from_log(log_loc, name=None):
        metrics = get_metrics_from_log(log_loc)
        metric_iters = split_by_iter(metrics)
        metric_smooth = smooth_loss(metric_iters)
        if name is not None:
            for df in metric_smooth:
                df['name'] = name  # record baseline vs active, for example
        return metric_smooth


def get_iteration_dirs(run_dir):
    return [os.path.join(run_dir, d) for d in os.listdir(run_dir) if (os.path.isdir(os.path.join(run_dir, d)) and 'iteration_' in d)]


def get_final_train_locs(run_dir):
    iter_dirs = get_iteration_dirs(run_dir)
    latest_iter_dir = sorted(iter_dirs)[-1]  # assuming iteration_n convention
    latest_train_index = os.path.join(latest_iter_dir, 'train_records_index.json')
    return json.load(open(latest_train_index, 'r'))
    


def show_subjects_by_iteration(tfrecord_locs, n_subjects, size, channels, save_loc):
    assert isinstance(tfrecord_locs, list)
    nrows = len(tfrecord_locs)
    ncols = n_subjects
    scale = 2.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * scale, nrows * scale))

    for iteration_n, tfrecord_loc in enumerate(tfrecord_locs):
        subjects = read_tfrecord.load_examples_from_tfrecord(
            [tfrecord_loc], 
            read_tfrecord.custom_feature_spec(['matrix', 'label']),
            n_examples=n_subjects)
        
        for subject_n, subject in enumerate(subjects):
            read_tfrecord.show_example(subject, size, channels, axes[iteration_n][subject_n])

    fig.tight_layout()
    fig.savefig(save_loc)


def verify_tfrecord_matches_catalog(tfrecord_loc, catalog):
    feature_spec = read_tfrecord.id_label_counts_feature_spec()
    subjects = read_tfrecord.load_examples_from_tfrecord(tfrecord_loc, feature_spec)
    id_strs = []
    for subject in subjects:
        id_str = subject['id_str'].decode('utf-8')
        label = subject['label']
        matching_catalog_labels = catalog[catalog['id_str'].astype(str) == id_str]['label'].values
        assert len(matching_catalog_labels) == 1
        assert np.allclose(matching_catalog_labels, np.ones_like(matching_catalog_labels) * label)
        id_strs.append(id_str)
    # check shard is unique, while we're here
    assert len(id_strs) == len(set(id_strs))
    return id_strs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyse active learning')
    parser.add_argument('--active-dir', dest='active_dir', type=str,
                    help='')
    parser.add_argument('--baseline-dir', dest='baseline_dir', type=str, default=None,
                    help='')
    parser.add_argument('--initial', dest='initial', type=int,
                    help='')
    parser.add_argument('--per-iter', dest='per_iter', type=int,
                    help='')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                    help='')
    parser.add_argument('--catalog-loc', dest='catalog_loc', type=str,
                    help='')

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    initial = args.initial
    per_iter = args.per_iter
    title = 'Initial: {}. Per iter: {}. From scratch.'.format(initial, per_iter)
    name = '{}init_{}per'.format(initial, per_iter)

    # will be re-used for subject history of baseline, if provided
    n_subjects = 15
    size = 128
    channels = 3

    catalog = pd.read_csv(args.catalog_loc)
    catalog_cols = ['ra', 'dec', 'smooth-or-featured_smooth_fraction']

    active_train_locs = get_final_train_locs(args.active_dir)
    show_subjects_by_iteration(active_train_locs, n_subjects, size, channels, os.path.join(args.output_dir, 'subject_history_active.png'))
    
    
    # active_history = identify_catalog_subjects_history(active_train_locs, catalog)
    # for col in catalog_cols:
    #     show_catalog_col_by_iteration(
    #         active_history, 
    #         col, 
    #         os.path.join(args.output_dir, '{}_history_active.png'.format(col))
    #     )

    active_iteration_dirs = get_iteration_dirs(args.active_dir)
    active_states = [metrics.load_iteration_state(iteration_dir) for iteration_dir in active_iteration_dirs]
    active_timeline = simulation_timeline.Timeline(active_states, catalog, args.per_iter, args.output_dir)
    active_timeline.save_acquistion_comparison()
    active_timeline.save_model_histograms()
    

    active_log_loc = find_log(args.active_dir)
    active_save_loc = os.path.join(args.output_dir, 'acc_metrics_active_' + name + '.png')

    active_smooth_metrics = get_smooth_metrics_from_log(active_log_loc, name='active')
    plot_log_metrics(active_smooth_metrics, active_save_loc, title=title)

    if args.baseline_dir is not None:
        baseline_train_locs = get_final_train_locs(args.baseline_dir)
        show_subjects_by_iteration(baseline_train_locs, n_subjects, size, channels, os.path.join(args.output_dir, 'subject_history_baseline.png'))
        
        baseline_iteration_dirs = get_iteration_dirs(args.baseline_dir)
        baseline_states = [metrics.load_iteration_state(iteration_dir) for iteration_dir in baseline_iteration_dirs]
        args.mkdir(os.path.join(args.output_dir, 'baseline'))
        baseline_timeline = simulation_timeline.Timeline(baseline_states, catalog, args.per_iter, os.path.join(args.output_dir, 'baseline'))
        baseline_timeline.save_acquistion_comparison()
        baseline_timeline.save_model_histograms()

        baseline_log_loc = find_log(args.baseline_dir)
        baseline_save_loc = os.path.join(args.output_dir, 'acc_metrics_baseline_' + name + '.png')
        baseline_smooth_metrics = get_smooth_metrics_from_log(baseline_log_loc, name='baseline')
        plot_log_metrics(baseline_smooth_metrics, baseline_save_loc, title=title)

        comparison_save_loc = os.path.join(args.output_dir, 'loss_comparison_' + name + '.png')
        compare_loss_over_time(active_smooth_metrics, baseline_smooth_metrics, comparison_save_loc, title=title)

        best_result_save_loc = os.path.join(args.output_dir, 'loss_best_results_' + name + '.png')
        compare_metrics(baseline_smooth_metrics + active_smooth_metrics, best_result_save_loc, title=title)
