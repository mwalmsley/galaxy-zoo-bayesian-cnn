import os
import logging
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
import pandas as pd

from zoobot.estimators import make_predictions, bayesian_estimator_funcs
from zoobot.tfrecord import read_tfrecord
from zoobot.uncertainty import discrete_coverage
from zoobot.estimators import input_utils
from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.active_learning import metrics, acquisition_utils, simulated_metrics


def compare_model_errors(model_a, model_b, save_dir):
    # save distribution of various error, compared against baseline that predicts the mean
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))

    ax0.hist(model_a.abs_error, label=model_a.name, density=True, alpha=0.5)
    ax0.hist(model_b.abs_error, label=model_b.name, density=True, alpha=0.5)
    ax0.set_xlabel('Absolute Error')

    ax1.hist(model_a.square_error, label=model_a.name, density=True, alpha=0.5)
    ax1.hist(model_b.square_error, label=model_b.name, density=True, alpha=0.5)
    ax1.set_xlabel('Square Error')

    ax2.hist(model_a.bin_loss_per_subject, label=model_a.name, density=True, alpha=0.5)
    ax2.hist(model_b.bin_loss_per_subject, label=model_b.name, density=True, alpha=0.5)
    ax2.set_xlabel('Binomial Error')

    fig.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics_vs_baseline.png'))
    plt.close()


def compare_models(model_a, model_b):
    logging.info('{} mean square error: {}'.format(model_a.name, model_a.mean_square_error))
    logging.info('{} mean square error: {}'.format(model_b.name, model_b.mean_square_error))
    logging.info('{} mean absolute error: {}'.format(model_a.name, model_a.mean_abs_error))
    logging.info('{} mean absolute error: {}'.format(model_b.name, model_b.mean_abs_error))
    logging.info('{} binomial loss: {}'.format(model_a.name, model_a.mean_bin_loss))
    logging.info('{} mean binomial loss: {}'.format(model_b.name, model_b.mean_bin_loss))


def calculate_predictions(tfrecord_loc, n_galaxies, results_dir, model_name, inital_size=256, n_samples=30):
    images_g, _, id_str_g = input_utils.predict_input_func(tfrecord_loc, n_galaxies=n_galaxies, initial_size=inital_size, mode='id_str')  # tf graph
    with tf.Session() as sess:
        images, id_strs = sess.run([images_g, id_str_g])
    predictor_loc = os.path.join(results_dir, model_name)
    model = make_predictions.load_predictor(predictor_loc)
    results = make_predictions.get_samples_of_images(model, images, n_samples=n_samples)
    return images, id_strs, results


def save_metrics(subjects, labels, state, save_dir, name, mse_comparison=False):
    """Describe the performance of prediction results with paper-quality figures.
    
    Args:
        subjects (np.array): galaxies on which predictions were made, shape (batch, x, y, channel)
        labels (np.array): true labels for galaxies on which predictions were made
                results (np.array): predictions of shape (galaxy_n, sample_n)
        save_dir (str): directory into which to save figures of metrics
    """
    sns.set(context='paper', font_scale=1.5)
    save_sample_distributions(state.samples, labels, save_dir)

    model = metrics.Model(state, name=name)
    model.show_mutual_info_vs_predictions(save_dir)
    acquisition_utils.save_acquisition_examples(
        subjects, model.mutual_info, 'mutual_info', save_dir
    )

    # add in catalog details for more metrics
    catalog = pd.DataFrame('data/panoptes_predictions_selected.csv')
    sim_model = simulated_metrics.SimulatedModel(model, catalog)

    sim_model.show_coverage(save_dir)
    sim_model.compare_binomial_and_abs_error(save_dir)
    sim_model.show_acquisition_vs_label(save_dir)


    # compare_with_baseline(model)
    # if mse_comparison:
    #     compare_with_mse(model)


def save_sample_distributions(samples, labels, save_dir):
    # save histograms of samples, for first 20 galaxies 
    fig, axes = make_predictions.view_samples(samples[:20], labels[:20])
    fig.tight_layout()
    axes[-1].set_xlabel(r'Volunteer Vote Fraction $\frac{k}{N}$')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'sample_dist.png'))
    plt.close(fig)


def compare_with_baseline(model):
    baseline_results = np.ones_like(model.predictions) * labels.mean()  # sample always predicts the mean label
    baseline_model = metrics.Model(baseline_results, labels, name='baseline')
    compare_models(model, baseline_model)


def compare_with_mse(model):
    # warning: fixed to disk location of this reference model
    mse_results = np.load('analysis/uncertainty/al-binomial/five_conv_mse/samples.npy')  # baseline is the same model with deterministic labels and MSE loss
    mse_model = metrics.Model(mse_results, labels, name='mean_loss')
    compare_models(model, mse_model)
    compare_model_errors(model, mse_model, save_dir)


# if __name__ == '__main__':
#     """
#     tfrecord_loc=data/basic_split/panoptes_featured_s128_lfloat_test.tfrecord
#     dvc run -d zoobot/active_learning/check_uncertainty.py -d $tfrecord_loc -o analysis/uncertainty/al-binomial/five_conv_mse -f mse_metrics.dvc  python zoobot/active_learning/check_uncertainty.py --tfrecord_loc=$tfrecord_loc --model_name=five_conv_mse --new_predictions=True
#     latest_model=five_conv_fractions
#     dvc run -d zoobot/active_learning/check_uncertainty.py -d $tfrecord_loc -d analysis/uncertainty/al-binomial/five_conv_mse -o analysis/uncertainty/al-binomial/$latest_model -f latest_metrics.dvc  python zoobot/active_learning/check_uncertainty.py --tfrecord_loc=$tfrecord_loc --model_name=$latest_model
#     """
#     parser = argparse.ArgumentParser(description='Update Model Metrics for Basic Split')
#     parser.add_argument(
#         '--tfrecord_loc',
#         dest='tfrecord_loc',
#         type=str,
#         help='Basic split test tfrecord')
#     parser.add_argument(
#         '--model_name',
#         dest='model_name',
#         type=str,
#         help='Model to make predictions with, under results/[model_name]',
#         default='five_conv_fractions')
#     parser.add_argument(
#         '--mse_comparison',
#         dest='mse_comparison',
#         type=bool,
#         help='Compare with MSE model?',
#         default=False)
#     parser.add_argument(
#         '--new_predictions',
#         dest='new_predictions',
#         type=bool,
#         help='Make new predictions?',
#         default=False)
#     parser.add_argument(
#         '--n_galaxies',
#         dest='n_galaxies',
#         type=int,
#         help='Make new predictions on n_galaxies',
#         default=1024)
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO)

#     results_dir = 'results'

#     save_dir = 'analysis/uncertainty/al-binomial/{}'.format(args.model_name)
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)

#     subjects_loc = os.path.join(save_dir, 'subjects.npy')
#     labels_loc = os.path.join(save_dir, 'labels.npy')



#     if args.new_predictions:
#         subjects, labels, samples = calculate_predictions(args.tfrecord_loc, args.n_galaxies, )
#         np.save(subjects_loc, subjects)
#         np.save(labels_loc, labels)
#         metrics.save_iteration_state(save_dir, subjects, samples, acquisitions=None)

#     else:
#         assert all(os.path.exists(loc) for loc in [subjects_loc, labels_loc])
#         subjects = np.load(subjects_loc)
#         labels = np.load(labels_loc)
#         state = metrics.load_iteration_state(save_dir)

#     save_metrics(subjects, labels, state, save_dir, args.model_name, mse_comparison=args.mse_comparison)
