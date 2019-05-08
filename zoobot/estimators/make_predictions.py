import logging
import functools

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_predictor(predictor_loc):
    """Load a saved model as a callable mapping parsed subjects to class scores
    PredictorFromSavedModel expects feeddict of form {examples: batch of data}
    Batch of data should match predict input of estimator, e.g. (?, size, size, channels) shape
    Where '?' is the (flexible) batch dimension

    Args:
        predictor_loc (str): location of predictor (i.e. saved model)

    Returns:
        function: callable expecting parsed subjects according to saved model input configuration
    """
    model_unwrapped = predictor.from_saved_model(predictor_loc)
    # wrap to avoid having to pass around dicts all the time
    # expects image matrix, passes to model within dict of type {examples: matrix}
    # model returns several columns, select 'predictions_for_true' and flip
    return lambda x: model_unwrapped({'examples': x})['prediction']


def get_samples_of_images(model, images, n_samples):
    """Get many model predictions on each subject

    Args:
        model (function): callable mapping images (parsed subject matrices) to class scores
        images (np.ndarray): batch of subject matrices on which to make a prediction
        n_samples (int): number of samples (i.e. model calls) to calculate per image

    Returns:
        np.array: of form (subject_i, sample_j_of_subject_i)
    """
    assert isinstance(images, np.ndarray)
    results = np.zeros((len(images), n_samples))
    # make predictions batch-wise to avoid out-of-memory issues
    min_image = 0
    batch_size = 1000  # fits comfortably on K80
    logging.info('Making predictions on {} images, {} samples'.format(len(images), n_samples))
    while min_image < len(images):
        for sample_n in range(n_samples):
            image_slice = slice(min_image, min_image + batch_size)
            results[image_slice, sample_n] = model(images[image_slice])
        min_image += batch_size
    logging.info('Predictions complete')
    return results


def binomial_likelihood(labels, predictions, total_votes):
    """
    
    In our formalism:
    Labels are v, and labels * total votes are k.
    Predictions are rho.
    Likelihood is minimised (albeit negative) when rho is most likely given k  
    
    Args:
        labels ([type]): [description]
        predictions ([type]): [description]
        total_votes ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    yes_votes = labels * total_votes
    # must be within meaningful limits, or fail loudly
    assert predictions.min() >= 0.
    assert predictions.max() <= 1.
    est_p_yes = predictions  # rename for clarity
    epsilon = 1e-8
    return yes_votes * np.log(est_p_yes + epsilon) + (total_votes - yes_votes) * np.log(1. - est_p_yes + epsilon)


def bin_prob_of_samples(samples, total_votes):
    # designed to operate on (n_subjects, n_samples) standard response
    assert isinstance(samples, np.ndarray)
    assert len(samples) > 1
    assert len(total_votes) == len(samples)
    # nested lists of dim samples, with elements of another list of variable length (p of k)
    binomial_probs_of_subjects = []
    for subject_n in range(samples.shape[0]):
        binomial_probs_of_samples = []
        for sample_n in range(samples.shape[1]):
            rho = samples[subject_n, sample_n]
            n_draws = total_votes[subject_n]
            rho_rounded = np.around(rho, decimals=3)  # round to use caching, some precision loss
            binomial_probs_of_samples.append(binomial_prob_per_k(rho_rounded, n_draws))
        binomial_probs_of_subjects.append(binomial_probs_of_samples)
    # nested lists of shape (n_subject, n_samples, k) where k varies
    return binomial_probs_of_subjects


@functools.lru_cache(maxsize=16384)  # factor of 2, larger than possible 3dp numbers in {0, 1}
def binomial_prob_per_k(rho, n_draws):
    """Calculate p(k|rho, n_draws) over all possible k, for one rho and one n 
    
    Args:
        sampled_rho (float): MLEs of binomial probability. 1st dim=galaxy, 2nd dim=sample.
        n_draws (int): N draws for those MLEs.

    Returns:
        (np.array): p(k|rho, n_draws) for all possible k, starting with k=0 through k=n_draws
    """
    assert isinstance(rho, float)
    assert isinstance(n_draws, int) or isinstance(n_draws, np.int64)
    k = np.arange(0, n_draws + 1)  # include k=n
    bin_probs = np.array(scipy.stats.binom.pmf(k=k, p=rho, n=int(n_draws)))
    assert np.allclose(bin_probs.sum(), 1.)  # must be one k in (0, ..., n_draws)
    return bin_probs


def plot_samples(scores, labels, total_votes, fig, axes, alpha=0.15):
    for galaxy_n, ax in enumerate(axes):
        x = np.arange(0, total_votes[galaxy_n]+1) # inclusive
        if scores.shape[1] > 1:
            c='g'
            probability_record = []
            for score_n, score in enumerate(scores[galaxy_n]):
                if score_n == 0: 
                    name = 'Model Posteriors'
                else:
                    name = None
                probs = binomial_prob_per_k(score, n_draws=total_votes[galaxy_n])
                probability_record.append(probs)
                ax.plot(x, probs, 'k', alpha=alpha, label=name)
            probability_record = np.array(probability_record)
            mean_posterior = probability_record.mean(axis=0)
        else:
            mean_posterior = binomial_prob_per_k(scores[galaxy_n, 0], n_draws=total_votes[galaxy_n])
            c='k'
        ax.plot(x, mean_posterior, c=c, linewidth=2., label='Posterior')
        ax.axvline(labels[galaxy_n], c='r', label='Observed')
        ax.yaxis.set_visible(False)


def view_samples(scores, labels, total_votes, annotate=False, display_width=5):
    """For many subjects, view the distribution of scores and labels for that subject

    Args:\
        scores (np.array): class scores, of shape (n_subjects, n_samples)
        labels (np.array): class labels, of shape (n_subjects)
    """
    assert len(labels) == len(scores) > 1
    fig, axes = plt.subplots(nrows=len(labels), figsize=(len(labels) / display_width, len(labels)), sharex=True)
    plot_samples(scores, labels, total_votes, fig, axes)

    axes[0].legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.1),
        ncol=2, 
        fancybox=True, 
        shadow=False
    )
    fig.tight_layout()

        # hist_data = ax.hist(scores[galaxy_n])

        # lbound = 0
        # ubound = 0.5
        # if scores[galaxy_n].mean() > 0.5:
        #     lbound = 0.5
        #     ubound = 1

        # ax.axvline(labels[galaxy_n], c='k')
        # ax.axvline(labels[galaxy_n], c='r')
        # c='r'
        # if correct[galaxy_n]:
        #     c='g'
        # ax.axvspan(lbound, ubound, alpha=0.1, color=c)

        # if annotate:
        #     ax.text(
        #         0.7, 
        #         0.75 * np.max(hist_data[0]),
        #         'H: {:.2}'.format(entropies[galaxy_n])
        #     )
        #     ax.set_xlim([0, 1])

    return fig, axes
