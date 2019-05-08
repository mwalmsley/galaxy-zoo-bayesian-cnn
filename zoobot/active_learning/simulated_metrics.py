import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from zoobot.estimators import make_predictions
from zoobot.uncertainty import discrete_coverage
from zoobot.tfrecord import read_tfrecord
from zoobot.uncertainty import discrete_coverage
from zoobot.active_learning import acquisition_utils
from zoobot.tfrecord import catalog_to_tfrecord


class SimulatedModel():
    """
    Calculate and visualise additional metrics (vs. Model) using a provided catalog
    Useful to create more info from a Model, or for internal use within Timeline
    """

    def __init__(self, model, full_catalog, bin_probs=None):
        self.model = model
        self.catalog = match_id_strs_to_catalog(model.id_strs, full_catalog)

        self.labels = self.catalog['label'].values
        self.total_votes = self.catalog['total_votes'].values
        assert not any(np.isnan(self.labels))
        assert not any(np.isnan(self.total_votes))


        # for speed, calculate the (subject_n, sample_n, k) probabilities once here and re-use
        if bin_probs is None:
            self.bin_probs = make_predictions.bin_prob_of_samples(self.model.samples, total_votes=self.total_votes) 
        else:
            self.bin_probs = bin_probs
        
        self.calculate_mutual_info()

        self.mean_rho_prediction = np.mean(self.model.samples, axis=1)
        self.mean_k_prediction = acquisition_utils.get_mean_k_predictions(self.bin_probs)  # per subject per k
        self.expected_k_prediction = np.array([np.sum(np.arange(len(k_predictions)) * k_predictions) for k_predictions in self.mean_k_prediction])  # expected k per subject
    
        self.calculate_default_metrics()

    def calculate_mutual_info(self):
        self.predictive_entropy = acquisition_utils.predictive_binomial_entropy(self.bin_probs)
        self.expected_entropy = acquisition_utils.expected_binomial_entropy(self.bin_probs)
        self.mutual_info = self.predictive_entropy - self.expected_entropy


    def show_mutual_info_vs_predictions(self, save_dir):
        # How does being smooth or featured affect each entropy measuremement?
        fig, (row0, row1) = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12, 6))
        self.mutual_info_vs_mean_prediction(row0)
        self.delta_acquisition_vs_mean_prediction(row1)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'entropy_by_prediction.png'))
        plt.close()


    def mutual_info_vs_mean_prediction(self, row):
        mean_pred_range = np.linspace(0.02, 0.98)
        entropy_of_mean_pred_range = acquisition_utils.binomial_entropy(mean_pred_range, n_draws=self.total_votes)  # WARNING

        row[0].scatter(self.mean_rho_prediction, self.predictive_entropy)
        row[0].plot(mean_pred_range, entropy_of_mean_pred_range)
        row[0].set_xlabel('Mean Prediction')
        row[0].set_ylabel('Predictive Entropy')

        row[1].scatter(self.mean_rho_prediction, self.expected_entropy)
        row[1].plot(mean_pred_range, entropy_of_mean_pred_range)
        row[1].set_ylabel('Expected Entropy')
        row[2].scatter(self.mean_rho_prediction, self.mutual_info)
        row[2].set_ylabel('Mutual Information')


        row[1].set_xlabel('Mean Prediction')
        row[2].set_xlabel('Mean Prediction')


    def delta_acquisition_vs_mean_prediction(self, row):
        entropy_of_mean_prediction = acquisition_utils.binomial_entropy(self.mean_rho_prediction, n_draws=40)  # WARNING will need to be changed for bars N = 10
        row[0].scatter(self.mean_rho_prediction, self.predictive_entropy - entropy_of_mean_prediction)
        row[0].set_ylabel('Delta Predictive Entropy')
        row[1].scatter(self.mean_rho_prediction, self.expected_entropy - entropy_of_mean_prediction)
        row[1].set_ylabel('Delta Expected Entropy')


    def acquisitions_vs_mean_prediction(self, n_acquired, save_dir):
        acquisitions_vs_values(self.model.acquisitions, self.mean_rho_prediction, n_acquired, 'Mean Prediction', save_dir)



    def calculate_default_metrics(self):
        """
        Calculate common metrics for performance of sampled predictions vs. volunteer vote fractions
        Store these in object state
        
        Args:
            results (np.array): predictions of shape (galaxy_n, sample_n)
            labels (np.array): true labels for galaxies on which predictions were made
        
        Returns:
            None
        """
        assert self.labels is not None
        
        self.abs_rho_error = np.abs(self.mean_rho_prediction - (self.labels / self.total_votes))
        self.square_rho_error = ((self.labels / self.total_votes) - self.mean_rho_prediction) ** 2.
        self.mean_abs_rho_error = np.mean(self.abs_rho_error)
        self.mean_square_rho_error = np.mean(self.square_rho_error)

        # self.abs_k_error = np.abs(self.most_likely_k_prediction - self.labels)
        # self.square_k_error = (self.labels - self.most_likely_k_prediction) ** 2.
        # self.mean_abs_k_error = np.mean(self.abs_k_error)
        # self.mean_square_k_error = np.mean(self.square_k_error)


        # self.bin_likelihood = make_predictions.binomial_likelihood(self.labels, self.mean_rho_prediction, total_votes=self.total_votes)
        # self.bin_loss_per_sample = - self.bin_likelihood  # we want to minimise the loss to maximise the likelihood
        # self.bin_loss_per_subject = np.mean(self.bin_loss_per_sample, axis=1)
        # self.mean_bin_loss = np.mean(self.bin_loss_per_subject)  # scalar, mean likelihood


    def compare_binomial_and_abs_error(self, save_dir):
        # Binomial loss should increase with absolute error, but not linearly
        plt.figure()
        g = sns.jointplot(self.abs_error, self.bin_loss_per_subject, kind='reg')
        plt.xlabel('Abs. Error')
        plt.ylabel('Binomial Loss')
        plt.xlim([0., 0.5])
        plt.tight_layout()
        g.savefig(os.path.join(save_dir, 'bin_loss_vs_abs_error.png'))
        plt.close()


    def show_acquisition_vs_label(self, save_dir):
        fig, row = plt.subplots(ncols=3, sharex=True, figsize=(12, 4))
        _ = self.acquisition_vs_volunteer_votes(row)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'entropy_by_label.png'))
        plt.close()


    def acquisition_vs_volunteer_votes(self, row):
        ax00, ax01, ax02 = row
        ax00.scatter(self.labels, self.model.predictive_entropy)
        ax00.set_ylabel('Predictive Entropy')
        ax01.scatter(self.labels, self.model.expected_entropy)
        ax01.set_ylabel('Expected Entropy')
        ax02.scatter(self.labels, self.model.mutual_info)
        ax02.set_ylabel('Mutual Information')

        ax00.set_xlabel('Vote Fraction')
        ax01.set_xlabel('Vote Fraction')
        ax02.set_xlabel('Vote Fraction')

        return ax00, ax01, ax02


    def show_coverage(self, save_dir):
        if self.labels is None:
            raise ValueError('Calculating coverage requires volunteer votes to be known')
        fig, ax = plt.subplots()
        coverage_df = discrete_coverage.evaluate_discrete_coverage(self.votes, self.mean_k_prediction)
        discrete_coverage.plot_coverage_df(coverage_df, ax=ax)
        fig.tight_layout()
        save_loc = os.path.join(save_dir, 'discrete_coverage.png')
        fig.savefig(save_loc)


    def export_performance_metrics(self, save_dir):
        # requires labels. Might be better to extract from the log at execute.py level, via analysis.py.
        data = {}
        data['mean square error'] = self.mean_square_error
        data['mean absolute error'] = self.mean_abs_error
        data['binomial loss'] = self.mean_bin_loss
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(data, f) 


def match_id_strs_to_catalog(id_strs, catalog):
    filtered_catalog = catalog[catalog['subject_id'].isin(set(id_strs))]
    # id strs is sorted by acquisition - catalog must also become sorted
    # careful - reindexing by int-like strings will actually do int reindexing, not what I want
    filtered_catalog['subject_id'] = filtered_catalog['subject_id'].astype(str)
    sorted_catalog = filtered_catalog.set_index('subject_id', drop=True).reindex(id_strs).reset_index()
    return sorted_catalog



def acquisitions_vs_values(acquisitions, values, n_acquired, xlabel, save_dir):

    verify_ready_to_plot(acquisitions, n_acquired)
    
    acquired, not_acquired = values[:n_acquired], values[n_acquired:]

    fig, axes = plt.subplots(nrows=3, sharex=True)

    sns.scatterplot(
        x=values, 
        y=acquisitions, 
        hue=np.array(acquisitions) > acquisitions[n_acquired],
        ax=axes[0]
        )

    axes[0].set_ylabel('Acquisition')
    axes[0].set_xlabel(xlabel)

    axes[1].hist(acquired)
    axes[1].set_xlabel(xlabel)
    axes[1].set_title('Acquired')

    axes[2].hist(not_acquired)
    axes[2].set_xlabel(xlabel)
    axes[2].set_title('Not Acquired')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'acquistion_vs_{}.png'.format(xlabel.replace(' ', '_').lower())))


def verify_ready_to_plot(acquisitions, n_acquired):
    if acquisitions is None:
        raise ValueError('Acquistions is required')
    if len(acquisitions) < n_acquired:
        raise ValueError('N Acquired is set incorrectly: should be less than all subjects')

    # must already be sorted in descending order (no way to check 'value', ofc)
    assert np.allclose(acquisitions, np.sort(acquisitions)[::-1])
