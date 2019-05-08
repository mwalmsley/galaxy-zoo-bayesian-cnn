# from scipy import stats
import numpy as np
import matplotlib
matplotlib.use('Agg') # TODO move this to .matplotlibrc
import matplotlib.pyplot as plt
import seaborn as sns

# from zoobot.estimators import make_predictions
from zoobot.uncertainty import sample_statistics


def visualise_calibration(alpha_eval, coverage_at_alpha, save_loc):
    # will eventually add several series, one per dropout rate (residuals for best rate only)
    sns.set(font_scale=2)
    sns.set_context('paper')

    confidence_level = 1 - alpha_eval

    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
    ax1.plot(confidence_level, coverage_at_alpha, label='Observed Coverage')
    ax1.plot(confidence_level, confidence_level, 'k--', label='Coverage = Confidence')
    ax1.set_xscale('log')
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.NullFormatter())
    ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_ylabel('Coverage Fraction')
    ax1.legend()

    ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax2.plot(confidence_level, coverage_at_alpha - confidence_level)
    ax2.axhline(0., color='k')
    ax2.set_xscale('log')
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_ylabel('Residual')
    ax2.set_xlabel('Confidence Level')

    plt.tight_layout()
    plt.savefig(save_loc)


def check_coverage_fractions(predictions, true_params):
    alpha_eval = np.log10(np.logspace(0.05, 0.32))
    coverage_at_alpha = np.zeros_like(alpha_eval)
    # could vectorise this
    for n in range(len(alpha_eval)):
        coverage_at_alpha[n] = coverage_fraction(
            predictions, 
            true_params,
            alpha=alpha_eval[n]
        )
    return alpha_eval, coverage_at_alpha


def coverage_fraction(predictions, true_params, alpha):
    intervals = np.array(
        [sample_statistics.samples_to_interval(
            predictions[n], 
            alpha=alpha) 
            for n in range(len(predictions))]
        )
    within_interval = (true_params > intervals[:, 0]) & (true_params < intervals[:, 1])
    return float(np.sum(within_interval)) / float(len(predictions))  # coverage fraction

