
import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

from zoobot.active_learning import acquisition_utils


def evaluate_discrete_coverage(volunteer_votes, mean_k_predictions):
    data = []
    if volunteer_votes.mean() < 1.:  # make sure this isn't the vote fractions!
        raise ValueError('Expected integer vote counts (k), not fractions, but mean "vote" is below 1.')
    n_subjects = len(volunteer_votes)
    max_possible_k = 80  # don't test errors higher than this
    # mean_posterior = acquisition_utils.get_mean_predictions(sample_probs_by_k)
    for error_bar_width in range(max_possible_k + 1):  # include max_error = max_k in range
        for subject_n in range(n_subjects):
            p_of_k = mean_k_predictions[subject_n]
            expected_k = int(np.sum(p_of_k * np.arange(len(p_of_k))))  # expected k per subject
            # most_likely_k = p_of_k.argmax()
            actual_k = volunteer_votes[subject_n]
            # use min/max because slice (below) will fail if min_k or max_k are negative
            max_k = np.min([expected_k + error_bar_width, len(p_of_k)])
            min_k = np.max([expected_k - error_bar_width, 0])
            assert min_k <= max_k
            # warning, slice will fail if min_k or max_k are negative
            p_k_in_error_bar = np.sum(p_of_k[min_k:max_k+1])  # include max_k in slice
            k_in_error_bar = float(min_k <= actual_k <= max_k)
            data.append({
                'max_state_error': error_bar_width,
                'prediction': p_k_in_error_bar,
                'observed': k_in_error_bar,
                'max_k': max_k,
                'min_k': min_k,
                'most_likely_k': expected_k,
                'actual_k': actual_k,
                'subject_n': subject_n
                })
    df = pd.DataFrame(data=data)
    return df


def reduce_coverage_df(df):
    return df.groupby('max_state_error').agg({'prediction': 'mean', 'observed': 'mean'}).reset_index()


def calibrate_predictions(df):
    assert len(df) >= 4
    lr = sklearn.linear_model.LogisticRegression()
    df = df.sample(frac=1).reset_index(drop=True)
    train_df, test_df = df[:int(len(df)/4)], df[int(len(df)/4):]
    X_train = np.array(train_df['prediction']).reshape(-1, 1)
    X_test = np.array(test_df['prediction']).reshape(-1, 1)
    y_train = np.array(train_df['observed'])                
    lr.fit(X_train, y_train)
    test_df['prediction_calibrated'] = lr.predict_proba(X_test)[:,1]
    return test_df


def plot_coverage_df(df, ax):
    cols_to_plot = ['prediction', 'observed']
    if 'prediction_calibrated' in df.columns:
        cols_to_plot.append('prediction_calibrated')
    for col in cols_to_plot:
        sns.lineplot(data=df, x='max_state_error', y=col, ax=ax)
    legend_mapping = {
        'prediction': 'Model Expects',
        'observed': 'Actual',
        'prediction_calibrated': 'Calibrated Prediction'
    }
    ax.legend([legend_mapping[col] for col in cols_to_plot])
    ax.set_xlabel('Max Allowed Vote Error')
    ax.set_ylabel('Frequency Within Max Error')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))  # must expect 'x' kw arg

