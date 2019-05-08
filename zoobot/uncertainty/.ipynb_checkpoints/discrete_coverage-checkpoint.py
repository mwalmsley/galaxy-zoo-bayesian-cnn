
import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns


def evaluate_discrete_coverage(volunteer_votes, sample_probs_by_k):
    data = []
    if volunteer_votes.mean() < 1.:  # make sure this isn't the vote fractions!
        raise ValueError('Expected integer vote counts (k), not fractions, but mean "vote" is below 1.')
    n_subjects = sample_probs_by_k.shape[0]
    max_possible_k = sample_probs_by_k.shape[2]
    mean_posterior = sample_probs_by_k.mean(axis=1)
    for subject_n in range(n_subjects):
        most_likely_k = mean_posterior[subject_n].argmax()
        for max_error_in_k in range(max_possible_k + 1):  # include max_error = max_k in range
            max_k = np.min([most_likely_k + max_error_in_k, max_possible_k])
            min_k = np.max([most_likely_k - max_error_in_k, 0])
            prediction = mean_posterior[subject_n, min_k:max_k+1].sum()  # include max_k in slice
            actual_k = volunteer_votes[subject_n]
            observed = float(min_k <= actual_k <= max_k)
            data.append({
                    'max_state_error': max_error_in_k,
                    'prediction': prediction,
                    'observed': observed,
                    'max_k': max_k,
                    'min_k': min_k,
                    'most_likely_k': most_likely_k,
                    'actual_k': actual_k
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
        'prediction': 'Prediction',
        'observed': 'Empirical',
        'prediction_calibrated': 'Calibrated Prediction'
    }
    ax.legend([legend_mapping[col] for col in cols_to_plot])
    ax.set_xlabel('Max Vote Error')
    ax.set_ylabel('Probability Within Max Error')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))  # must expect 'x' kw arg

    # TODO axis formatter for ints only
# print('Calibrated tidal predictions: {}. Non-tidal: {}'.format(np.sum(test_df['prediction'] > 0.5), np.sum(test_df['prediction'] < 0.5)))
    # print('Calibrated tidal truth: {}. Non-tidal: {}'.format(np.sum(test_df['true_label'] > 0.5), np.sum(test_df['true_label'] < 0.5)))

if __name__ == '__main__':
    # TODO refactor this with updated API
    samples = np.load('/home/mike/repos/zoobot/analysis/uncertainty/al-binomial/five_conv_fractions/samples.npy')
    labels = np.load('/home/mike/repos/zoobot/analysis/uncertainty/al-binomial/five_conv_fractions/labels.npy')
    import os
    from zoobot.active_learning import metrics
    from zoobot.tests import TEST_FIGURE_DIR
    model = metrics.Model(samples, labels, 'five_conv_fractions')
    fig, ax = plt.subplots()
    ax.hist(model.abs_error)
    fig.savefig('temp.png')
    exit(0)
    model.show_coverage(save_dir=TEST_FIGURE_DIR)
