import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from zoobot.tfrecord.tfrecord_io import load_dataset


def load_examples_from_tfrecord(tfrecord_locs, feature_spec, n_examples=None, max_examples=1e8):
    dataset = load_dataset(tfrecord_locs, feature_spec)
    iterator = dataset.make_one_shot_iterator()
    dataset = dataset.batch(1)  # 1 image per batch
    dataset = dataset.prefetch(1)
    batch = iterator.get_next()

    with tf.Session() as sess:
        if n_examples is None:  # load full record
            data = []
            while len(data) < max_examples:
                try:
                    loaded_example = sess.run(batch)
                    data.append(loaded_example)
                except tf.errors.OutOfRangeError:
                    logging.debug('tfrecords {} exhausted'.format(tfrecord_locs))
                    break
        else:  # load exactly n examples, or throw an error
            logging.debug('Loading the first {} examples from {}'.format(n_examples, tfrecord_locs))
            data = [sess.run(batch) for n in range(n_examples)]

    return data


def matrix_feature_spec(size, channels):  # used for predict mode
    return {
        "matrix": tf.FixedLenFeature([], tf.string)}


def matrix_label_feature_spec(size, channels, float_label=True):
    if float_label:
        label_dtype = tf.float32
    else:
        label_dtype = tf.int64
    return {
        "matrix": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature((), label_dtype)}


def custom_feature_spec(features_requested):
    # TODO properly, with error checking
    features = {}
    if 'matrix' in features_requested:
        features["matrix"] = tf.FixedLenFeature([], tf.string)
    if 'label' in features_requested:
        features["label"] = tf.FixedLenFeature([], tf.int64)
    if 'total_votes' or 'count' in features_requested:
        features["total_votes"] = tf.FixedLenFeature([], tf.int64)
    if 'id_str' in features_requested:
        features["id_str"] = tf.FixedLenFeature([], tf.string)
    return features


def matrix_label_counts_feature_spec():
    return {
        "matrix": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature((), tf.int64),
        "total_votes": tf.FixedLenFeature([], tf.int64)
    }


def id_label_counts_feature_spec():
    return {
        "id_str": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.int64),
        "total_votes": tf.FixedLenFeature([], tf.int64)
    }


def matrix_id_feature_spec(size, channels):
    return {
        "matrix": tf.FixedLenFeature([], tf.string),
        "id_str": tf.FixedLenFeature((), tf.string)
        }


def matrix_label_id_feature_spec(size, channels):
    return {
        "matrix": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature((), tf.float32),
        "id_str": tf.FixedLenFeature((), tf.string)
        }


def id_feature_spec():
    return {"id_str": tf.FixedLenFeature((), tf.string)}


def id_label_feature_spec():
    return {
        "label": tf.FixedLenFeature((), tf.float32),
        "id_str": tf.FixedLenFeature((), tf.string)
        }


# not required, use tf.parse_single_example directly
# def parse_example(example, size, channels):
#     features = {
#         'matrix': tf.FixedLenFeature((size * size * channels), tf.float32),
#         'label': tf.FixedLenFeature([], tf.int64),
#         }

#     return tf.parse_single_example(example, features=features)


# these are actually not related to reading a tfrecord, they are very general
def show_examples(examples, size, channels):
    # simple wrapper for pretty example plotting
    # TODO make plots in a grid rather than vertical column
    fig, axes = plt.subplots(nrows=len(examples), figsize=(4, len(examples) * 3))
    for n, example in enumerate(examples):
        show_example(example, size, channels, ax=axes[n])
    fig.tight_layout()
    return fig, axes


def show_example(example, size, channels, ax, show_label=False):  # modifies ax inplace
    # saved as floats but truly int, show as int
    im = example['matrix'].astype(np.uint8).reshape(size, size, channels)
    ax.axis('off')
    if show_label:
        label = example['label']
        if isinstance(label, int):
            name_mapping = {
                0: 'Feat.',
                1: 'Smooth'
            }
            label_str = name_mapping[label]
        else:
            label_str = '{:.2}'.format(label)
        ax.text(60, 110, label_str, fontsize=16, color='r')
