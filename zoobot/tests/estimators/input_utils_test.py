import os
from collections import Counter

import numpy as np
import pytest
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # don't actually show any figures
import matplotlib.pyplot as plt

from zoobot.estimators import input_utils
from zoobot.tfrecord import read_tfrecord
from zoobot.tests import TEST_EXAMPLE_DIR, TEST_FIGURE_DIR



@pytest.fixture()
def batch_of_visual_check_image(visual_check_image):
    return tf.stack([visual_check_image for n in range(16)], axis=0)  # dimensions batch, height, width, channels


"""
Test augmentation applied to a single image (i.e. within map_fn)
"""

def test_geometric_augmentations_on_image(visual_check_image):

    final_image = input_utils.geometric_augmentation(visual_check_image, zoom=(1., 1.5), final_size=256)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        original_image = session.run(visual_check_image)
        final_image = np.squeeze(session.run(final_image))  # remove batch dimension

        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(original_image)
        axes[0].set_title('Before')
        axes[1].imshow(final_image)
        axes[1].set_title('After')
        fig.tight_layout()
        fig.savefig(os.path.join(TEST_FIGURE_DIR, 'geometric_augmentation_check_single_image.png'))


def test_photometric_augmentations_on_image(visual_check_image):
    final_image = input_utils.photographic_augmentation(visual_check_image, max_brightness_delta=0.1, contrast_range=(0.9, 1.1))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        input_image = session.run(visual_check_image)
        final_image = np.squeeze(session.run(final_image))  # adds a batch dimension

    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(input_image)
    axes[0].set_title('Before')
    print(final_image.shape)  # TODO assert
    axes[1].imshow(final_image)
    axes[1].set_title('After')
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'photometric_augmentation_check_single_image.png'))



def test_repeated_geometric_augmentations_on_image(batch_of_visual_check_image):
    transformed_images = input_utils.geometric_augmentation(batch_of_visual_check_image, zoom=(1., 1.5), final_size=256)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        transformed_images = session.run(transformed_images)

    fig, axes = plt.subplots(nrows=16, figsize=(4, 4 * 16))
    for image_n, image in enumerate(transformed_images):
        axes[image_n].imshow(image)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'geometric_augmentation_check_on_batch.png'))


def test_repeated_photometric_augmentations_on_image(batch_of_visual_check_image):
    transformed_images = input_utils.photographic_augmentation(batch_of_visual_check_image, max_brightness_delta=0.1, contrast_range=(0.9, 1.1))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        transformed_images = session.run(transformed_images)

    fig, axes = plt.subplots(nrows=16, figsize=(4, 4 * 16))
    for image_n, image in enumerate(transformed_images):
        axes[image_n].imshow(image)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'photometric_augmentation_check_on_batch.png'))


def test_all_augmentations_on_batch(batch_of_visual_check_image):

    input_config = input_utils.InputConfig(
        name='pytest',
        tfrecord_loc='',
        label_col='',
        initial_size=424,
        final_size=256,
        channels=3,
        batch_size=16,
        stratify=False,
        regression=False,
        repeat=False,
        shuffle=False,
        stratify_probs=None,
        geometric_augmentation=True,
        shift_range=None,
        zoom=(1., 1.5),
        fill_mode=None,
        photographic_augmentation=True,
        max_brightness_delta=0.2,
        contrast_range=(0.8, 1.2)
    )

    transformed_batch = input_utils.augment_images(batch_of_visual_check_image, input_config)

    with tf.Session() as sess:
        transformed_batch = sess.run(transformed_batch)

    assert not isinstance(transformed_batch, list)  # should be a single 4D tensor, not a list
    transformed_images = [transformed_batch[n] for n in range(len(transformed_batch))]  # back to list form
    fig, axes = plt.subplots(nrows=len(transformed_images), figsize=(4, 4 * len(transformed_images)))
    for image_n, image in enumerate(transformed_images):
        axes[image_n].imshow(image)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_FIGURE_DIR, 'all_augmentations_check.png'))


def test_predict_input_func_subbatch_with_labels(tfrecord_matrix_ints_loc, size):
    
    # tfrecord_matrix_loc
    n_galaxies = 24
    subjects, labels, _ = input_utils.predict_input_func(
        tfrecord_matrix_ints_loc,
        n_galaxies=n_galaxies,
        initial_size=size,
        mode='labels'
    )
    with tf.Session() as sess:
        subjects = sess.run(subjects)
        assert subjects.shape == (n_galaxies, size, size, 3)
        labels = sess.run(labels)
        assert len(labels) == n_galaxies
        # should not have shuffled
        assert labels[0] < labels [1] < labels [2] < labels [10] < labels[23]

def test_predict_input_func_with_id(shard_locs, size):
    n_galaxies = 24
    tfrecord_loc = shard_locs[0]
    subjects, _, id_strs = input_utils.predict_input_func(tfrecord_loc, n_galaxies=n_galaxies, initial_size=size, mode='id_str')
    with tf.Session() as sess:
        subjects, id_strs = sess.run([subjects, id_strs])
    assert subjects.shape == (n_galaxies, size, size, 3)  # does not do augmentations, that happens at predict time
    assert len(id_strs) == 24


def test_predict_input_func_subbatch_no_labels(tfrecord_matrix_loc, size):
    n_galaxies = 24
    subjects, _, _ = input_utils.predict_input_func(tfrecord_matrix_loc, n_galaxies=n_galaxies, initial_size=size, mode='matrix')
    with tf.Session() as sess:
        subjects = sess.run(subjects)
    assert subjects.shape == (n_galaxies, size, size, 3)  # does not do augmentations, that happens at predict time


def test_get_batch(tfrecord_matrix_id_loc, size, channels):
    feature_spec = read_tfrecord.matrix_id_feature_spec(size, channels)
    batch = input_utils.get_batch(tfrecord_matrix_id_loc, feature_spec, batch_size=24, shuffle=True, repeat=True)
    with tf.Session() as sess:  
        batches = []
        for i in range(10):
            batches.append(sess.run(batch))
    assert len(batches) == 10

    id_strs = [id_str.decode('utf-8') for b in batches for id_str in b['id_str']]
    assert '12' in set(id_strs)
    assert len(set(id_strs)) == 128  # all ids in tfrecord


def test_get_batch_double_locs(tfrecord_matrix_id_loc, tfrecord_matrix_id_loc_distinct):
    feature_spec = read_tfrecord.id_feature_spec()
    tfrecord_locs = [tfrecord_matrix_id_loc, tfrecord_matrix_id_loc_distinct]

    n_batches = 81  # enough to cycle through full records, and get close to expected counts
    shuffle = False

    # load individually
    batch_tf_0 = input_utils.get_batch(tfrecord_locs[0], feature_spec, batch_size=24, shuffle=shuffle, repeat=True)
    with tf.Session() as sess:  
        batches_tf_0 = []
        for i in range(n_batches):
            batches_tf_0.append(sess.run(batch_tf_0))
    assert len(batches_tf_0) == n_batches

    batch_tf_1 = input_utils.get_batch(tfrecord_locs[1], feature_spec, batch_size=24, shuffle=shuffle, repeat=True)
    with tf.Session() as sess:  
        batches_tf_1 = []
        for i in range(n_batches):
            batches_tf_1.append(sess.run(batch_tf_1))
    assert len(batches_tf_1) == n_batches

    # load from both
    batch = input_utils.get_batch(tfrecord_locs, feature_spec, batch_size=24, shuffle=shuffle, repeat=True)
    with tf.Session() as sess:  
        batches = []
        for i in range(n_batches):
            batches.append(sess.run(batch))
    assert len(batches) == n_batches

    id_strs = [id_str.decode('utf-8') for b in batches for id_str in b['id_str']]
    id_strs_0 = [id_str.decode('utf-8') for b in batches_tf_0 for id_str in b['id_str']]
    id_strs_1 = [id_str.decode('utf-8') for b in batches_tf_1 for id_str in b['id_str']]
    # for batch in batches:
        # print(batch)
    # assert False

    #  for tests to work, should be some ids only in one record or the other
    assert len(set(id_strs_0) ^ set(id_strs_1)) > 0

    #  should have loaded all ids in double-mode (n_batches is plenty to cycle through both)
    assert set(id_strs_0) | set(id_strs_1) == set(id_strs)

    counts_0 = Counter(id_strs_0)
    counts_1 = Counter(id_strs_1)
    counts = Counter(id_strs)

    # check that for all tfrecords, looped through all elements evenly (3-4 times for double, 7-8 times for single, as records are shorter)
    for counter in [counts_0, counts_1, counts]:
        most_common_n_reads = counter.most_common()[0][1]
        least_common_n_reads = counter.most_common()[::-1][0][1]
        assert most_common_n_reads == least_common_n_reads + 1  # batch size won't exactly match record size

    mean_tfrecord_0_reads = np.mean([counts[id_str] for id_str in id_strs_0])
    mean_tfrecord_1_reads = np.mean([counts[id_str] for id_str in id_strs_1])
    # images should be read at approximately the same rate, regardless of the size of the tfrecord that holds them
    assert np.abs(mean_tfrecord_0_reads - mean_tfrecord_1_reads) < 0.2

"""
Test augmentation applied by map_fn to a chain of images from from_tensor_slices
"""


# @pytest.fixture()
# def benchmark_image():
#     single_channel = np.array([[1., 2., 3., 4.] for n in range(4)])  # each channel has rows of 1 2 3 4
#     return np.array([single_channel for n in range(3)])  # copied 3 times

# """
# Functional test on fake data, saved to temporary tfrecords
# """
# def test_input_utils(stratified_tfrecord_locs, size, channels, true_image_values, false_image_values):
#     # stratified_tfrecord_locs writes up the tfrecords to read
#     # needs to be an arg but is implicitly called by pytest

#     train_batch = 64
#     test_batch = 128

#     train_loc, test_loc = stratified_tfrecord_locs
#     assert os.path.exists(train_loc)
#     assert os.path.exists(test_loc)

#     train_config = input_utils.InputConfig(
#         name='train',
#         tfrecord_loc=train_loc,
#         initial_size=size,
#         final_size=size,
#         channels=channels,
#         label_col=None,  # TODO not sure about this
#         batch_size=train_batch,
#         stratify=False,
#         regression=False,
#         repeat=False,
#         shuffle=False,
#         stratify_probs=None,
#         geometric_augmentation=False,
#         photographic_augmentation=False
#     )
#     train_features, train_labels = input_utils.get_input(train_config)
#     train_images = train_features['x']

#     train_strat_config = input_utils.InputConfig(
#         name='train',
#         tfrecord_loc=train_loc,
#         initial_size=size,
#         final_size=size,
#         channels=3,
#         label_col=None,  # TODO not sure about this
#         batch_size=train_batch,
#         stratify=True,
#         regression=False,
#         repeat=False,
#         shuffle=False,
#         stratify_probs=np.array([0.8, 0.2]),
#         geometric_augmentation=False,
#         photographic_augmentation=False
#     )
#     train_features_strat, train_labels_strat = input_utils.get_input(train_strat_config)
#     train_images_strat = train_features_strat['x']

#     test_config = input_utils.InputConfig(
#         name='test',
#         tfrecord_loc=test_loc,
#         initial_size=size,
#         final_size=size,
#         channels=3,
#         label_col=None,  # TODO not sure about this
#         batch_size=test_batch,
#         stratify=False,
#         regression=False,
#         repeat=False,
#         shuffle=False,
#         stratify_probs=None,
#         geometric_augmentation=False,
#         photographic_augmentation=False
#     )
#     test_features, test_labels = input_utils.get_input(test_config)
#     test_images = test_features['x']

#     test_strat_config = input_utils.InputConfig(
#         name='test_strat',
#         tfrecord_loc=test_loc,
#         initial_size=size,
#         final_size=size,
#         channels=3,
#         label_col=None,  # TODO not sure about this
#         batch_size=test_batch,
#         stratify=True,
#         regression=False,
#         repeat=False,
#         shuffle=False,
#         stratify_probs=np.array([0.8, 0.2]),
#         geometric_augmentation=False,
#         photographic_augmentation=False
#     )
#     test_features_strat, test_labels_strat = input_utils.get_input(test_strat_config)
#     test_images_strat = test_features_strat['x']

#     with tf.train.MonitoredSession() as sess:  # mimic Estimator environment

#         train_images, train_labels = sess.run([train_images, train_labels])
#         assert len(train_labels) == train_batch
#         assert train_images.shape[0] == train_batch
#         assert train_labels.mean() < .6  # should not be stratified
#         assert train_images.shape == (train_batch, size, size, 1)
#         verify_images_match_labels(train_images, train_labels, true_image_values, false_image_values, size)

#         train_images_strat, train_labels_strat = sess.run([train_images_strat, train_labels_strat])
#         assert len(train_labels_strat) == train_batch
#         assert train_images_strat.shape[0] == train_batch
#         assert train_labels_strat.mean() < 0.75 and train_labels_strat.mean() > 0.25  # stratify not very accurate...
#         assert train_images_strat.shape == (train_batch, size, size, 1)
#         verify_images_match_labels(train_images_strat, train_labels_strat, true_image_values, false_image_values, size)

#         test_images, test_labels = sess.run([test_images, test_labels])
#         assert len(test_labels) == test_batch
#         assert test_images.shape[0] == test_batch
#         assert test_labels.mean() < 0.6  # should not be stratified
#         assert test_images.shape == (test_batch, size, size, 1)
#         verify_images_match_labels(test_images, test_labels, true_image_values, false_image_values, size)

#         test_images_strat, test_labels_strat = sess.run([test_images_strat, test_labels_strat])
#         assert len(test_labels_strat) == test_batch
#         assert test_images_strat.shape[0] == test_batch
#         assert test_labels_strat.mean() < 0.75 and test_labels_strat.mean() > 0.25  # stratify not very accurate...
#         assert test_images_strat.shape == (test_batch, size, size, 1)
#         verify_images_match_labels(test_images_strat, test_labels_strat, true_image_values, false_image_values, size)


# def verify_images_match_labels(images, labels, true_values, false_values, size):
#     for example_n in range(len(labels)):
#         if labels[example_n] == 1:
#             expected_values = true_values
#         else:
#             expected_values = false_values
#         expected_matrix = np.ones((size, size, 1), dtype=np.float32) * expected_values
#         assert images[example_n, :, :, :] == pytest.approx(expected_matrix)


# def test_input_utils_visual(tfrecord_matrix_ints_loc, size, channels):
#     # example_tfrecords sets up the tfrecords to read - needs to be an arg but is implicitly called by pytest
#     batch_size = 16

#     config = input_utils.InputConfig(
#         name='train',
#         tfrecord_loc=tfrecord_matrix_ints_loc,
#         initial_size=size,
#         final_size=size,
#         channels=channels,
#         label_col=None,  # TODO not sure about this
#         batch_size=batch_size,
#         stratify=False,
#         regression=False,
#         repeat=False,
#         shuffle=False,
#         stratify_probs=None,
#         geometric_augmentation=False,
#         photographic_augmentation=False)

#     batch_images, _ = input_utils.load_batches(config)

#     with tf.train.MonitoredSession() as sess:
#         batch_images = sess.run(batch_images)

#     plt.clf()
#     plt.imshow(batch_images[0])
#     plt.savefig(os.path.join(TEST_FIGURE_DIR + 'original_loaded_image.png'))
