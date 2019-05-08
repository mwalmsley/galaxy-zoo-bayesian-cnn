import os

import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tfrecord import read_tfrecord


def test_matrix_label_feature_spec(size, channels, serialized_matrix_label_example):
    example = tf.parse_single_example(
        serialized_matrix_label_example, 
        features=read_tfrecord.matrix_label_feature_spec(size, channels))


def test_matrix_label_id_feature_spec(size, channels, serialized_matrix_id_example):
    example = tf.parse_single_example(
        serialized_matrix_id_example, 
        features=read_tfrecord.matrix_id_feature_spec(size, channels)
    )


def test_load_examples_from_tfrecord(tfrecord_matrix_ints_loc, size, channels):
    feature_spec = read_tfrecord.matrix_label_feature_spec(size, channels, float_label=True)
    tfrecord_locs = [tfrecord_matrix_ints_loc]
    examples = read_tfrecord.load_examples_from_tfrecord(tfrecord_locs, feature_spec, 5)
    assert len(examples) == 5
    example = examples[0]
    assert 0. < example['matrix'].mean() < 255.
    assert 0. <= example['label'] <= 1.
    assert isinstance(example['label'], np.float32)
    plt.clf()
    plt.imshow(example['matrix'].reshape(size, size, channels))
    plt.savefig(os.path.join(TEST_EXAMPLE_DIR, 'loaded_image_from_example_tfrecord.png'))


def test_load_examples_from_tfrecord_all(tfrecord_matrix_ints_loc, size, channels):
    feature_spec = read_tfrecord.matrix_label_feature_spec(size, channels, float_label=True)
    tfrecord_locs = [tfrecord_matrix_ints_loc]
    examples = read_tfrecord.load_examples_from_tfrecord(tfrecord_locs, feature_spec, None)
    assert len(examples) > 5
    example = examples[0]
    assert 0. < example['matrix'].mean() < 255.
    assert 0. <= example['label'] <= 1.
    assert isinstance(example['label'], np.float32)
    plt.clf()
    plt.imshow(example['matrix'].reshape(size, size, channels))
    plt.savefig(os.path.join(TEST_EXAMPLE_DIR, 'loaded_image_from_example_tfrecord.png'))


def test_show_example(parsed_example, size, channels):
    fig, ax = plt.subplots()
    read_tfrecord.show_example(parsed_example, size, channels, ax)
    fig.savefig(os.path.join(TEST_EXAMPLE_DIR, 'show_example_for_visual_check_image.png'))


def test_show_examples(parsed_example, size, channels):
    fig, axes = read_tfrecord.show_examples([parsed_example for n in range(6)], size, channels)
    fig.savefig(os.path.join(TEST_EXAMPLE_DIR, 'show_examples_for_many_visual_check_image.png'))
