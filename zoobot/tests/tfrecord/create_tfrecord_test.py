import pytest

import os

import tensorflow as tf
import numpy as np

from zoobot.tfrecord import create_tfrecord, read_tfrecord

from zoobot.tfrecord.tfrecord_io import load_dataset


@pytest.fixture()
def extra_data_feature_spec(size, channels):
    return {
        'label': tf.FixedLenFeature([], tf.float32),
        'matrix': tf.FixedLenFeature([], tf.string),
        'an_int': tf.FixedLenFeature([], tf.int64),
        'a_float': tf.FixedLenFeature([], tf.float32),
        'some_floats': tf.FixedLenFeature([3], tf.float32),
        'a_string': tf.FixedLenFeature([], tf.string)}


def test_serialize_image_example(visual_check_image_data, size, channels):
    serialized_example = create_tfrecord.serialize_image_example(visual_check_image_data, label=1.)
    # parse back and confirm it matches. Must be within session for tensors to be comparable to np
    with tf.Session() as sess:
        example = tf.parse_single_example(
            serialized_example, 
            features=read_tfrecord.matrix_label_feature_spec(size, channels, float_label=True)
            )
        recovered_matrix = tf.io.decode_raw(example['matrix'], out_type=tf.uint8).eval()
        assert np.allclose(recovered_matrix, visual_check_image_data.flatten())
        assert example['label'].eval() == 1.


def test_serialize_image_example_extra_data(visual_check_image_data, extra_data_feature_spec):
    label = 1.
    an_int = 1
    a_float = .5
    some_floats = np.array([1., 2., 3.])
    a_string = 'hello world'
    extra_kwargs = {
        'an_int': 1,
        'a_float': .5,
        'some_floats': some_floats,
        'a_string': a_string
    }
    serialized_example = create_tfrecord.serialize_image_example(
        visual_check_image_data,
        label=label, 
        **extra_kwargs
    )
    with tf.Session() as sess:
        example = tf.parse_single_example(serialized_example, extra_data_feature_spec)
        recovered_matrix = tf.io.decode_raw(example['matrix'], out_type=tf.uint8).eval()
        assert np.allclose(recovered_matrix, visual_check_image_data.flatten())
        assert example['label'].eval() == label
        assert example['an_int'].eval() == an_int
        assert np.isclose(example['a_float'].eval(), a_float)
        assert np.allclose(example['some_floats'].eval(), some_floats)
        assert example['a_string'].eval() == a_string.encode()


# def read_first_example(example_loc, feature_spec):
#     dataset = load_dataset(example_loc, feature_spec)
#     iterator = dataset.make_one_shot_iterator()
#     return iterator.get_next()


# def test_matrix_to_tfrecord(matrix_label_feature_spec, visual_check_image_data, size, tfrecord_dir):
#     label = 1
#     save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
#     assert not os.path.exists(save_loc)
#     writer = tf.python_io.TFRecordWriter(save_loc)
#     writer.write(create_tfrecord.serialize_image_example(visual_check_image_data.fal, label))
#     assert os.path.exists(save_loc)
#     writer.close()  # important
#     feature_spec = matrix_label_feature_spec(size=size)
#     # load tfrecord as dataset, read the first example with parser/spec
#     example_features = read_first_example(save_loc, feature_spec)
#     saved_label = example_features['label']
#     saved_image = example_features['matrix']
#     # execute graph (with queuerunners)
#     with tf.Session() as sess:
#         init = tf.global_variables_initializer()
#         sess.run(init)
#         tf.train.start_queue_runners(sess=sess)
#         saved_label, saved_image = sess.run([saved_label, saved_image])
#         assert saved_label == label
#         # image returned flat, float32 dtype
#         assert saved_image == pytest.approx(visual_check_image_data.flatten().astype(np.float32))




# def test_matrix_to_tfrecord_with_extra_data(example_image_data, tfrecord_dir, extra_data_feature_spec):
#     example_label = 1
#     save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
#     example_extra_data = {
#         'an_int': 1,
#         'a_float': .5,
#         'some_floats': np.array([1., 2., 3.])
#     }
#     assert not os.path.exists(save_loc)
#     writer = tf.python_io.TFRecordWriter(save_loc)
#     create_tfrecord.image_to_tfrecord(example_image_data, example_label, writer, extra_data=example_extra_data.copy())
#     assert os.path.exists(save_loc)
#     writer.close()

#     example_features = read_first_example(save_loc, extra_data_feature_spec)

#     with tf.Session() as sess:
#         init = tf.global_variables_initializer()
#         sess.run(init)
#         tf.train.start_queue_runners(sess=sess)
#         example_features = sess.run(example_features)
#         label = example_features['label']
#         image = example_features['matrix']
#         an_int = example_features['an_int']
#         a_float = example_features['a_float']
#         some_floats = example_features['some_floats']

#         assert label == example_label
#         assert image == pytest.approx(example_image_data.flatten().astype(np.float32))
#         assert an_int == example_extra_data['an_int']
#         assert a_float == pytest.approx(example_extra_data['a_float'])
#         assert some_floats == pytest.approx(example_extra_data['some_floats'].flatten().astype(np.float32))

#
# def test_matrix_to_tfrecord_with_two_examples(example_image_data, tfrecord_dir):
#     save_loc = '{}/example.tfrecords'.format(tfrecord_dir)
#     assert not os.path.exists(save_loc)
#
#     example_label_a = 1
#     example_extra_data_a = {
#         'an_int': 1,
#         'a_float': .5,
#         'some_floats': np.array([1., 2., 3.])
#     }
#     example_label_b = 0
#     example_extra_data_b = {
#         'an_int': 2,
#         'a_float': .3,
#         'some_floats': np.array([3., 2., 1.])
#     }
#
#     writer = tf.python_io.TFRecordWriter(save_loc)
#     create_tfrecord.image_to_tfrecord(example_image_data, example_label_a, writer, extra_data=example_extra_data_a.copy())
#     create_tfrecord.image_to_tfrecord(example_image_data, example_label_b, writer, extra_data=example_extra_data_b.copy())
#     assert os.path.exists(save_loc)
#     writer.close()
#
#     dataset = load_dataset(save_loc, extra_data_feature_spec())
#     iterator = dataset.make_one_shot_iterator()
#
#     example_a = iterator.get_next()
#     example_b = iterator.get_next()
#
#     with tf.Session() as sess:
#         init = tf.global_variables_initializer()
#         sess.run(init)
#         tf.train.start_queue_runners(sess=sess)
#
#         example_a = sess.run(example_a)
#         assert example_a['label'] == example_label_a
#         assert example_a['matrix'] == pytest.approx(example_image_data.flatten().astype(np.float32))
#         assert example_a['an_int'] == example_extra_data_a['an_int']
#         assert example_a['a_float'] == pytest.approx(example_extra_data_a['a_float'])
#         assert example_a['some_floats'] == pytest.approx(example_extra_data_a['some_floats'])
#
#         example_b = sess.run(example_b)
#         assert example_b['label'] == example_label_b
#         assert example_b['matrix'] == pytest.approx(example_image_data.flatten().astype(np.float32))
#         assert example_b['an_int'] == example_extra_data_b['an_int']
#         assert example_b['a_float'] == pytest.approx(example_extra_data_b['a_float'])
#         assert example_b['some_floats'] == pytest.approx(example_extra_data_b['some_floats'])
