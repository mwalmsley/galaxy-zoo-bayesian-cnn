import os
from functools import partial
import logging

import numpy as np
import tensorflow as tf


def cast_bytes_of_uint8_to_float32(some_bytes):
    return tf.cast(tf.io.decode_raw(some_bytes, out_type=tf.uint8), tf.float32)


def general_parsing_function(serialized_example, features):
    """Parse example. Decode feature 'matrix' into float32 if present"""
    example = tf.parse_single_example(serialized_example, features)
    if 'matrix' in features.keys():
        example['matrix'] = cast_bytes_of_uint8_to_float32(example['matrix'])
    return example


def load_dataset(filenames, feature_spec, num_parallel_calls=4, shuffle=False):
    # TODO consider num_parallel_calls = len(list)?
    # small wrapper around loading a TFRecord as a single tensor tuples
    logging.debug('tfrecord.io: Loading dataset from {}'.format(filenames))
    parse_function = partial(general_parsing_function, features=feature_spec)    
    if isinstance(filenames, str):
        logging.debug('Loading single tfrecord')
        dataset = tf.data.TFRecordDataset(filenames)
        return dataset.map(parse_function, num_parallel_calls=num_parallel_calls)  # Parse the record into tensors
    else:
        # see https://github.com/tensorflow/tensorflow/issues/14857#issuecomment-365439428
        logging.warning('Loading multiple tfrecords with interleaving, shuffle={}'.format(shuffle))
        assert isinstance(filenames, list)
        assert len(filenames) > 0
        # tensorflow will NOT raise an error if a tfrecord file is missing, if the directory exists!
        assert all([os.path.isfile(loc) for loc in filenames])
        num_shards = len(filenames)
        # get tfrecords matching filenames, optionally shuffling order of shards to be read
        # dataset = tf.data.Dataset.list_files(filenames, shuffle=shuffle)
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(filenames, dtype=tf.string))
        if shuffle:
            dataset = dataset.shuffle(num_shards)
        # read 1 file per shard, cycling through shards
        print_op = tf.print("loading filenames:", dataset)
        with tf.control_dependencies([print_op]):
            dataset = dataset.interleave(
                lambda filename: tf.data.TFRecordDataset(filename).map(parse_function),
                cycle_length=num_shards
            )
            # could add num_parallel_calls if desired, but let's leave for now 
        # for extra randomness, may shuffle those (1st in s1, 1st in s2, ...) subjects
        return dataset
