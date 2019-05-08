import logging
import functools

import tensorflow as tf

from zoobot.estimators import input_utils, bayesian_estimator_funcs


def restart_estimator(config):

    def serving_input_receiver_fn_image():
        """
        An input receiver that expects an image array
        """
        images = tf.placeholder(
            dtype=tf.float32,
            shape=(None, config.initial_size, config.initial_size, config.channels), 
            name='images')
        receiver_tensors = {'examples': images}

        new_features = input_utils.preprocess_batch(
            images,
            config=config.eval_config
        )
        return tf.estimator.export.ServingInputReceiver(new_features, receiver_tensors)

    assert config.is_ready_to_train()

    # Create the Estimator
    model_fn_partial = functools.partial(bayesian_estimator_funcs.estimator_wrapper)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn_partial,
        model_dir=config.log_dir,
        params=config.model
    )

    save_model(estimator, config, -1, serving_input_receiver_fn_image)


def save_model(estimator, config, epoch_n, serving_input_receiver_fn):
    logging.info('Saving model at epoch {}'.format(epoch_n))
    estimator.export_savedmodel(
        export_dir_base=config.log_dir,
        serving_input_receiver_fn=serving_input_receiver_fn)


"""Deprecated

    def serving_input_receiver_fn_tfrecord_loc():
        tfrecord_loc = tf.placeholder(
            dtype=tf.string, 
            name='tfrecord_loc')
        receiver_tensors = {'examples': tfrecord_loc}

        serving_input_config = config.eval_config.copy()
        serving_input_config.name = 'serving_config'
        serving_input_config.shuffle = False  # absolutely do not reshuffle!
        serving_input_config.stratify = False  # absolutely do not stratify!
        serving_input_config.tfrecord_loc = tfrecord_loc  # change tfrecord loc to passed value
        features, _ = input_utils.get_input(serving_input_config)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
"""
