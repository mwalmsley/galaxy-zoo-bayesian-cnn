import logging
import os
import shutil
import time
import copy
from functools import partial

import numpy as np
import tensorflow as tf
from zoobot.estimators import input_utils, bayesian_estimator_funcs


class RunEstimatorConfig():

    def __init__(
            self,
            initial_size,
            final_size,
            channels,
            label_col,
            epochs=50,
            train_steps=30,
            eval_steps=3,
            batch_size=128,
            min_epochs=0,
            early_stopping_window=10,
            max_sadness=4.,
            log_dir='runs/default_run_{}'.format(time.time()),
            save_freq=10,
            warm_start=True,
            warm_start_settings=None
    ):  # TODO refactor for consistent order
        self.initial_size = initial_size
        self.final_size=final_size
        self.channels = channels
        self.label_col = label_col
        self.epochs = epochs
        self.train_batches = train_steps
        self.eval_batches = eval_steps
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.warm_start = warm_start
        self.max_sadness = max_sadness
        self.early_stopping_window = early_stopping_window
        self.min_epochs = min_epochs
        self.train_config = None
        self.eval_config = None
        self.model = None
        self.warm_start_settings = warm_start_settings

    
    def assemble(self, train_config, eval_config, model):
        self.train_config = train_config
        self.eval_config = eval_config
        self.model = model
        assert self.is_ready_to_train()


    def is_ready_to_train(self):
        # TODO can make this check much more comprehensive
        return (self.train_config is not None) and (self.eval_config is not None)

    def log(self):
        logging.info('Parameters used: ')
        for config_object in [self, self.train_config, self.eval_config, self.model]:
            for key, value in config_object.asdict().items():
                logging.info('{}: {}'.format(key, value))

    # TODO move to shared utilities
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])


def train_input(input_config):
    return input_utils.get_input(config=input_config)


def eval_input(input_config):
    return input_utils.get_input(config=input_config)


def run_estimator(config):
    """
    Train and evaluate an estimator

    Args:
        config (RunEstimatorConfig): parameters controlling both estimator and train/test procedure

    Returns:
        None
    """
    assert config.is_ready_to_train()

    if not config.warm_start:  # don't try to load any existing models
        if os.path.exists(config.log_dir):
            shutil.rmtree(config.log_dir)


    # Create the Estimator

    model_fn_partial = partial(bayesian_estimator_funcs.estimator_wrapper)

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=5*60,  # Save checkpoints every 5 minutes (but actually much faster)
        keep_checkpoint_max=5      # Retain the 5 most recent checkpoints
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn_partial,
        model_dir=config.log_dir,
        params=config.model,
        config=estimator_config,
        warm_start_from=config.warm_start_settings
    )

    def serving_input_receiver_fn_image():
        """
        An input receiver that expects an image array (batch, size, size, channels)
        """
        images = tf.placeholder(
            dtype=tf.float32,
            shape=(None, config.initial_size, config.initial_size, config.channels), 
            name='images')
        receiver_tensors = {'examples': images}  # dict of tensors the predictor will expect. Images as above.

        predict_config = copy.copy(config.eval_config)
        predict_config.name = 'predict'
        predict_config.shuffle = False
        predict_config.repeat = False

        new_features = input_utils.preprocess_batch(  # apply greyscale, augment, etc
            images,
            config=config.eval_config  # using eval config setup
        )
        return tf.estimator.export.ServingInputReceiver(new_features, receiver_tensors)

    train_input_partial = partial(train_input, input_config=config.train_config)
    eval_input_partial = partial(eval_input, input_config=config.eval_config)

    train_logging, eval_logging, predict_logging = config.model.logging_hooks

    eval_loss_history = []
    epoch_n = 0

    while epoch_n < config.epochs:

        logging.info('Beginning epoch {}'.format(epoch_n))

        # Train the estimator
        # logging.debug('training {} steps'.format(config.train_batches))
        estimator.train(
            input_fn=train_input_partial,
            steps=config.train_batches,
            # hooks=train_logging + [tf.train.ProfilerHook(save_secs=10)]
            hooks=train_logging
        )

        # Evaluate the estimator and logging.info results
        eval_metrics = estimator.evaluate(
            input_fn=eval_input_partial,
            steps=config.eval_batches,
            hooks=eval_logging
        )
        eval_loss_history.append(eval_metrics['loss'])

        
        if (epoch_n % config.save_freq == 0) or (epoch_n == config.epochs):
            save_model(estimator, config, epoch_n, serving_input_receiver_fn_image)

        if epoch_n > config.min_epochs:
            sadness = early_stopper(eval_loss_history, config.early_stopping_window)
            logging.info('Current sadness: {}'.format(sadness))
            if sadness > config.max_sadness:
                logging.info('Ending training at epoch {} with {} sadness'.format(
                    epoch_n,
                    sadness))
                break  # stop training early

        logging.info('End epoch {}'.format(epoch_n))
        epoch_n += 1

    logging.info('All epochs completed - finishing gracefully')

    return eval_loss_history


def loss_instability(loss_history, window):
    return (np.mean(loss_history[-window:]) / np.min(loss_history[-window:])) - 1


def generalised_loss(loss_history):
    return (loss_history[-1] / np.min(loss_history)) - 1


def early_stopper(loss_history, window):
    return generalised_loss(loss_history) / loss_instability(loss_history, window)


def save_model(estimator, config, epoch_n, serving_input_receiver_fn):
    """

    Args:
        estimator:
        config (R:
        epoch_n:
        serving_input_receiver_fn:

    Returns:

    """
    logging.info('Saving model at epoch {} to {}'.format(epoch_n, config.log_dir))
    estimator.export_savedmodel(
        export_dir_base=config.log_dir,
        serving_input_receiver_fn=serving_input_receiver_fn)
