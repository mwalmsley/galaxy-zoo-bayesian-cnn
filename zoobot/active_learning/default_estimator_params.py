import logging
import os

import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from zoobot.estimators import bayesian_estimator_funcs, run_estimator, input_utils


def get_run_config(params, log_dir, train_records, eval_records, learning_rate, epochs):

    channels = 3

    run_config = run_estimator.RunEstimatorConfig(
        initial_size=params.initial_size,
        final_size=params.final_size,
        channels=channels,
        label_col='label',
        epochs=epochs,  # to tweak - 2000 for overnight at 8 iters, 650 for 2h per iter
        train_steps=15,
        eval_steps=5,
        batch_size=256,
        min_epochs=2000,  # no early stopping
        early_stopping_window=10,  # to tweak
        max_sadness=5.,  # to tweak
        log_dir=log_dir,
        save_freq=10,
        warm_start=params.warm_start
    )

    train_config = input_utils.InputConfig(
        name='train',
        tfrecord_loc=train_records,
        label_col=run_config.label_col,
        stratify=False,
        shuffle=True,
        repeat=True,
        stratify_probs=None,
        geometric_augmentation=True,
        photographic_augmentation=True,
        zoom=(1.1, 1.3),  # SMOOTH MODE
        contrast_range=(0.98, 1.02),
        fill_mode='wrap',
        batch_size=run_config.batch_size,
        initial_size=run_config.initial_size,
        final_size=run_config.final_size,
        channels=run_config.channels,
        noisy_labels=False,  # train using softmax proxy for binomial loss,
        greyscale=True,  # both modes
        zoom_central=False  # SMOOTH MODE
    )

    eval_config = input_utils.InputConfig(
        name='eval',
        tfrecord_loc=params.eval_tfrecord_loc,
        label_col=run_config.label_col,
        stratify=False,
        shuffle=True,
        repeat=False,
        stratify_probs=None,
        geometric_augmentation=True,
        photographic_augmentation=True,
        zoom=(1.1, 1.3),  # SMOOTH MODE
        contrast_range=(0.98, 1.02),
        fill_mode='wrap',
        batch_size=run_config.batch_size,
        initial_size=run_config.initial_size,
        final_size=run_config.final_size,
        channels=run_config.channels,
        noisy_labels=False,  # eval using binomial loss
        greyscale=True,  # both modes
        zoom_central=False  # SMOOTH MODE
    )

    model = bayesian_estimator_funcs.BayesianModel(
        learning_rate=learning_rate,
        optimizer=tf.train.AdamOptimizer,
        conv1_filters=32,
        conv1_kernel=3,
        conv2_filters=64,
        conv2_kernel=3,
        conv3_filters=128,
        conv3_kernel=3,
        dense1_units=128,
        dense1_dropout=0.5,
        predict_dropout=0.5,  # change this to calibrate
        regression=True,  # important!
        log_freq=10,
        image_dim=run_config.final_size  # not initial size
    )

    run_config.assemble(train_config, eval_config, model)
    return run_config
