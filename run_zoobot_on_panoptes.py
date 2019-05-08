import logging
import os
import argparse
import shutil

import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import git

from zoobot.estimators import bayesian_estimator_funcs, run_estimator, input_utils, warm_start

import panoptes_to_tfrecord
from zoobot.settings import GlobalConfig


# expects tfrecord in data, and logs/estimator in runs

# dvc run -d data/basic_split_gz2 -o results/latest_basic_run_gz2 -f basic_run.dvc python run_zoobot_on_panoptes.py --ec2=True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Zoobot on basic panoptes split')
    parser.add_argument('--ec2', dest='ec2', type=bool,
                    help='Running on EC2?')
    args = parser.parse_args()

    ec2 = args.ec2
    gc = GlobalConfig(ec2)
    initial_size = 128
    channels = 3
    final_size = 64

    if ec2:
        # train_tfrecord_loc = '/home/ubuntu/root/zoobot/data/basic_split/panoptes_featured_s{}_lfloat_train.tfrecord'.format(initial_size)
        # test_tfrecord_loc = '/home/ubuntu/root/zoobot/data/basic_split/panoptes_featured_s{}_lfloat_test.tfrecord'.format(initial_size)
        # update for GZ2
        train_tfrecord_loc = '/home/ubuntu/root/zoobot/data/basic_split_gz2/gz2_smooth_frac_{}_train.tfrecord'.format(initial_size)
        test_tfrecord_loc = '/home/ubuntu/root/zoobot/data/basic_split_gz2/gz2_smooth_frac_{}_test.tfrecord'.format(initial_size)

    else:
        train_tfrecord_loc = '/data/repos/zoobot/data/basic_split/panoptes_featured_s{}_lfloat_train.tfrecord'.format(initial_size)
        test_tfrecord_loc = '/data/repos/zoobot/data/basic_split/panoptes_featured_s{}_lfloat_test.tfrecord'.format(initial_size)


    # run_name = 'bayesian_panoptes_featured_si{}_sf{}_lfloat_filters'.format(initial_size, final_size)
    run_name = 'latest_basic_run_gz2'

    log_loc = run_name + '.log'
    logging.basicConfig(
        filename=log_loc,
        format='%(asctime)s %(message)s',
        filemode='w',
        level=logging.INFO)

    new_tfrecords = False

    if new_tfrecords:
        panoptes_to_tfrecord.save_panoptes_to_tfrecord(gc.catalog_loc, gc.tfrecord_dir)

    run_config = run_estimator.RunEstimatorConfig(
        initial_size=initial_size,
        final_size=final_size,
        channels=channels,
        label_col='label',
        epochs=1500,  # for debugging, min is 1
        train_steps=15,
        eval_steps=3,
        batch_size=256,
        min_epochs=1000,  # don't stop early automatically, wait for me
        early_stopping_window=10,
        max_sadness=4.,
        # log_dir='runs/{}'.format(run_name),
        log_dir = 'results/{}'.format(run_name),
        save_freq=25
    )

    train_config = input_utils.InputConfig(
        name='train',
        tfrecord_loc=train_tfrecord_loc,
        label_col=run_config.label_col,
        stratify=False,
        shuffle=True,
        repeat=True,
        stratify_probs=None,
        regression=True,
        geometric_augmentation=True,
        photographic_augmentation=True,
        zoom=(1., 1.2),
        fill_mode='wrap',
        batch_size=run_config.batch_size,
        initial_size=run_config.initial_size,
        final_size=run_config.final_size,
        channels=run_config.channels,
        noisy_labels=False
    )

    eval_config = input_utils.InputConfig(
        name='eval',
        tfrecord_loc=test_tfrecord_loc,
        label_col=run_config.label_col,
        stratify=False,
        shuffle=True,
        repeat=False,
        stratify_probs=None,
        regression=True,
        geometric_augmentation=True,
        photographic_augmentation=True,
        zoom=(1., 1.2),
        fill_mode='wrap',
        batch_size=run_config.batch_size,
        initial_size=run_config.initial_size,
        final_size=run_config.final_size,
        channels=run_config.channels,
        noisy_labels=False
    )

    model = bayesian_estimator_funcs.BayesianModel(
        learning_rate=0.001,
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
    assert run_config.is_ready_to_train()

    # logging.info('Parameters used: ')
    for config_object in [run_config, train_config, eval_config, model]:
        for key, value in config_object.asdict().items():
            logging.info('{}: {}'.format(key, value))
        logging.info('Next object \n')


    # start fresh?
    run_config.warm_start = False

    run_estimator.run_estimator(run_config)
    # warm_start.restart_estimator(run_config)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    shutil.move(log_loc, '{}.log'.format(sha))
