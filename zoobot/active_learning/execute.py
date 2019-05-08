import argparse
import os
import shutil
import logging
import json
import time
import sqlite3
import json
import subprocess
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
import git

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params, make_shards, analysis, iterations, acquisition_utils
from zoobot.tests import TEST_EXAMPLE_DIR

class ActiveConfig():
    """
    Define and run active learning using a tensorflow estimator on pre-made shards
    (see make_shards.py for shard creation)
    """


    def __init__(
        self,
        shard_config,
        run_dir,
        n_iterations, 
        shards_per_iter,  # 4 mins per shard of 4096 images
        subjects_per_iter,
        initial_estimator_ckpt):
        """
        Controller to define and run active learning on pre-made shards

        To use:
        active_config.prepare_run_folders()
        assert active_config.ready()
        active_config.run(
            train_callable,
            get_acquisition_func
        )
        For the form of train_callable and get_acquisition func, see active_config.run
        
        Args:
            shard_config (ShardConfig): metadata of shards, e.g. location on disk, image size, etc.
            run_dir (str): path to save run outputs e.g. trained models, new shards
            iterations (int): how many iterations to train the model (via train_callable)
            shards_per_iter (int): how many shards to find acquisition values for
            subjects_per_iter (int): how many subjects to acquire per training iteration
            initial_estimator_ckpt (str): path to checkpoint folder (datetime) of est. for initial iteration
        """
        self.shards = shard_config
        self.run_dir = run_dir

        self.n_iterations = n_iterations  
        self.subjects_per_iter = subjects_per_iter
        self.shards_per_iter = shards_per_iter

        self.initial_estimator_ckpt = initial_estimator_ckpt
        self.db_loc = os.path.join(self.run_dir, 'run_db.db')  

        self.prepare_run_folders()


    # TODO test the case where run_dir does not yet exist
    def prepare_run_folders(self):
        """
        Create the folders needed to run active learning. 
        Copy the shard database, to be modified by the run
        Wipes any existing folders in run_dir
        """
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)

        directories = [self.run_dir]
        for directory in directories:
            os.mkdir(directory)

        shutil.copyfile(self.shards.db_loc, self.db_loc)


    def ready(self):
        assert self.shards.ready()  # delegate
        if self.initial_estimator_ckpt is not None:
            assert os.path.isdir(self.initial_estimator_ckpt)
            assert os.path.exists(os.path.join(self.initial_estimator_ckpt, 'saved_model.pb'))
        assert os.path.isdir(self.run_dir)
        return True


    def run(self, train_callable, acquisition_func, n_samples=20):
        """Main active learning training loop. 
        
        Learn with train_callable
        Calculate acquisition functions for each subject in the shards
        Load .fits of top subjects and save to a new shard
        Repeat for self.iterations
        After each iteration, copy the model history to new directory and start again
        Designed to work with tensorflow estimators
        
        Args:
            train_callable (func): train a tf model. Arg: list of tfrecord locations
            acquisition_func (func): expecting samples of shape [n_subject, n_sample]
        """
        # clear any leftover mocked labels awaiting collection
        # won't do this in production
        from zoobot.active_learning import mock_panoptes
        if os.path.exists(mock_panoptes.SUBJECTS_REQUESTED):
            os.remove(mock_panoptes.SUBJECTS_REQUESTED)

        assert self.ready()
        db = sqlite3.connect(self.db_loc)
        all_shard_locs = [os.path.join(self.shards.shard_dir, os.path.split(loc)[-1]) for loc in active_learning.get_all_shard_locs(db)]
        shards_iterable = itertools.cycle(all_shard_locs)  # cycle through shards

        iteration_n = 0
        initial_estimator_ckpt = self.initial_estimator_ckpt  # for first iteration, the first model is the one passed to ActiveConfig
        initial_db_loc = self.db_loc
        initial_train_tfrecords = self.shards.train_tfrecord_locs()
        eval_tfrecords = self.shards.eval_tfrecord_locs()

        learning_rate = 0.001

        iterations_record = []

        while iteration_n < self.n_iterations:

            if iteration_n == 0:
                epochs = 125
            else:
                epochs = 50

            prediction_shards = [next(shards_iterable) for n in range(self.shards_per_iter)]

            iteration = iterations.Iteration(
                run_dir=self.run_dir, 
                iteration_n=iteration_n, 
                prediction_shards=prediction_shards,
                initial_db_loc=initial_db_loc,
                initial_train_tfrecords=initial_train_tfrecords,
                eval_tfrecords=eval_tfrecords,
                train_callable=train_callable,
                acquisition_func=acquisition_func,
                n_samples=n_samples,
                n_subjects_to_acquire=self.subjects_per_iter,
                initial_size=self.shards.size,
                learning_rate=learning_rate,
                initial_estimator_ckpt=initial_estimator_ckpt,  # will only warm start with --warm_start, though
                epochs=epochs)

            # train as usual, with saved_model being placed in estimator_dir
            logging.info('Training iteration {}'.format(iteration_n))
            iteration.run()

            iteration_n += 1
            initial_db_loc = iteration.db_loc
            initial_train_tfrecords = iteration.get_train_records()  # includes newly acquired shards
            initial_estimator_ckpt = iteration.estimators_dir
            iterations_record.append(iteration)

        return iterations_record


def get_train_callable(params):

    def train_callable(log_dir, train_records, eval_records, learning_rate, epochs):
        logging.info('Training model on: {}'.format(train_records))
        run_config = default_estimator_params.get_run_config(params, log_dir, train_records, eval_records, learning_rate, epochs)
        if params.test: # overrides warm_start
            run_config.epochs = 2  # minimal training, for speed

        # Do NOT update eval_config: always eval on the same fixed shard
        return run_estimator.run_estimator(run_config)

    return train_callable


def mock_acquisition_func(samples):
    logging.critical('Applying MOCK random acquisition function')
    return [np.random.rand() for n in range(len(samples))]


def get_acquisition_func(baseline, expected_votes):
    if baseline:
        logging.critical('Using mock acquisition function, baseline test mode!')
        return mock_acquisition_func
    else:  # callable expecting samples np.ndarray, returning list
        logging.critical('Using mutual information acquisition function')
        return lambda x: acquisition_utils.mutual_info_acquisition_func(x, expected_votes)  


TrainCallableParams = namedtuple(
    'TrainCallableParams', 
    ['initial_size', 'final_size', 'warm_start', 'eval_tfrecord_loc', 'test']
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute active learning')
    parser.add_argument('--shard_config', dest='shard_config_loc', type=str,
                    help='Details of shards to use')
    parser.add_argument('--run_dir', dest='run_dir', type=str,
                    help='Path to save run outputs: models, new shards, log')
    parser.add_argument('--baseline', dest='baseline', action='store_true', default=False,
                    help='Use random subject selection only')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Only do a minimal run to verify that everything works')
    parser.add_argument('--warm-start', dest='warm_start', action='store_true', default=False,
                    help='After each iteration, continue training the same model')
    args = parser.parse_args()

    log_loc = 'execute_{}.log'.format(time.time())

    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # instructions for the run
    if args.test:  # do a brief run only
        n_iterations = 2
        subjects_per_iter = 256  # tests show acquiring this way gives good tfrecords
        shards_per_iter = 2  # temp
        final_size = 32
    else:
        n_iterations = 25
        subjects_per_iter = 128
        shards_per_iter = 4
        final_size = 128  # for both modes

    # shards to use
    shard_config = make_shards.load_shard_config(args.shard_config_loc)
    new_shard_dir = os.path.dirname(args.shard_config_loc)
    shard_config.shard_dir = new_shard_dir
    attrs = [
        'train_dir',
        'eval_dir',
        'labelled_catalog_loc',
        'unlabelled_catalog_loc',
        'config_save_loc',
        'db_loc']
    for attr in attrs:
        old_loc = getattr(shard_config, attr)
        new_loc = os.path.join(new_shard_dir, os.path.split(old_loc)[-1])
        print(attr, new_loc)
        setattr(shard_config, attr, new_loc)
    
    active_config = ActiveConfig(
        shard_config, 
        args.run_dir,
        n_iterations=n_iterations, 
        subjects_per_iter=subjects_per_iter,
        shards_per_iter=shards_per_iter,
        initial_estimator_ckpt=None
    )

    # these do not change per iteration
    train_callable_params = TrainCallableParams(
        initial_size=active_config.shards.size,
        final_size=final_size,
        warm_start=args.warm_start,
        # TODO remove?
        eval_tfrecord_loc=active_config.shards.eval_tfrecord_locs(),
        test=args.test
    )

    train_callable = get_train_callable(train_callable_params)
    # TODO generalise to many classes at once, don't need to manually set expected_votes
    acquisition_func = get_acquisition_func(baseline=args.baseline, expected_votes=40)  # IMPORTANT SMOOTH MODE
    if args.test or args.baseline:
        n_samples = 2
    else:
        n_samples = 15

    ###
    iterations_record = active_config.run(train_callable, acquisition_func, n_samples)
    ###

    # finally, tidy up by moving the log into the run directory
    # could not be create here because run directory did not exist at start of script
    if os.path.exists(log_loc):  # temporary workaround for disappearing log
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        shutil.move(log_loc, os.path.join(args.run_dir, '{}.log'.format(sha)))

    analysis.show_subjects_by_iteration(iterations_record[-1].get_train_records(), 15, active_config.shards.size, 3, os.path.join(active_config.run_dir, 'subject_history.png'))
