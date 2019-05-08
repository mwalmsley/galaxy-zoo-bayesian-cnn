import os
import shutil
import logging
import json
import sqlite3

import numpy as np

from zoobot.estimators import make_predictions
from zoobot.active_learning import mock_panoptes
from zoobot.active_learning import active_learning, metrics, acquisition_utils


class Iteration():

    def __init__(
        self, 
        run_dir,
        iteration_n,
        prediction_shards,
        initial_db_loc,
        initial_train_tfrecords,
        eval_tfrecords,
        train_callable,
        acquisition_func,
        n_samples,  # may need more samples?
        n_subjects_to_acquire,
        initial_size,
        learning_rate,
        epochs,
        initial_estimator_ckpt=None
        ):

        self.name = 'iteration_{}'.format(iteration_n)
        # shards should be unique, or everything falls apart.
        assert len(prediction_shards) == len(set(prediction_shards))
        self.prediction_shards = prediction_shards
        
        for (tfrecords, attr) in [
            (initial_train_tfrecords, 'initial_train_tfrecords'), # acquired up to start of iteration
            (eval_tfrecords, 'eval_tfrecords')]:
            assert isinstance(initial_train_tfrecords, list)
            try:
                assert all([os.path.isfile(loc) for loc in initial_train_tfrecords])
            except AssertionError:
                logging.critical('Fatal error: missing {}!'.format(attr))
                logging.critical(tfrecords)
            setattr(self, attr, tfrecords)

        assert callable(train_callable)
        self.train_callable = train_callable
        assert callable(acquisition_func)
        self.acquisition_func = acquisition_func
        self.n_samples = n_samples
        self.n_subjects_to_acquire = n_subjects_to_acquire
        self.initial_size = initial_size  # need to know what size to write new images to shards
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.iteration_dir = os.path.join(run_dir, self.name)
        self.estimators_dir = os.path.join(self.iteration_dir, 'estimators')
        self.acquired_tfrecords_dir = os.path.join(self.iteration_dir, 'acquired_tfrecords')
        self.metrics_dir = os.path.join(self.iteration_dir, 'metrics')

        os.mkdir(self.iteration_dir)
        os.mkdir(self.acquired_tfrecords_dir)
        os.mkdir(self.metrics_dir)

        self.db_loc = os.path.join(self.iteration_dir, 'iteration.db')
        assert os.path.isfile(initial_db_loc)
        shutil.copy(initial_db_loc, self.db_loc)
        self.db = sqlite3.connect(self.db_loc)

        # TODO have a test that verifies new folder structure
        self.initial_estimator_ckpt = initial_estimator_ckpt
        
        src = self.initial_estimator_ckpt
        dest = self.estimators_dir
        if initial_estimator_ckpt is not None:
            logging.info('Copying {} initial estimator ckpt'.format(initial_estimator_ckpt))

            # copy the initial estimator folder inside estimators_dir, keeping the same name
            shutil.copytree(
                src=src, 
                dst=dest
            )

            # copy the files only, subdirs are saved models
            # [shutil.copy(f, dest) for f in os.listdir(src) if os.path.isfile(f)]

            # remove this log from the copy, to save space
            [os.remove(os.path.join(dest, f)) for f in os.listdir(dest) if f.startswith('events.out.tfevents')]
        else:
            os.mkdir(self.estimators_dir)

        # record which tfrecords were used, for later analysis
        self.tfrecords_record = os.path.join(self.iteration_dir, 'train_records_index.json')


    def get_acquired_tfrecords(self):
        return [os.path.join(self.acquired_tfrecords_dir, loc) for loc in os.listdir(self.acquired_tfrecords_dir)]


    def get_train_records(self):
            return self.initial_train_tfrecords + self.get_acquired_tfrecords()


    def make_predictions(self, shard_locs, initial_size):
        predictor = self.get_latest_model()
        logging.debug('Loaded predictor {}'.format(predictor))
        logging.info('Making and recording predictions')
        logging.info('Using shard_locs {}'.format(shard_locs))
        unlabelled_subjects, samples = active_learning.make_predictions_on_tfrecord(
            shard_locs,
            predictor,
            self.db,
            n_samples=self.n_samples,
            size=initial_size
        )
        # subjects should all be unique, otherwise there's a bug
        id_strs = [subject['id_str'] for subject in unlabelled_subjects]
        assert len(id_strs) == len(set(id_strs)) 
        assert isinstance(unlabelled_subjects, list)
        return unlabelled_subjects, samples


    def get_latest_model(self):
        predictor_loc = active_learning.get_latest_checkpoint_dir(self.estimators_dir)
        logging.info('Loading model from {}'.format(predictor_loc))
        return make_predictions.load_predictor(predictor_loc)


    def record_state(self, subjects, samples, acquisitions):
        metrics.save_iteration_state(self.iteration_dir, subjects, samples, acquisitions)


    def run(self):
        subject_ids, labels, total_votes = get_labels()
        if len(subject_ids) > 0:
            active_learning.add_labels_to_db(subject_ids, labels, total_votes, self.db)
            top_subject_df = active_learning.get_file_loc_df_from_db(self.db, subject_ids)
            active_learning.write_catalog_to_tfrecord_shards(
                top_subject_df,
                db=None,
                img_size=self.initial_size,
                columns_to_save=['id_str', 'label', 'total_votes'],
                save_dir=self.acquired_tfrecords_dir,
                shard_size=4096  # hardcoded, awkward TODO
            )
            # self.acquired_tfrecord = os.path.join(self.acquired_tfrecords_dir, 'acquired_shard.tfrecord')
            # active_learning.add_labelled_subjects_to_tfrecord(self.db, subject_ids, self.acquired_tfrecord, self.initial_size)

        """
        Callable should expect 
        - log dir to train models in
        - list of tfrecord files to train on
        - list of tfrecord files to eval on
        - learning rate to use 
        - epochs to train for
        """  
        self.record_train_records()
        logging.info('Saving to {}'.format(self.estimators_dir))
        self.train_callable(
            self.estimators_dir,
            self.get_train_records(),
            self.eval_tfrecords,
            self.learning_rate,
            self.epochs
        )  # could be docker container to run, save model

        # TODO getting quite messy throughout with lists vs np.ndarray - need to clean up
        # make predictions and save to db, could be docker container
        subjects, samples = self.make_predictions(self.prediction_shards, self.initial_size)

        acquisitions = self.acquisition_func(samples)  # returns list of acquisition values
        self.record_state(subjects, samples, acquisitions)
        logging.debug('{} {} {}'.format(len(acquisitions), len(subjects), len(samples)))

        _, top_acquisition_ids = pick_top_subjects(subjects, acquisitions, self.n_subjects_to_acquire)
        request_labels(top_acquisition_ids)


    def record_train_records(self):
        with open(os.path.join(self.tfrecords_record), 'w') as f:
            json.dump(self.get_train_records(), f)


def request_labels(top_acquisition_ids):
    mock_panoptes.request_labels(top_acquisition_ids)


def get_labels():
    return mock_panoptes.get_labels()


# to be shared for consistency
def pick_top_subjects(subjects, acquisitions, n_subjects_to_acquire):
    args_to_sort = np.argsort(acquisitions)[::-1]  # reverse order, highest to lowest
    top_acquisition_subjects = [subjects[i] for i in args_to_sort][:n_subjects_to_acquire]
    top_acquisition_ids = [subject['id_str'] for subject in top_acquisition_subjects]
    assert len(top_acquisition_ids) == len(set(top_acquisition_ids))  # no duplicates allowed
    return top_acquisition_subjects, top_acquisition_ids
