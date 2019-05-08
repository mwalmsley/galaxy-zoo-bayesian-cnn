import pytest

import os
import shutil
import time
import json
import random

import numpy as np
import pandas as pd

from zoobot.active_learning import iterations
from zoobot.tests.active_learning import conftest


@pytest.fixture(params=[True, False])
def initial_estimator_ckpt(tmpdir, request):
    if request.param:
        return tmpdir.mkdir('some_datetime_ckpt').strpath
    else:
        return None  # no initial ckpt


@pytest.fixture()
def new_iteration(tmpdir, initial_estimator_ckpt, active_config):
        run_dir = active_config.run_dir
        iteration_n = 0
        prediction_shards = ['first_shard.tfrecord', 'second_shard.tfrecord']

        iteration = iterations.Iteration(
            run_dir,
            iteration_n,
            prediction_shards,
            initial_db_loc=active_config.db_loc,
            initial_train_tfrecords=active_config.shards.train_tfrecord_locs(),
            eval_tfrecords=active_config.shards.eval_tfrecord_locs(),
            train_callable=conftest.mock_train_callable,
            acquisition_func=conftest.mock_acquisition_func,
            n_samples=10,  # may need more samples?
            n_subjects_to_acquire=50,
            initial_size=64,
            initial_estimator_ckpt=initial_estimator_ckpt,
            learning_rate=0.001,
            epochs=2
        )

        return iteration


# TODO could maybe refactor into the fixture above
def test_init(tmpdir, initial_estimator_ckpt, active_config):
        run_dir = active_config.run_dir
        iteration_n = 0
        prediction_shards = ['some', 'shards']

        iteration = iterations.Iteration(
            run_dir,
            iteration_n,
            prediction_shards,
            initial_db_loc=active_config.db_loc,
            initial_train_tfrecords=active_config.shards.train_tfrecord_locs(),
            eval_tfrecords=active_config.shards.eval_tfrecord_locs(),
            train_callable=np.random.rand,
            acquisition_func=np.random.rand,
            n_samples=10,  # may need more samples?
            n_subjects_to_acquire=50,
            initial_size=64,
            initial_estimator_ckpt=initial_estimator_ckpt,
            learning_rate=0.001,
            epochs=2
        )

        assert not iteration.get_acquired_tfrecords()
 
        expected_iteration_dir = os.path.join(run_dir, 'iteration_{}'.format(iteration_n))
        assert os.path.isdir(expected_iteration_dir)

        expected_estimators_dir = os.path.join(expected_iteration_dir, 'estimators')
        assert os.path.isdir(expected_estimators_dir)

        expected_metrics_dir = os.path.join(expected_iteration_dir, 'metrics')
        assert os.path.isdir(expected_metrics_dir)

        expected_db_loc = os.path.join(expected_iteration_dir, 'iteration.db')
        assert os.path.exists(expected_db_loc)

        # if initial estimator was provided, it should have been copied into the of 0th iteration subdir
        # TODO
        # if initial_estimator_ckpt is not None:
        #     expected_initial_estimator_copy = os.path.join(expected_estimators_dir, 'some_datetime_ckpt')
        #     assert os.path.isdir(expected_initial_estimator_copy)


def test_get_latest_model(monkeypatch, new_iteration):
    def mock_get_latest_checkpoint(base_dir):
        return 'latest_ckpt'
    monkeypatch.setattr(iterations.active_learning, 'get_latest_checkpoint_dir', mock_get_latest_checkpoint)

    def mock_load_predictor(predictor_loc):
        if predictor_loc == 'latest_ckpt':
            return 'loaded latest model'
        else:
            return 'loaded another model'
    monkeypatch.setattr(iterations.make_predictions, 'load_predictor', mock_load_predictor)

    assert new_iteration.get_latest_model() == 'loaded latest model'


def test_make_predictions(monkeypatch, shard_locs, size, new_iteration):
    def mock_get_latest_model(self):
        return 'loaded latest model'
    monkeypatch.setattr(iterations.Iteration, 'get_latest_model', mock_get_latest_model)

    def mock_make_predictions_on_tfrecord(shard_locs, predictor, db, size, n_samples):
        assert isinstance(shard_locs, list)
        assert predictor == 'loaded latest model'
        assert isinstance(size, int)
        assert isinstance(n_samples, int)
        n_subjects = 112 * len(shard_locs)  # 112 unlabelled subjects per shard
        subjects = [{'matrix': np.random.rand(size, size, 3), 'id_str': str(n)} 
        for n in range(n_subjects)]
        samples = np.random.rand(n_subjects, n_samples)
        return subjects, samples
    monkeypatch.setattr(iterations.active_learning, 'make_predictions_on_tfrecord', mock_make_predictions_on_tfrecord)

    subjects, samples = new_iteration.make_predictions(shard_locs, size)
    assert len(subjects) == 112 * len(shard_locs)
    assert samples.shape == (112 * len(shard_locs), new_iteration.n_samples)


def test_get_train_records(new_iteration, active_config):
    assert new_iteration.get_train_records() == new_iteration.initial_train_tfrecords
    tfrecord_loc = os.path.join(new_iteration.acquired_tfrecords_dir, 'something.tfrecord')
    with open(tfrecord_loc, 'w') as f:
        f.write('a mock tfrecord')
    assert new_iteration.get_train_records() == new_iteration.initial_train_tfrecords + [tfrecord_loc]


def test_record_train_records(new_iteration):
    new_iteration.record_train_records()
    with open(os.path.join(new_iteration.iteration_dir, 'train_records_index.json'), 'r') as f:
        train_records = json.load(f)
    assert train_records == new_iteration.get_train_records()


@pytest.fixture(params=[False, True])
def previously_requested_subjects(request, new_iteration):
    if request.param:  # previous iteration has picked random subjects to be acquired
        return [str(n) for n in range(new_iteration.n_subjects_to_acquire)]
    else:
        return []


def test_run(mocker, monkeypatch, new_iteration, previously_requested_subjects):
    SUBJECTS = [
        {'matrix': np.random.rand(new_iteration.initial_size, new_iteration.initial_size, 3),
        'id_str': str(n)}
        for n in range(1024)]
    
    SUBJECTS_REQUESTED = previously_requested_subjects.copy()  # may recieve random new subjects
    def mock_get_labels():
        selected_ids = SUBJECTS_REQUESTED.copy()
        selected_labels = list(np.random.rand(len(selected_ids)))
        SUBJECTS_REQUESTED.clear()
        assert len(SUBJECTS_REQUESTED) == 0
        total_votes = [40 for n in range(len(selected_labels))]  # always pretend 40 total votes
        return selected_ids, selected_labels, total_votes  
    monkeypatch.setattr(iterations, 'get_labels', mock_get_labels)
    def mock_request_labels(subject_ids):
        SUBJECTS_REQUESTED.extend(subject_ids)
    monkeypatch.setattr(iterations, 'request_labels', mock_request_labels)

    def mock_get_file_loc_df_from_db(db, subject_ids):
        data = {
            'id_str': subject_ids,
            'file_loc': 'somewhere',
            'label': np.random.randint(low=0, high=40, size=len(subject_ids))
        }
        return pd.DataFrame(data=data)
    monkeypatch.setattr(iterations.active_learning, 'get_file_loc_df_from_db', mock_get_file_loc_df_from_db)
    

    def mock_write_catalog_to_tfrecord_shards(df, db, img_size, columns_to_save, save_dir, shard_size):
        # save the subject ids here, pretending to be a tfrecord of those subjects
        assert set(columns_to_save) == {'id_str', 'label', 'total_votes'}
        tfrecord_locs = [os.path.join(save_dir, loc) for loc in ['shard_0.tfrecord', 'shard_1.tfrecord']]
        for loc in tfrecord_locs:
            with open(loc, 'w') as f:
                json.dump([str(x) for x in df['id_str']], f)
    monkeypatch.setattr(iterations.active_learning, 'write_catalog_to_tfrecord_shards', mock_write_catalog_to_tfrecord_shards)


    # def mock_add_labelled_subjects_to_tfrecord(db, subject_ids, tfrecord_loc, size):
    #     assert len(subject_ids) > 0
    #     assert isinstance(subject_ids, list)
    #     assert os.path.exists(os.path.dirname(tfrecord_loc))
    #     assert isinstance(size, int)
    #     # save the subject ids here, pretending to be a tfrecord of those subjects
    #     with open(tfrecord_loc, 'w') as f:
    #         json.dump(subject_ids, f)
    # monkeypatch.setattr(iterations.active_learning, 'add_labelled_subjects_to_tfrecord', mock_add_labelled_subjects_to_tfrecord)


    def mock_add_labels_to_db(subject_ids, labels, total_votes, db):
        assert isinstance(subject_ids, list)
        assert isinstance(subject_ids[0], str)
        assert isinstance(labels, list)
        assert isinstance(labels[0], float)
        assert isinstance(total_votes, list)
        assert isinstance(total_votes[0], int)
        pass  # don't actually bother adding the new labels to the db
        # TODO use a mock and check the call for ids and labels
    monkeypatch.setattr(iterations.active_learning, 'add_labels_to_db', mock_add_labels_to_db)

    def mock_make_predictions(self, prediction_shards, initial_size):
        subjects = SUBJECTS[:len(prediction_shards) * 256]  # imagining there are 256 subjects per shard
        unlabelled_subjects = random.sample(subjects, 212)  # some of which are labelled
        images = np.array([subject['matrix'] for subject in unlabelled_subjects])
        samples = conftest.mock_get_samples_of_images(None, images, n_samples=self.n_samples)
        return unlabelled_subjects, samples
    monkeypatch.setattr(iterations.Iteration, 'make_predictions', mock_make_predictions)

    # TODO check for single call with correct attrs?
    # mocker.Mock('zoobot.active_learning.iterations.Iteration.record_state')

    ####
    new_iteration.run()
    ####

    # TODO review
    # check that the initial ckpt was copied successfully, if one was given
    if new_iteration.initial_estimator_ckpt is not None:
        expected_ckpt_copy = os.path.join(new_iteration.estimators_dir, new_iteration.initial_estimator_ckpt)
        assert os.path.isdir(expected_ckpt_copy)

    # previous iteration may have asked for some subjects - check they were acquired and used
    expected_tfrecords = [
        os.path.join(new_iteration.acquired_tfrecords_dir, 'shard_0.tfrecord'),
        os.path.join(new_iteration.acquired_tfrecords_dir, 'shard_1.tfrecord')
    ]
    if previously_requested_subjects:
        assert new_iteration.get_acquired_tfrecords() == expected_tfrecords
        # mock_write_catalog_to_tfrecord_shards saved json of acquired id_strs to each tfrecord
        subjects_saved_from_earlier_request = json.load(open(expected_tfrecords[0]))
        assert subjects_saved_from_earlier_request == previously_requested_subjects
        assert set(expected_tfrecords).issubset(set(new_iteration.get_train_records()))
    else:
        assert set(expected_tfrecords).intersection(set(new_iteration.get_train_records())) == set()

    # check records were saved TODO
    # assert os.path.exists(os.path.join(new_iteration.metrics_dir, 'some_metrics.txt'))

    # check the correct subjects were requested
    assert SUBJECTS_REQUESTED != previously_requested_subjects
    assert len(SUBJECTS_REQUESTED) == new_iteration.n_subjects_to_acquire
    subjects_acquired = [SUBJECTS[int(id_str)] for id_str in SUBJECTS_REQUESTED]
    subjects_means = np.array([subject['matrix'].mean() for subject in subjects_acquired])
    assert all(subjects_means[1:] < subjects_means[:-1])
