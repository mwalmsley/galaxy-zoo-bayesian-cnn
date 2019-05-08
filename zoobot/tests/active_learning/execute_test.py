import pytest

import os
import json
import time
import copy

import numpy as np

from zoobot.tests.active_learning import conftest
from zoobot.tfrecord import read_tfrecord
from zoobot.active_learning import active_learning, make_shards, execute, iterations


def test_run(active_config, tmpdir, monkeypatch, catalog_random_images, tfrecord_dir, mocker):
    # TODO need to test we're using the estimators we expect, needs refactoring first
    # catalog_random_images is a required arg because the fits files must actually exist

    # retrieve these shard locs for use in .run()
    mocker.patch('zoobot.active_learning.execute.active_learning.get_all_shard_locs')
    execute.active_learning.get_all_shard_locs.return_value = ['shard_loc_a', 'shard_loc_b', 'shard_loc_c', 'shard_loc_d']

    # mock out Iteration
    mocker.patch('zoobot.active_learning.execute.iterations.Iteration', autospec=True)
    mock_iteration = execute.iterations.Iteration.return_value  # shorthand reference for the instantiated class
    # set db_loc and estimator_dirs attributes
    type(mock_iteration).db_loc = mocker.PropertyMock(side_effect=['first_db_loc', 'second_db_loc', 'third_db_loc'])  
    type(mock_iteration).estimators_dir = mocker.PropertyMock(side_effect=['first_est_dir', 'second_est_dir', 'third_est_dir'])  
    # set train_records
    mock_iteration.get_train_records.side_effect = ['first_records', 'second_records', 'third_records']

    # TODO mock iterations as a whole, piecemeal moved to iterations
    active_config.run(
        conftest.mock_acquisition_func, 
        conftest.mock_acquisition_func)

    # created three iterations
    assert execute.iterations.Iteration.call_count == 3
    # called with:
    # print(execute.iterations.Iteration.mock_calls)
    calls = execute.iterations.Iteration.mock_calls
    assert len(calls) == 9  # 3 inits, 3 .get_train_records, 3 .run
    init_calls = [call for call in calls if call[0] == '']
    assert len(init_calls) == 3

    # for each call, check that args are updated correctly over iterations
    first_init_call_args = init_calls[0][2]
    assert first_init_call_args['prediction_shards'] == [
        os.path.join(active_config.shards.shard_dir, shard_loc) for shard_loc in ['shard_loc_a', 'shard_loc_b']
        ]
    assert first_init_call_args['initial_estimator_ckpt'] == active_config.initial_estimator_ckpt
    assert first_init_call_args['initial_db_loc'] == active_config.db_loc
    assert first_init_call_args['initial_train_tfrecords'] == [os.path.join(active_config.shards.train_dir, loc) for loc in os.listdir(active_config.shards.train_dir) if loc.endswith('.tfrecord')]
    assert first_init_call_args['eval_tfrecords'] == [os.path.join(active_config.shards.eval_dir, loc) for loc in os.listdir(active_config.shards.eval_dir) if loc.endswith('.tfrecord')]

    second_init_call_args = init_calls[1][2]
    assert second_init_call_args['prediction_shards'] == [
        os.path.join(active_config.shards.shard_dir, shard_loc) for shard_loc in ['shard_loc_c', 'shard_loc_d']
        ]
    assert second_init_call_args['initial_estimator_ckpt'] == 'first_est_dir'
    assert second_init_call_args['initial_db_loc'] == 'first_db_loc'
    assert second_init_call_args['initial_train_tfrecords'] == 'first_records'

    third_init_call_args = init_calls[2][2]
    assert third_init_call_args['prediction_shards'] == [
        os.path.join(active_config.shards.shard_dir, shard_loc) for shard_loc in ['shard_loc_a', 'shard_loc_b']
        ]
    assert third_init_call_args['initial_estimator_ckpt'] == 'second_est_dir'
    assert third_init_call_args['initial_db_loc'] == 'second_db_loc'
    assert third_init_call_args['initial_train_tfrecords'] == 'second_records'

    # ran three times (mock iteration itself is always used, not created afresh)
    assert mock_iteration.run.call_count == 3


@pytest.fixture(params=[{'warm_start': True}, {'warm_start': False}])
def train_callable_params(request):
    return execute.TrainCallableParams(
        initial_size = 128,
        final_size = 64,
        warm_start = request.param['warm_start'],
        eval_tfrecord_loc='some_eval.tfrecord',
        test=False
    )

def test_get_train_callable(mocker, train_callable_params):
    train_callable = execute.get_train_callable(train_callable_params)
    assert callable(train_callable)

    from zoobot.estimators import run_estimator
    mocker.patch('zoobot.estimators.run_estimator.run_estimator')
    log_dir = 'log_dir'
    train_records = 'train_records'
    eval_records = 'eval_records'
    train_callable(log_dir, train_records, eval_records, learning_rate=0.001, epochs=2)
    run_estimator.run_estimator.assert_called_once()
    config = run_estimator.run_estimator.mock_calls[0][1][0]  # first call, positional args, first arg
    assert config.log_dir == log_dir
    assert config.train_config.tfrecord_loc == train_records
    assert config.eval_config.tfrecord_loc == train_callable_params.eval_tfrecord_loc
    assert config.warm_start == train_callable_params.warm_start



@pytest.fixture(params=[True, False])
def baseline(request):
    return request.param

def test_get_acquisition_func(baseline, samples):
    acq_func = execute.get_acquisition_func(baseline)
    if baseline:
        assert acq_func == execute.mock_acquisition_func
    else:
        assert acq_func == execute.acquisition_utils.mutual_info_acquisition_func
    

    # # verify the folders appear as expected
    # for iteration_n in range(active_config.iterations):
    #     # copied to iterations_test.py
    #     # separate dir for each iteration
    #     iteration_dir = os.path.join(active_config.run_dir, 'iteration_{}'.format(iteration_n))
    #     assert os.path.isdir(iteration_dir)
    #     # which has a subdir recording the estimators
    #     estimators_dir = os.path.join(iteration_dir, 'estimators')
    #     assert os.path.isdir(estimators_dir)
    #     # if initial estimator was provided, it should have been copied into the of 0th iteration subdir
    #     if iteration_n == 0:
    #         if active_config.initial_estimator_ckpt is not None:
    #             assert os.path.isdir(os.path.join(estimators_dir, active_config.initial_estimator_ckpt))
    #     else:
    #         if active_config.warm_start:
    #             # should have copied the latest estimator from the previous iteration
    #             latest_previous_estimators_dir = os.path.join(active_config.run_dir, 'iteration_{}'.format(iteration_n - 1), 'estimators')
    #             latest_previous_estimator = active_learning.get_latest_checkpoint_dir(latest_previous_estimators_dir)  # TODO double-check this func!
    #             assert os.path.isdir(os.path.join(estimators_dir, os.path.split(latest_previous_estimator)[-1]))
    
    # # read back the training tfrecords and verify they are sorted by order of mean
    # with open(active_config.train_records_index_loc, 'r') as f:
    #     acquired_shards = json.load(f)[1:]  # includes the initial shard, which is unsorted
    
    # matrix_means = []
    # for shard in acquired_shards:
    #     subjects = read_tfrecord.load_examples_from_tfrecord(
    #         [shard], 
    #         read_tfrecord.matrix_label_id_feature_spec(active_config.shards.initial_size, active_config.shards.channels)
    #     )
    #     shard_matrix_means = np.array([x['matrix'].mean() for x in subjects])

    #     # check that images have been saved to shards in monotonically decreasing order...
    #     assert all(shard_matrix_means[1:] < shard_matrix_means[:-1])
    #     matrix_means.append(shard_matrix_means)
    # # ...but not across all shards, since we only predict on some shards at a time
    # all_means = np.concatenate(matrix_means)
    # assert not all(all_means[1:] < all_means[:-1])



    # def mock_load_predictor(loc):
    #     # assumes run is configured for 3 iterations in total
    #     with open(os.path.join(loc, 'dummy_model.txt'), 'r') as f:
    #         training_records = json.load(f)
    #     if len(training_records) == 1:
    #         assert 'initial_train' in training_records[0]
    #         return 'initial train only'
    #     if len(training_records) == 2:
    #         return 'one acquired record'
    #     if len(training_records) == 3:
    #         return 'two acquired records'
    #     else:
    #         raise ValueError('More training records than expected!')
    # monkeypatch.setattr(active_learning.make_predictions, 'load_predictor', mock_load_predictor)

    # def mock_get_samples_of_subjects(model, subjects, n_samples):
    #     # only give the correct samples if you've trained on two acquired records
    #     # overrides conftest
    #     assert isinstance(subjects, list)
    #     example_subject = subjects[0]
    #     assert isinstance(example_subject, dict)
    #     assert 'matrix' in example_subject.keys()
    #     assert isinstance(n_samples, int)

    #     response = []
    #     for subject in subjects:
    #         if model == 'two acquired records':
    #             response.append([np.mean(subject['matrix'])] * n_samples)
    #         else:
    #             response.append(np.random.rand(n_samples))
    #     return np.array(response)
    # monkeypatch.setattr(active_learning.make_predictions, 'get_samples_of_subjects', mock_get_samples_of_subjects)