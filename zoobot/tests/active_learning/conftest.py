import pytest

import copy
import os
import time
import json

import numpy as np
import pandas as pd
from PIL import Image

from astropy.io import fits
from zoobot.active_learning import make_shards, execute


@pytest.fixture()
def n_subjects():
    return 256


@pytest.fixture()
def id_strs(n_subjects):
    return [str(n) for n in range(n_subjects)]


@pytest.fixture(params=['png_loc', 'fits_loc'])
def file_col(request):
    return request.param


@pytest.fixture
def catalog_random_images(size, channels, n_subjects, id_strs, fits_native_dir, png_native_dir, file_col):
    """Construct labelled/unlabelled catalogs for testing active learning"""
    
    
    assert os.path.isdir(fits_native_dir)
    assert os.path.isdir(png_native_dir)
    matrices = np.random.rand(n_subjects, size, size, channels)
    some_feature = np.random.rand(n_subjects)

    catalog = pd.DataFrame(data={'id_str': id_strs, 'some_feature': some_feature})

    if file_col == 'fits_loc':
        relative_fits_locs = ['random_{}.fits'.format(n) for n in range(n_subjects)]
        fits_locs = list(map(lambda rel_loc: os.path.join(fits_native_dir, rel_loc), relative_fits_locs))
        for matrix, loc in zip(matrices, fits_locs):  # write to fits
            hdu = fits.PrimaryHDU(matrix)
            hdu.writeto(loc, overwrite=True)
            assert os.path.isfile(loc)
        catalog['file_loc'] = fits_locs

    if file_col == 'png_loc':
        relative_png_locs = ['random_{}.png'.format(n) for n in range(n_subjects)]
        png_locs = list(map(lambda rel_loc: os.path.join(png_native_dir, rel_loc), relative_png_locs))
        for matrix, loc in zip(matrices, png_locs):  # write to fits
            rgb_matrix = (matrix * 256).astype(np.uint8)
            Image.fromarray(rgb_matrix, mode='RGB').save(loc)
            assert os.path.isfile(loc)
        catalog['file_loc'] = png_locs

    return catalog


# due to conftest.py, catalog fits_loc is absolute and points to fits_native_dir
@pytest.fixture()
def labelled_catalog(catalog_random_images):
    catalog = catalog_random_images.copy()
    catalog['id_str'] = catalog_random_images['id_str'] + '_from_labelled'  # must be unique
    catalog['label'] = np.random.rand(len(catalog))
    catalog['total_votes'] = np.random.randint(low=1, high=41, size=len(catalog))
    return catalog


@pytest.fixture()
def unlabelled_catalog(catalog_random_images):
    catalog = catalog_random_images.copy()
    catalog['id_str'] = catalog_random_images['id_str'] + '_from_unlabelled'  # must be unique
    return catalog


@pytest.fixture()
def shard_config(tmpdir, size, channels):
    config = make_shards.ShardConfig(
        shard_dir=tmpdir.mkdir('base_dir').strpath,
        shard_size=128,
        inital_size=size,
        final_size=size,
        channels=channels)
    return config


@pytest.fixture()
def shard_config_ready(shard_config, labelled_catalog, unlabelled_catalog):
    config = copy.copy(shard_config)
    config.prepare_shards(labelled_catalog, unlabelled_catalog, train_test_fraction=0.8)
    assert config.ready()
    return config


@pytest.fixture(
    params=[
        {
            'initial_estimator_ckpt': 'use_predictor',
            'warm_start': True
        },
        {
            'initial_estimator_ckpt': None,
            'warm_start': False
        }
    ])
def active_config(shard_config_ready, tmpdir, predictor_model_loc, request):

    warm_start = request.param['warm_start']

    # work around using fixture as param by having param toggle whether the fixture is used
    if request.param['initial_estimator_ckpt'] == 'use_predictor':
        initial_estimator_ckpt = predictor_model_loc
    else:
        initial_estimator_ckpt = None
    
    config = execute.ActiveConfig(
        shard_config_ready, 
        run_dir=tmpdir.mkdir('run_dir').strpath,
        n_iterations=3,  # 1st is only the initial cycle
        shards_per_iter=2,
        subjects_per_iter=10,
        initial_estimator_ckpt=initial_estimator_ckpt
        )

    assert os.path.isdir(config.run_dir)  # permanent directory for dvc control
    assert os.path.exists(config.db_loc)

    assert config.ready()
    return config


@pytest.fixture()
def db_loc(tmpdir):
    return os.path.join(tmpdir.mkdir('db_dir').strpath, 'db_is_here.db')


def mock_acquisition_func(samples):
    assert isinstance(samples, np.ndarray)
    assert len(samples.shape) == 2
    return samples.mean(axis=1)  # sort by mean prediction (here, mean of each subject)


def mock_train_callable(estimators_dir, train_tfrecord_locs, eval_tfrecord_locs, learning_rate, epochs):
    # pretend to save a model in subdirectory of estimator_dir
    assert os.path.isdir(estimators_dir)
    subdir_loc = os.path.join(estimators_dir, str(time.time()))
    os.mkdir(subdir_loc)
    with open(os.path.join(subdir_loc, 'dummy_model.txt'), 'w') as f:
        json.dump(train_tfrecord_locs, f)


@pytest.fixture()
def acquisition():
    return np.random.rand()


@pytest.fixture()
def acquisitions(subjects):
    return list(np.random.rand(len(subjects)))


@pytest.fixture()
def images(n_subjects, size):
    return np.random.rand(n_subjects, size, size, 3)


@pytest.fixture()
def subjects(size, images):
    return [{'matrix': images[n], 'id_str': 'id_' + str(n)} for n in range(len(images))]


def mock_get_samples_of_images(model, images, n_samples):
    # predict the mean of image batch, 10 times
    assert isinstance(images, np.ndarray)
    assert len(images.shape) == 4
    assert isinstance(n_samples, int)
    response = [[np.mean(images[n])] * 10 for n in range(len(images))]
    return np.array(response)


@pytest.fixture()
def samples(images):
    return mock_get_samples_of_images(model=None, images=images, n_samples=10)


@pytest.fixture()
def estimators_dir(tmpdir):
    base_dir = tmpdir.mkdir('estimators').strpath
    checkpoint_dirs = ['157001', '157002', '157003']
    for directory in checkpoint_dirs:
        os.mkdir(os.path.join(base_dir, directory))
    files = ['checkpoint', 'graph.pbtxt', 'events.out.tfevents.1545', 'model.ckpt.2505.index', 'model.ckpt.2505.meta']
    for file_name in files:
        with open(os.path.join(base_dir, file_name), 'w') as f:
            f.write('Dummy file')
    return base_dir
