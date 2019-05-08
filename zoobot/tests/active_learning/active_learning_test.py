import pytest

import logging
import os
import random
import hashlib
import sqlite3
import time
import json

import numpy as np
import tensorflow as tf
import pandas as pd
from astropy.io import fits

from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.tfrecord import create_tfrecord, read_tfrecord
from zoobot.estimators.estimator_params import default_four_layer_architecture, default_params
from zoobot.active_learning import active_learning
from zoobot.estimators import make_predictions
from zoobot.tests.active_learning import conftest


@pytest.fixture()
def unknown_subject(size, channels):
    return {
        'matrix': np.random.rand(size, size, channels),
        'id_str': hashlib.sha256(b'some_id_bytes').hexdigest()
    }


@pytest.fixture()
def known_subject(known_subject):
    known_subject = unknown_subject.copy()
    known_subject['label'] = np.random.randint(1)
    return known_subject


@pytest.fixture()
def test_dir(tmpdir):
    return tmpdir.strpath


@pytest.fixture()
def empty_shard_db():
    db = sqlite3.connect(':memory:')

    cursor = db.cursor()

    cursor.execute(
        '''
        CREATE TABLE catalog(
            id_str STRING PRIMARY KEY,
            label INT DEFAULT NULL,
            total_votes INT DEFAULT NULL,
            file_loc STRING)
        '''
    )
    db.commit()

    cursor.execute(
        '''
        CREATE TABLE shardindex(
            id_str STRING PRIMARY KEY,
            tfrecord TEXT)
        '''
    )
    db.commit()

    cursor.execute(
        '''
        CREATE TABLE acquisitions(
            id_str STRING PRIMARY KEY,
            acquisition_value FLOAT)
        '''
    )
    db.commit()
    return db


@pytest.fixture(params=['fits', 'png'])
def file_loc_of_image(request):
    if request.param == 'fits':
        loc = os.path.join(TEST_EXAMPLE_DIR, 'example_a.fits')
    elif request.param == 'png':
        loc = os.path.join(TEST_EXAMPLE_DIR, 'example_a.png')
    else:
        raise ValueError(request.param)
    assert os.path.isfile(loc)
    return loc


@pytest.fixture()
def filled_shard_db(empty_shard_db, file_loc_of_image):
    db = empty_shard_db
    cursor = db.cursor()

    cursor.execute(
        '''
        INSERT INTO catalog(id_str, file_loc)
                  VALUES(:id_str, :file_loc)
        ''',
        {
            'id_str': 'some_hash',
            'file_loc': file_loc_of_image,
            'label': 1,  # already labelled!,
            'total_votes': 1
        }
    )
    db.commit()
    cursor.execute(
        '''
        INSERT INTO acquisitions(id_str, acquisition_value)
                  VALUES(:id_str, :acquisition_value)
        ''',
        {
            'id_str': 'some_hash',
            'acquisition_value': 0.9
        }
    )
    db.commit()
    cursor.execute(
        '''
        INSERT INTO shardindex(id_str, tfrecord)
                  VALUES(:id_str, :tfrecord)
        ''',
        {
            'id_str': 'some_hash',
            'tfrecord': 'tfrecord_a'
        }
    )
    db.commit()

    cursor.execute(
        '''
        INSERT INTO catalog(id_str, file_loc)
                  VALUES(:id_str, :file_loc)
        ''',
        {
            'id_str': 'some_other_hash',
            'file_loc': file_loc_of_image
        }
    )
    cursor.execute(
        '''
        INSERT INTO acquisitions(id_str, acquisition_value)
                  VALUES(:id_str, :acquisition_value)
        ''',
        {
            'id_str': 'some_other_hash',
            'acquisition_value': 0.3
        }
    )
    db.commit()
    cursor.execute(
        '''
        INSERT INTO shardindex(id_str, tfrecord)
                  VALUES(:id_str, :tfrecord)
        ''',
        {
            'id_str': 'some_other_hash',
            'tfrecord': 'tfrecord_b'
        }
    )
    db.commit()

    cursor.execute(
        '''
        INSERT INTO catalog(id_str, file_loc)
                  VALUES(:id_str, :file_loc)
        ''',
        {
            'id_str': 'yet_another_hash',
            'file_loc': file_loc_of_image
        }
    )
    cursor.execute(
        '''
        INSERT INTO acquisitions(id_str, acquisition_value)
                  VALUES(:id_str, :acquisition_value)
        ''',
        {
            'id_str': 'yet_another_hash',
            'acquisition_value': 0.1
        }
    )
    db.commit()
    cursor.execute(
        '''
        INSERT INTO shardindex(id_str, tfrecord)
                  VALUES(:id_str, :tfrecord)
        ''',
        {
            'id_str': 'yet_another_hash',
            'tfrecord': 'tfrecord_a'  # same as first entry, should be selected if filter on rec a
        }
    )
    db.commit()
    return db


@pytest.fixture()
def filled_shard_db_with_labels(filled_shard_db):
    db = filled_shard_db
    cursor = db.cursor()
    rows = [
        {
            'id_str': 'some_hash',
            'label': 1,
            'total_votes': 1
         },
         {
            'id_str': 'some_other_hash',
            'label': 1,
            'total_votes': 1
        },
        {
            'id_str': 'yet_another_hash',
            'label': 0,
            'total_votes': 1
        }
    ]
    for row in rows:
        cursor.execute(
            '''
            UPDATE catalog SET label = ?, total_votes = ?
            WHERE id_str = ?
            ''',
            (row['label'], row['total_votes'], row['id_str'])
        )
        db.commit()
    cursor.execute(
        '''
        SELECT id_str, label, total_votes FROM catalog
        '''
    )
    # trust but verify
    catalog = cursor.fetchall()
    assert catalog == [('some_hash', 1, 1), ('some_other_hash', 1, 1), ('yet_another_hash', 0, 1)]
    return db


def test_write_catalog_to_tfrecord_shards(unlabelled_catalog, empty_shard_db, size, channels, tfrecord_dir):
    columns_to_save = ['id_str', 'some_feature']
    active_learning.write_catalog_to_tfrecord_shards(
        unlabelled_catalog,
        empty_shard_db,
        size,
        columns_to_save,
        tfrecord_dir,
        shard_size=15)
    # verify_db_matches_catalog(catalog, empty_shard_db, 'id_str', label_col)
    verify_db_matches_shards(empty_shard_db, size, channels)
    verify_catalog_matches_shards(unlabelled_catalog, empty_shard_db, size, channels)


def verify_db_matches_catalog(labelled_catalog, db):
     # db should contain the catalog in 'catalog' table
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT id_str, file_loc FROM catalog
        '''
    )
    catalog_entries = cursor.fetchall()
    for entry in catalog_entries:
        recovered_id = str(entry[0])
        recovered_loc = entry[1]
        expected_loc = labelled_catalog[labelled_catalog['id_str'] == recovered_id].squeeze()['file_loc']
        assert recovered_loc == expected_loc


def load_shardindex(db):
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT id_str, tfrecord FROM shardindex
        '''
    )
    shardindex_entries = cursor.fetchall()
    shardindex_data = []
    for entry in shardindex_entries:
        shardindex_data.append({
            'id_str': str(entry[0]),  # TODO shardindex id is str, loaded from byte string
            'tfrecord': entry[1]
        })
    return pd.DataFrame(data=shardindex_data)


def verify_db_matches_shards(db, size, channels):
    # db should contain file locs in 'shardindex' table
    # tfrecords should have been written with the right files
    shardindex = load_shardindex(db)
    tfrecord_locs = shardindex['tfrecord'].unique()
    for tfrecord_loc in tfrecord_locs:
        expected_shard_ids = set(shardindex[shardindex['tfrecord'] == tfrecord_loc]['id_str'].unique())
        examples = read_tfrecord.load_examples_from_tfrecord(
            [tfrecord_loc], 
            read_tfrecord.matrix_id_feature_spec(size, channels)
        )
        actual_shard_ids = set([example['id_str'].decode() for example in examples])
        assert expected_shard_ids == actual_shard_ids


def verify_catalog_matches_shards(unlabelled_catalog, db, size, channels):
    from collections import Counter
    # TODO why do I need to import Counter here?! Surely it should be script scoped...
    shardindex = load_shardindex(db)
    tfrecord_locs = shardindex['tfrecord'].unique()
    # check that every catalog id is in exactly one shard
    assert not any(unlabelled_catalog['id_str'].duplicated())  # catalog must be unique to start with
    catalog_ids = Counter(unlabelled_catalog['id_str'])  # all 1's
    shard_ids = Counter()

    for tfrecord_loc in tfrecord_locs:
        examples = read_tfrecord.load_examples_from_tfrecord(
            [tfrecord_loc],
            read_tfrecord.matrix_id_feature_spec(size, channels)
        )
        ids_in_shard = [x['id_str'].decode() for x in examples]
        assert len(ids_in_shard) == len(set(ids_in_shard))  # must be unique within shard
        shard_ids = Counter(ids_in_shard) + shard_ids

    assert catalog_ids == shard_ids



def test_add_tfrecord_to_db(tfrecord_matrix_ints_loc, empty_shard_db, unlabelled_catalog):  # bad loc
    active_learning.add_tfrecord_to_db(tfrecord_matrix_ints_loc, empty_shard_db, unlabelled_catalog)
    cursor = empty_shard_db.cursor()
    cursor.execute(
        '''
        SELECT id_str, tfrecord FROM shardindex
        '''
    )
    saved_subjects = cursor.fetchall()
    for n, subject in enumerate(saved_subjects):
        assert str(subject[0]) == unlabelled_catalog.iloc[n]['id_str']  # strange string casting when read back
        assert subject[1] == tfrecord_matrix_ints_loc


def test_save_acquisition_to_db(unknown_subject, acquisition, empty_shard_db):
    active_learning.save_acquisition_to_db(unknown_subject['id_str'], acquisition, empty_shard_db)
    cursor = empty_shard_db.cursor()
    cursor.execute(
        '''
        SELECT id_str, acquisition_value FROM acquisitions
        '''
    )
    saved_subject = cursor.fetchone()
    assert saved_subject[0] == unknown_subject['id_str']
    assert np.isclose(saved_subject[1], acquisition)


def test_make_predictions_on_tfrecord(monkeypatch, tfrecord_matrix_id_loc, filled_shard_db, size):
    
    monkeypatch.setattr(
        active_learning.make_predictions,
        'get_samples_of_images',
        conftest.mock_get_samples_of_images
    )

    MAX_ID_STR = 64
    def mock_subject_is_unlabelled(id_str, db):
        return int(id_str) > MAX_ID_STR
    monkeypatch.setattr(
        active_learning,
        'subject_is_unlabelled',
        mock_subject_is_unlabelled
    )

    n_samples = 10
    model = None  # avoid this via mocking, above
    subjects, samples = active_learning.make_predictions_on_tfrecord(
        [tfrecord_matrix_id_loc],
        model,  
        filled_shard_db,
        n_samples=n_samples,
        size=size,
        max_images=20000
    )
    assert samples.shape == (len(subjects), n_samples)
    assert [int(subject['id_str']) > MAX_ID_STR for subject in subjects]  # no labelled subjects 


def test_subject_is_unlabelled(filled_shard_db):
    id_strs = ['some_hash', 'some_other_hash', 'yet_another_hash']
    labelled_ids = [active_learning.subject_is_unlabelled(id_str, filled_shard_db) for id_str in id_strs]
    labelled_ids == [True, False, False]

    with pytest.raises(ValueError):
        active_learning.subject_is_unlabelled('missing_subject', filled_shard_db)


def test_add_labelled_subjects_to_tfrecord(monkeypatch, filled_shard_db_with_labels, tfrecord_dir, size, channels):
    # e.g. root image directory is tests, with images in subdirectory test_examples
    monkeypatch.setattr(active_learning, 'LOCAL_IMAGE_FOLDER', os.path.dirname(TEST_EXAMPLE_DIR))
    tfrecord_loc = os.path.join(tfrecord_dir, 'active_train.tfrecord')
    subject_ids = ['some_hash', 'yet_another_hash']
    active_learning.add_labelled_subjects_to_tfrecord(filled_shard_db_with_labels, subject_ids, tfrecord_loc, size)

    # open up the new record and check
    subjects = read_tfrecord.load_examples_from_tfrecord([tfrecord_loc], read_tfrecord.matrix_id_feature_spec(size, channels))
    # should NOT be read back shuffled!
    assert subjects[0]['id_str'] == 'some_hash'.encode('utf-8')  # tfrecord saves as bytes
    assert subjects[1]['id_str'] == 'yet_another_hash'.encode('utf-8')  #tfrecord saves as bytes


def test_add_labels_to_db(filled_shard_db):
    subjects = [
        {
            'id_str': 'some_hash',
            'label': 0,
            'total_votes': 0
        },
        {
            'id_str': 'yet_another_hash',
            'label': 1,
            'total_votes': 1
        }
    ]
    subject_ids = [x['id_str'] for x in subjects]
    labels = [x['label'] for x in subjects]
    total_votes = [x['total_votes'] for x in subjects]
    active_learning.add_labels_to_db(subject_ids, labels, total_votes, filled_shard_db)
    # read db, check labels match
    cursor = filled_shard_db.cursor()
    for subject in subjects:
        cursor.execute(
            '''
            SELECT label FROM catalog
            WHERE id_str = (:id_str)
            ''',
            (subject['id_str'],)
        )
        results = list(cursor.fetchall())
        assert len(results) == 1
        assert results[0][0] == subject['label']


def test_get_all_shard_locs(filled_shard_db):
    assert active_learning.get_all_shard_locs(filled_shard_db) == ['tfrecord_a', 'tfrecord_b']

def test_get_latest_checkpoint_dir(estimators_dir):
    latest_ckpt = active_learning.get_latest_checkpoint_dir(estimators_dir)
    assert os.path.split(latest_ckpt)[-1] == '157003'
