import logging
import os
import shutil
import functools
import itertools
import sqlite3
import time
import json
from collections import namedtuple

import tensorflow as tf
import pandas as pd
import numpy as np

from zoobot.estimators import input_utils
from zoobot.tfrecord import catalog_to_tfrecord, create_tfrecord
from zoobot import shared_utilities
from zoobot.estimators import estimator_params
from zoobot.estimators import run_estimator
from zoobot.estimators import make_predictions
from zoobot.tfrecord import read_tfrecord

try:
    LOCAL_IMAGE_FOLDER = '/Volumes/alpha/gz2/png'
    assert os.path.isdir(LOCAL_IMAGE_FOLDER)
except AssertionError:
    LOCAL_IMAGE_FOLDER = 'data/gz2_shards/gz2/png'
    assert os.path.isdir(LOCAL_IMAGE_FOLDER)



def create_db(catalog, db_loc):
    """Instantiate sqlite database at db_loc with the following tables:
    1. `catalog`: analogy of catalog dataframe, for fits locations and (sometimes) labels
    2. `shardindex`: which shard contains each (serialized) galaxy
    3. `acquisitions`: the latest acquistion function value for each galaxy
    
    Args:
        catalog (pd.DataFrame): with id_col (id string) and fits_loc (image on disk) columns
        db_loc (str): file location to save database
    
    Returns:
        sqlite3.Connection: connection to database as described above. Intended for active learning
    """
    db = sqlite3.connect(db_loc)

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

    add_catalog_to_db(catalog, db)  # does NOT add labels, labels are unknown at this point

    return db


def add_catalog_to_db(df, db):
    """Add id and fits image location info from galaxy catalog to db `catalog` table
    
    Args:
        df (pd.DataFrame): Galaxy catalog with 'id_str' and 'fits_loc' fields
        db (sqlite3.Connection): database with `catalog` table to record df id_col and fits_loc
    """

    catalog_entries = list(df[['id_str', 'file_loc']].itertuples(index=False, name='CatalogEntry'))
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO catalog(id_str, label, total_votes, file_loc) VALUES(?,NULL,NULL,?)''',
        catalog_entries
    )
    db.commit()


def write_catalog_to_tfrecord_shards(df, db, img_size, columns_to_save, save_dir, shard_size=1000):
    """Write galaxy catalog of id_str and file_loc across many tfrecords, and record in db.
    Useful to quickly load images for repeated predictions.

    Args:
        df (pd.DataFrame): Galaxy catalog with 'id_str' and 'fits_loc' fields
        db (sqlite3.Connection): database with `catalog` table to record df id_col and fits_loc
        img_size (int): height/width dimension of image matrix to rescale and save to tfrecords
        columns_to_save (list): Catalog data to save with each subject. Names will match tfrecord.
        save_dir (str): disk directory path into which to save tfrecords
        shard_size (int, optional): Defaults to 1000. Max subjects per shard. Final shard has less.
    """
    assert not df.empty
    assert 'id_str' in columns_to_save
    assert all(column in df.columns.values for column in columns_to_save)

    df = df.copy().sample(frac=1).reset_index(drop=True)  # shuffle
    # split into shards
    shard_n = 0
    n_shards = (len(df) // shard_size) + 1
    df_shards = [df.iloc[n * shard_size:(n + 1) * shard_size] for n in range(n_shards)]

    for shard_n, df_shard in enumerate(df_shards):
        save_loc = os.path.join(save_dir, 's{}_shard_{}.tfrecord'.format(img_size, shard_n))
        catalog_to_tfrecord.write_image_df_to_tfrecord(
            df_shard, 
            save_loc,
            img_size,
            columns_to_save,
            reader=catalog_to_tfrecord.get_reader(df['file_loc']),
            append=False)
        if db is not None:  # explicitly not passing db will skip this step, for e.g. train/test
            add_tfrecord_to_db(save_loc, db, df_shard)  # record ids in db, not labels


def add_tfrecord_to_db(tfrecord_loc, db, df):
    """Update db to record catalog entries as inside a tfrecord.
    Note: does not actually load/verify the tfrecord, simply takes info from catalog
    TODO scan through the record rather than just reading df?
    TODO eventually, consider the catalog being SQL as source-of-truth and csv output

    Args:
        tfrecord_loc (str): disk path of tfrecord
        db (sqlite3.Connection): database with `catalog` table to record df id_col and fits_loc
        df (pd.DataFrame): Galaxy catalog with 'id_str' and 'fits_loc' fields
    """
    shard_index_entries = list(zip(
        df['id_str'].values, 
        [tfrecord_loc for n in range(len(df))]
    ))
    cursor = db.cursor()
    cursor.executemany(
        '''
        INSERT INTO shardindex(id_str, tfrecord) VALUES(?,?)''',
        shard_index_entries
    )
    db.commit()

# should make each shard a comparable size to the available memory, but can iterate over several if needed
def make_predictions_on_tfrecord(tfrecord_locs, model, db, n_samples, size, max_images=10000):
    """Record model predictions (samples) on a list of tfrecords, for each unlabelled galaxy.
    Load in batches to avoid maxing out GPU memory
    See make_predictions_on_tfrecord_batch for more details.
    Uses db to identify which subjects are unlabelled by matching on id_str feature.
    
    Args:
        tfrecord_locs (list): paths to tfrecords to load
        model (function): callable returning [batch, n_samples] predictions
        db (sqlite3): database of galaxies, with id_str and label columns. Not modified.
        n_samples (int): number of MC dropout samples to calculate (i.e. n forward passes)
        size (int): size of saved images
        max_images (int, optional): Defaults to 10000. Load no more than this many images per record
    
    Returns:
        list: unlabelled subjects of form [{id_str: .., matrix: ..}, ..]
        np.ndarray: predictions on those subjects, with shape [n_unlabelled_subjects, n_samples]
    """
    assert isinstance(tfrecord_locs, list)
    # TODO review this once images have been made smaller, may be able to have much bigger batches

    records_per_batch = 2  # best to be >= num. of CPU, for parallel reading. But at 256px, only fits 2 records.
    min_tfrecord = 0
    all_unlabelled_subjects = []  # list of generators for unlabelled subjects
    all_samples = []  # list of np arrays of samples

    while min_tfrecord < len(tfrecord_locs):
        unlabelled_subjects, samples = make_predictions_on_tfrecord_batch(
            tfrecord_locs[min_tfrecord:min_tfrecord + records_per_batch],
            model, db, n_samples, size, max_images=10000)
       
        all_unlabelled_subjects.extend(unlabelled_subjects)
        all_samples.append(samples)

        assert len(unlabelled_subjects) != 0

        min_tfrecord += records_per_batch

    all_samples_arr = np.concatenate(all_samples, axis=0)
    logging.debug('Finished predictions {} on subjects {}'.format(
        all_samples_arr.shape, len(all_unlabelled_subjects))
    )
    return all_unlabelled_subjects, all_samples_arr


def make_predictions_on_tfrecord_batch(tfrecords_batch_locs, model, db, n_samples, size, max_images=10000):
    """Record model predictions (samples) on a list of tfrecords, for each unlabelled galaxy.
    Uses db to identify which subjects are unlabelled by matching on id_str feature.
    Data in tfrecords_batch_locs must fit in memory; otherwise use `make_predictions_on_tfrecord`

    Args:
        tfrecords_batch_locs (list): paths to tfrecords to load
        model (function): callable returning [batch, samples] predictions
        db (sqlite3): database of galaxies, with id_str and label columns. Not modified.
        n_samples (int): number of MC dropout samples to calculate (i.e. n forward passes)
        size (int): size of saved images
        max_images (int, optional): Defaults to 10000. Load no more than this many images per record
    
    Returns:
        list: unlabelled subjects of form [{id_str: .., matrix: ..}, ..]
        np.ndarray: predictions on those subjects, with shape [n_unlabelled_subjects, n_samples]
    """
    batch_images, _, batch_id_str = input_utils.predict_input_func(
                tfrecords_batch_locs,
                n_galaxies=max_images,
                initial_size=size,
                mode='id_str'
            )
    with tf.Session() as sess:
        images, id_str_bytes = sess.run([batch_images, batch_id_str])
        if len(images) == max_images:
            logging.critical(
                'Warning! Shards are larger than memory! Loaded {} images'.format(max_images)
            )

    # tfrecord will have encoded to bytes, need to decode
    logging.debug('Constructing subjects from loaded data')
    subjects = (
        {'matrix': image, 'id_str': id_st.decode('utf-8')} 
        for image, id_st in zip(images, id_str_bytes)
    )
    del images  # free memory

    # exclude subjects with labels in db
    logging.debug('Filtering for unlabelled subjects')
    unlabelled_subjects = [
        subject for subject in subjects
        if subject_is_unlabelled(subject['id_str'], db)
    ]
    logging.debug('Loaded {} unlabelled subjects from {} of size {}'.format(
        len(unlabelled_subjects),
        tfrecords_batch_locs,
         size)
        )
    del subjects  # free memory

    # make predictions on only those subjects
    logging.debug('Extracting images from unlabelled subjects')
    # need to construct array from list: required to have known length
    unlabelled_subject_data = np.array([subject['matrix'] for subject in unlabelled_subjects])
    for subject in unlabelled_subjects:
        del subject['matrix']  # we don't need after this, so long as we skip recording state
    samples = make_predictions.get_samples_of_images(model, unlabelled_subject_data, n_samples)
    return unlabelled_subjects, samples


def save_acquisition_to_db(subject_id, acquisition, db): 
    """Save the acquisition value for the subject to the database
    Warning: will overwrite previous acquisitions for that subject
    
    Args:
        subject_id (str): id string of subject, expected to match db.acquisitions.id_str values
        acquisition (float): latest acquisition value for subject with `subject_id` id string
        db (sqlite3.Connection): database with `acquisitions` table to record acquisition
    """
    assert isinstance(acquisition, float)  # can't be np.float32, else written as bytes
    assert isinstance(subject_id, str)
    cursor = db.cursor()  
    cursor.execute('''  
    INSERT OR REPLACE INTO acquisitions(id_str, acquisition_value)  
                  VALUES(:id_str, :acquisition_value)''',
                  {
                      'id_str': subject_id, 
                      'acquisition_value': acquisition})
    db.commit()


def subject_is_unlabelled(id_str, db):
    """Get the subject ids of up to `n_subjects`:
        1. Without labels (required for later training)
        1. With the highest acquisition values, up to the first `n_subjects`

    Args:
        db (sqlite3.Connection): database with `acquisitions` table to read acquisition
        n_subjects (int, optional): Defaults to 1000. Max subject ids to return.
        shard_loc (str, optional): Defaults to None. Get top subjects from only this shard.

    Raises:
        ValueError: all subjects in `db.catalog` have labels
        IndexError: if no top subjects are found, optionally looking only in shard_loc

    Returns:
        list: ordered list (descending) of subject id strings with highest acquisition values
    """
    cursor = db.cursor()
    # check that at least 1 subject has no label
    cursor.execute(
        '''
        SELECT id_str FROM catalog
        WHERE label IS NULL
        '''
    )
    unlabelled_subject = cursor.fetchone()
    if unlabelled_subject is None:
        logging.critical('WARNING - shard appears to be completely labelled')
        return True

    # find subject in db.catalog
    cursor.execute(
        '''
        SELECT id_str, label
        FROM catalog
        WHERE id_str IS (:id_str)
        ''',
        (id_str,)
    )
    matching_subjects = cursor.fetchall()  # sqlite3 may autoconvert id_str to int at this step
    if len(matching_subjects) < 1:
        raise ValueError('Subject not found: {}'.format(id_str))
    if len(matching_subjects) > 1:
        raise ValueError('Duplicate subject in db: {}'.format(id_str))
    return matching_subjects[0][1] is None  # not sure why bool( ... ) tests passed, possibly upside down

# bad name, actually gets all table columns
def get_file_loc_df_from_db(db, subject_ids):
    """
    Look up the file_loc and label in db for the subjects with ids in `subject_ids`
    df will include the image_loc stored at `db.catalog.file_loc`
    df will include the label stored under `db.catalog.label`
    Useful for creating additional training tfrecords during active learning
    
    Args:
        db (sqlite3.Connection): database with `db.catalog` table to get subject image and label
        subject_ids (list): of subject ids matching `db.catalog.id_str` values
    
    Raises:
        IndexError: No matches found to an id in `subject_ids` within db.catalog.id_str
        ValueError: Subject with an id in `subject_ids` has null label (code error in this func.)
    """
    assert len(subject_ids) > 0
    cursor = db.cursor()

    rows = []
    for subject_id in subject_ids:
        assert isinstance(subject_id, str)
        logging.debug(subject_id)
        cursor.execute(
            '''
            SELECT id_str, label, total_votes, file_loc FROM catalog
            WHERE id_str = (:id_str)
            ''',
            (subject_id,)
        )
        # namedtuple would allow better type checking
        subject = cursor.fetchone()  # TODO ensure distinct
        if subject is None:
            raise IndexError('Fatal: top ids not found in catalog or label missing!')

        if subject[1] is None:
            raise ValueError('Fatal: {} missing label in db!'.format(subject_id))
        assert subject[1] != b'\x00\x00\x00\x00\x00\x00\x00\x00'  # i.e. np.int64 write error
        rows.append({
            'id_str': str(subject[0]),  # db cursor casts to int-like string to int...
            'label': int(subject[1]),
            'total_votes': int(subject[2]),
            'file_loc': get_relative_loc(str(subject[3]))
        })

    top_subject_df = pd.DataFrame(data=rows)
    assert len(top_subject_df) == len(subject_ids)
    return top_subject_df


def add_labelled_subjects_to_tfrecord(db, subject_ids, tfrecord_loc, size):
    assert not os.path.isfile(tfrecord_loc)  # this will overwrite, tfrecord can't append
    df = get_file_loc_df_from_db(db, subject_ids)
    catalog_to_tfrecord.write_image_df_to_tfrecord(
        df,
        tfrecord_loc,
        size,
        columns_to_save=['id_str', 'label', 'total_votes'],
        reader=catalog_to_tfrecord.get_reader(df['file_loc']))



def get_relative_loc(loc):
    fname = os.path.basename(loc)
    subdir = os.path.basename(os.path.dirname(loc))
    return os.path.join(LOCAL_IMAGE_FOLDER, subdir, fname)



def get_latest_checkpoint_dir(base_dir):
    saved_models = [x for x in os.listdir(base_dir) if x.startswith('15')]  # subfolders with timestamps
    saved_models.sort(reverse=True)  # sort by name i.e. timestamp, early timestamp first
    return os.path.join(base_dir, saved_models[0])  # the subfolder with the most recent time


def add_labels_to_db(subject_ids, labels, total_votes, db):
    cursor = db.cursor()

    for subject_n in range(len(subject_ids)):
        label = labels[subject_n]
        total_votes_val = total_votes[subject_n]
        subject_id = subject_ids[subject_n]

        # np.int64 is wrongly written as byte string e.g. b'\x00...',  b'\x01...'
        label = int(label)
        total_votes_val = int(total_votes_val)
        assert isinstance(label, int)
        assert isinstance(label, int)
        assert isinstance(subject_id, str)
        cursor.execute(
            '''
            UPDATE catalog
            SET label = (:label), total_votes = (:total_votes)
            WHERE id_str = (:subject_id)
            ''',
            {
                'label': label,
                'subject_id': subject_id,
                'total_votes': total_votes_val
            }
        )
        db.commit()

        if subject_n == 0:  # careful check on first write only, for speed
            # check labels really have been added, and not as byte string
            cursor.execute(
                '''
                SELECT label, total_votes FROM catalog
                WHERE id_str = (:subject_id)
                LIMIT 1
                ''',
                (subject_id,)
            )
            row = cursor.fetchone()
            assert row is not None
            retrieved_label = row[0]
            assert retrieved_label == label
            retrieved_total_votes = row[1]
            assert retrieved_total_votes == total_votes_val


def get_all_shard_locs(db):
    cursor = db.cursor()
    cursor.execute(
        '''
        SELECT DISTINCT tfrecord FROM shardindex
        
        ORDER BY tfrecord ASC
        '''
    )
    return [row[0] for row in cursor.fetchall()]  # list of shard locs
