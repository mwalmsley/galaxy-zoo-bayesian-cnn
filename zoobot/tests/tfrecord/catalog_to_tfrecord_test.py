import os

import pytest
import pandas as pd

from zoobot.tfrecord import catalog_to_tfrecord, read_tfrecord
from zoobot.tests import TEST_EXAMPLE_DIR


@pytest.mark.xfail
def test_get_reader():
    raise NotImplementedError


def test_write_catalog_to_train_test_records(catalog, tfrecord_dir, label_col, size, columns_to_save):
    catalog_to_tfrecord.write_catalog_to_train_test_tfrecords(
        catalog,
        tfrecord_dir + '/train.tfrecords',
        tfrecord_dir + '/test.tfrecords',
        size,
        columns_to_save,
        reader=catalog_to_tfrecord.load_decals_as_pil)


def test_write_image_df_to_tfrecord(tfrecord_dir, size, channels):
    df = pd.DataFrame(
        data=[
            {
                'id_str': 'some_id', 
                'fits_loc': os.path.join(TEST_EXAMPLE_DIR, 'example_a.fits')
            },
            {
                'id_str': 'some_other_id',
                'fits_loc': os.path.join(TEST_EXAMPLE_DIR, 'example_b.fits')
                }
        ]
    )
    columns_to_save = ['id_str']
    tfrecord_loc = os.path.join(tfrecord_dir, 'some.tfrecord')
    catalog_to_tfrecord.write_image_df_to_tfrecord(
        df,
        tfrecord_loc,
        size, 
        columns_to_save, 
        reader=catalog_to_tfrecord.load_decals_as_pil,
        append=False
    )
    subjects = read_tfrecord.load_examples_from_tfrecord(
        [tfrecord_loc], 
        read_tfrecord.matrix_id_feature_spec(size, channels)
    )
    assert len(subjects) == len(df)
    assert subjects[0]['id_str'] == 'some_id'.encode('utf-8')
    assert subjects[1]['id_str'] == 'some_other_id'.encode('utf-8')
