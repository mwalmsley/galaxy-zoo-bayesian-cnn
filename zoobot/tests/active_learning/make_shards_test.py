import os
import json
import shutil
import sqlite3

import pytest

from zoobot.tests import TEST_EXAMPLE_DIR
from zoobot.active_learning import make_shards
from zoobot.tests.active_learning.active_learning_test import verify_db_matches_shards, verify_catalog_matches_shards


def test_prepare_shards(shard_config, labelled_catalog, unlabelled_catalog):
    # put fits in the right place
    shard_config.prepare_shards(labelled_catalog, unlabelled_catalog)
    assert shard_config.ready()


def test_write_and_load(shard_config, tmpdir):
    config_save_loc = os.path.join(tmpdir.mkdir('config_save_dir'), 'saved_config.json')
    shard_config.config_save_loc = config_save_loc
    shard_config.write()
    loaded_config = make_shards.load_shard_config(config_save_loc)
    # TODO check equality of fields


def test_make_database_and_shards(unlabelled_catalog, db_loc, size, channels, tfrecord_dir):
    assert os.path.isdir(tfrecord_dir)
    make_shards.make_database_and_shards(unlabelled_catalog, db_loc, size, tfrecord_dir, shard_size=25)
    db = sqlite3.connect(db_loc)
    # verify_db_matches_catalog(catalog, db)
    verify_db_matches_shards(db, size, channels)
    verify_catalog_matches_shards(unlabelled_catalog, db, size, channels)
