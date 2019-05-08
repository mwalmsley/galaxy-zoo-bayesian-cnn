import argparse
import os
import shutil
import logging
import json
import time

import numpy as np
import pandas as pd
import git

from shared_astro_utils import matching_utils

from zoobot.tfrecord import catalog_to_tfrecord
from zoobot.estimators import run_estimator, make_predictions
from zoobot.active_learning import active_learning, default_estimator_params
from zoobot.tests import TEST_EXAMPLE_DIR


class ShardConfig():
    """
    Assumes that you have:
    - a directory of fits files  (e.g. `fits_native`)
    - a catalog of files, with file locations under the column 'fits_loc' (relative to repo root)

    Checks that catalog paths match real fits files
    Creates unlabelled shards and single shard of labelled subjects
    Creates sqlite database describing what's in those shards

    JSON serializable for later loading
    """

    def __init__(
        self,
        shard_dir,  # to hold a new folder, named after the shard config 
        size=256,  # IMPORTANT
        shard_size=4096,
        **overflow_args  # TODO review removing this
        ):
        """
        Args:
            shard_dir (str): directory into which to save shards
            size (int, optional): Defaults to 128. Resolution to save fits to tfrecord
            final_size (int, optional): Defaults to 64. Resolution to load from tfrecord into model
            shard_size (int, optional): Defaults to 4096. Galaxies per shard.
        """
        self.size = size
        self.shard_size = shard_size
        self.shard_dir = shard_dir

        self.channels = 3  # save 3-band image to tfrecord. Augmented later by model input func.

        self.db_loc = os.path.join(self.shard_dir, 'static_shard_db.db')  # record shard contents

        # paths for fixed tfrecords for initial training and (permanent) evaluation
        self.train_dir = os.path.join(self.shard_dir, 'train_shards') 
        self.eval_dir = os.path.join(self.shard_dir, 'eval_shards')

        # paths for catalogs. Used to look up .fits locations during active learning.
        self.labelled_catalog_loc = os.path.join(self.shard_dir, 'labelled_catalog.csv')
        self.unlabelled_catalog_loc = os.path.join(self.shard_dir, 'unlabelled_catalog.csv')

        self.config_save_loc = os.path.join(self.shard_dir, 'shard_config.json')


    def train_tfrecord_locs(self):
        return [os.path.join(self.train_dir, loc) for loc in os.listdir(self.train_dir)
            if loc.endswith('.tfrecord')]


    def eval_tfrecord_locs(self):
        return [os.path.join(self.eval_dir, loc) for loc in os.listdir(self.eval_dir)
            if loc.endswith('.tfrecord')]


    def prepare_shards(self, labelled_catalog, unlabelled_catalog, train_test_fraction):
        """[summary]
        
        Args:
            labelled_catalog (pd.DataFrame): labelled galaxies, including fits_loc column
            unlabelled_catalog (pd.DataFrame): unlabelled galaxies, including fits_loc column
            train_test_fraction (float): fraction of labelled catalog to use as training data
        """
        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir)  # always fresh
        os.mkdir(self.shard_dir)
        os.mkdir(self.train_dir)
        os.mkdir(self.eval_dir)

        # check that file paths resolve correctly
        assert all(os.path.isfile(path) for path in labelled_catalog['file_loc'])
        assert all(os.path.isfile(path) for path in unlabelled_catalog['file_loc'])

        # assume the catalog is true, don't modify halfway through
        logging.info('\nLabelled subjects: {}'.format(len(labelled_catalog)))
        logging.info('Unlabelled subjects: {}'.format(len(unlabelled_catalog)))
        labelled_catalog.to_csv(self.labelled_catalog_loc)
        unlabelled_catalog.to_csv(self.unlabelled_catalog_loc)

        # save train/test split into training and eval shards
        train_df, eval_df = catalog_to_tfrecord.split_df(labelled_catalog, train_test_fraction=train_test_fraction)
        logging.info('\nTraining subjects: {}'.format(len(train_df)))
        logging.info('Eval subjects: {}'.format(len(eval_df)))
        if len(train_df) < len(eval_df):
            print('More eval subjects than training subjects - is this intended?')
        train_df.to_csv(os.path.join(self.train_dir, 'train_df.csv'))
        eval_df.to_csv(os.path.join(self.eval_dir, 'eval_df.csv'))
        for (df, save_dir) in [(train_df, self.train_dir), (eval_df, self.eval_dir)]:
            active_learning.write_catalog_to_tfrecord_shards(
                df,
                db=None,
                img_size=self.size,
                columns_to_save=['id_str', 'label', 'total_votes'],
                save_dir=save_dir,
                shard_size=self.shard_size
            )

        make_database_and_shards(
            unlabelled_catalog, 
            self.db_loc, 
            self.size, 
            self.shard_dir, 
            self.shard_size)

        assert self.ready()

        # serialized for later/logs
        self.write()


    def ready(self):
        assert os.path.isdir(self.shard_dir)
        assert os.path.isdir(self.train_dir)
        assert os.path.isdir(self.eval_dir)
        assert os.path.isfile(self.db_loc)
        assert os.path.isfile(self.labelled_catalog_loc)
        assert os.path.isfile(self.unlabelled_catalog_loc)
        return True


    # TODO move to shared utilities
    def to_dict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        # TODO use dict comprehension
        return dict(
            [(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys]
            )

    
    def write(self):
        with open(self.config_save_loc, 'w+') as f:
            json.dump(self.to_dict(), f)


def load_shard_config(shard_config_loc):
    with open(shard_config_loc, 'r') as f:
        shard_config_dict = json.load(f)
    return ShardConfig(**shard_config_dict)


def make_database_and_shards(catalog, db_loc, size, shard_dir, shard_size):
    if os.path.exists(db_loc):
        os.remove(db_loc)
    # set up db and shards using unknown catalog data
    db = active_learning.create_db(catalog, db_loc)
    columns_to_save = ['id_str']
    active_learning.write_catalog_to_tfrecord_shards(
        catalog,
        db,
        size,
        columns_to_save,
        shard_dir,
        shard_size
    )


if __name__ == '__main__':

    # Write catalog to shards (tfrecords as catalog chunks) for use in active learning
    parser = argparse.ArgumentParser(description='Make shards')
    parser.add_argument('--shard_dir', dest='shard_dir', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--catalog_loc', dest='catalog_loc', type=str,
                    help='Path to csv catalog of Panoptes labels and file_loc, for shards')
    args = parser.parse_args()

    log_loc = 'make_shards_{}.log'.format(time.time())
    logging.basicConfig(
        filename=log_loc,
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )


    # needs update
    usecols = [
        't01_smooth_or_features_a01_smooth_count',
        't01_smooth_or_features_a01_smooth_weighted_fraction',  # annoyingly, I only saved the weighted fractions. Should be quite similar, I hope.
        't01_smooth_or_features_a02_features_or_disk_count',
        't01_smooth_or_features_a03_star_or_artifact_count',
        't04_spiral_a08_spiral_count',
        't04_spiral_a09_no_spiral_count',
        't03_bar_a06_bar_count',
        't03_bar_a07_no_bar_count',
        'id',
        'ra',
        'dec',
        'png_loc',
        'png_ready',
        'sample'
    ]

    # only exists if zoobot/get_catalogs/gz2 instructions have been followed
    unshuffled_catalog = pd.read_csv(args.catalog_loc,
                        usecols=usecols,
                        nrows=None)

    # THIS IS CRUCIAL. GZ catalog is not properly shuffled, and featured-ness changes systematically
    catalog = unshuffled_catalog.sample(len(unshuffled_catalog)).reset_index()

    catalog = catalog[catalog['sample'] == 'original']

    # previous catalog didn't include total classifications/votes, so we'll need to work around that for now
    catalog['smooth-or-featured_total-votes'] = catalog['t01_smooth_or_features_a01_smooth_count'] + catalog['t01_smooth_or_features_a02_features_or_disk_count'] + catalog['t01_smooth_or_features_a03_star_or_artifact_count']
    catalog['bar_total-votes'] = catalog['t03_bar_a06_bar_count'] + catalog['t03_bar_a07_no_bar_count']

    # for consistency
    catalog['id_str'] = catalog['id'].astype(str)

    # SMOOTH MODE
    # artificially enforce as simple test case
    catalog = catalog[catalog['smooth-or-featured_total-votes'] > 36]
    catalog['label'] = catalog['t01_smooth_or_features_a01_smooth_count']
    catalog['total_votes'] = catalog['smooth-or-featured_total-votes']
    # BAR MODE
    # catalog = catalog[catalog['bar_total-votes'] > 10]  # filter to at least a bit featured
    # catalog['label'] = catalog['t03_bar_a06_bar_count']
    # catalog['total_votes'] = catalog['bar_total-votes']
    # SPIRAL MODE
    # catalog = catalog[catalog['spiral_total-votes'] > 10]  # filter to at least a bit featured
    # catalog['total_votes'] = catalog['spiral_total-votes']
    # catalog['label'] = catalog['t04_spiral_a08_spiral_count']


    # local
    # catalog['file_loc'] = catalog['png_loc']
    # ec2
    catalog['file_loc'] = catalog['png_loc'].apply(lambda x: 'data/gz2_shards/' + x.lstrip('/Volumes/alpha'))  # active learning will load from png by default
    assert all(loc for loc in catalog['file_loc'])
    del catalog['png_loc']  # else may load this by default

    print(catalog['file_loc'].sample(5))
    # catalog['id_str'] = catalog['subject_id'].astype(str)  # useful to crossmatch later

    # with basic split, we do 80% train/test split
    # here, use 80% also but with 5*1024 pool held back as oracle (should be big enough)
    # select 1024 new training images
    # verify model is nearly as good as basic split (only missing about 4k images)
    # verify that can add thAese images to training pool without breaking everything!
    # may need to disable interleave, and instead make dataset of joined tfrecords (starting with new ones?)

    # of 18k (exactly 40 votes), initial train on 6k, eval on 3k, and pool the remaining 9k
    # split catalog and pretend most is unlabelled
    # real mode:
    train_size = 256
    eval_size = 2500
    labelled_size = train_size + eval_size
    # labelled_size = len(catalog) - 5000
    # test mode:
    # catalog = catalog[:13000]
    # labelled_size = 6000

    labelled_catalog = catalog[:labelled_size]  # for training and eval. Could do basic split on these!
    unlabelled_catalog = catalog[labelled_size:]  # for pool

    # nair_catalog = pd.read_csv('data/nair_sdss_catalog_interpreted.csv')
    # train on galaxies that are NOT in Nair (minus 500 for eval). Later, will evaluate on galaxies in Nair.
    # Note that these will be in unlabelled shards, not eval shard.
    # unlabelled_catalog, labelled_catalog = matching_utils.match_galaxies_to_catalog_pandas(catalog, nair_catalog)
    # print('Not in Nair: {}'.format(len(labelled_catalog)))
    # print('In Nair" {}'.format(len(unlabelled_catalog)))

    del unlabelled_catalog['label']

    # in memory for now, but will be serialized for later/logs
    shard_config = ShardConfig(shard_dir=args.shard_dir)  
    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog,
        train_test_fraction=(len(labelled_catalog) - eval_size)/len(labelled_catalog))  # always eval on random 2500 galaxies
    # must be able to end here, snapshot created and ready to go (hopefully)

    # temporary hacks for mocking panoptes
    # do this last as shard_dir is wiped and remade when making shards
    # save catalog for mock_panoptes.py to return (now added to git)
    catalog[['id_str', 'total_votes', 'label']].to_csv(os.path.join(args.shard_dir, 'oracle.csv'), index=False)
    catalog.to_csv(os.path.join(args.shard_dir, 'full_catalog.csv'), index=False)

    # finally, tidy up by moving the log into the shard directory
    # could not be create here because shard directory did not exist at start of script
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    shutil.move(log_loc, os.path.join(args.shard_dir, '{}.log'.format(sha)))
