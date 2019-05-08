"""
Temporary script to take only fits files with panoptes labels, and put them in a separate folder
Useful to prototype maching learning
Adds column with fits loc relative to an unknown root, and optionally saves to new file

Initial catalog comes from `panoptes_mock_predictions.csv`, on s3://mikewalmsley
New catalog (i.e. with updated `fits_loc`, `fits_loc_relative`) optionally saved
"""
import os
import logging
from subprocess import call
import argparse

import pandas as pd
from tqdm import tqdm

from zoobot.tests import TEST_EXAMPLE_DIR

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Copy labelled fits')
    parser.add_argument('--old_catalog_loc', dest='old_catalog_loc', type=str,
                    help='Path to csv catalog of Panoptes labels and fits_loc')
    parser.add_argument('--new_catalog_loc', dest='new_catalog_loc', type=str,
                    help='Path to csv catalog of Panoptes labels and fits_loc')
    parser.add_argument('--old_fits_dir', dest='old_fits_dir', type=str,
                    help='Oldlocation of fits files')
    parser.add_argument('--new_fits_dir', dest='new_fits_dir', type=str,
                    help='Directory into which to place shard directory')

    args = parser.parse_args()

    df_loc = args.old_catalog_loc
    assert os.path.exists(df_loc)

    df = pd.read_csv(
        df_loc, 
        dtype={'id_str': str, 'label': float},
        # nrows=10
    )

    old_fits_dir = args.old_fits_dir
    new_fits_dir = args.new_fits_dir

    df['fits_loc_old'] = df['fits_loc']

    old_dir_chars = len(old_fits_dir)
    df['fits_loc_relative'] = df['fits_loc_old'].apply(lambda x: x[old_dir_chars+1:]) # no initial \
    df['fits_loc'] = df['fits_loc_relative'].apply(lambda x: os.path.join(new_fits_dir, x))

    logging.info('Old directory: {}'.format(df.iloc[0]['fits_loc_old']))
    logging.info('New directory: {}'.format(df.iloc[0]['fits_loc']))
    logging.info('Relative directoy: {}'.format(df.iloc[0]['fits_loc_relative']))

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # copy file to new native directory
        target_dir = os.path.dirname(row['fits_loc'])
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)  # careful, will make directories recursively
        assert row['fits_loc_old']
        call(["cp", row['fits_loc_old'], row['fits_loc']])


    del df['fits_loc_old']

    df.to_csv(args.new_catalog_loc, index=False)
