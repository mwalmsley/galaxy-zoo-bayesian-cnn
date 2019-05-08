
# given a catalog with fits_loc_relative, and target dir
# work out the s3 location
# copy to target dir
# save catalog with new accurate fits loc

"""
Temporary script to take only fits files with panoptes labels, and put them in a separate folder
Useful to prototype maching learning
Adds column with fits loc relative to an unknown root, and optionally saves to new file

Initial catalog comes from `panoptes_mock_predictions.csv`, on s3://mikewalmsley
New catalog (i.e. with updated `fits_loc`, `fits_loc_relative`) optionally saved
"""
import os
from subprocess import call

import pandas as pd

from zoobot.tests import TEST_EXAMPLE_DIR

# must already have fits_loc_relative column

# base_dir = '/home/ec2-user'
# base_dir = '/home/ubuntu/root/zoobot/data'
# predictions_loc = os.path.join(base_dir, 'panoptes_predictions_selected.csv')

# s3_dir = 's3://galaxy-zoo/decals/fits_native'

# target_dir = '/users/mikewalmsley/pretend_ec2_root/fits_native'
# target_dir = '/home/ec2-user/fits_native'
# target_dir = '/home/ubuntu/fits_native'

predictions_loc = '/home/ubuntu/root/zoobot/data/panoptes_predictions_selected.csv'
df = pd.read_csv(
    predictions_loc, 
    dtype={'id_loc': str, 'label': float},
    # nrows=10
)

# df['fits_loc_s3'] = s3_dir + df['fits_loc_relative']

target_dir = '/home/ubuntu/root/zoobot/data/fits_native'
# if not os.path.isdir(target_dir):
    # os.mkdir(target_dir)
df['fits_loc'] = target_dir + df['fits_loc_relative']

df.to_csv('panoptes_predictions_updated.csv', index=False)
