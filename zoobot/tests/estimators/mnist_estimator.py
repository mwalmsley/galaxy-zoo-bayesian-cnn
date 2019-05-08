import logging
import time
import os

import pandas as pd

from zoobot.active_learning import make_shards
from zoobot.tests import TEST_EXAMPLE_DIR

if __name__ == '__main__':

    base_dir = os.path.join(TEST_EXAMPLE_DIR, 'mnist_shards')

    logging.basicConfig(
        filename='{}/make_mnist_shards_{}.log'.format(base_dir, time.time()),
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.DEBUG
    )

    dtypes = {'id_str': str, 'label': int}
    labelled_catalog = pd.read_csv(os.path.join(TEST_EXAMPLE_DIR, 'mnist_catalog_train.csv'), dtype=dtypes)
    unlabelled_catalog = pd.read_csv(os.path.join(TEST_EXAMPLE_DIR, 'mnist_catalog_test.csv'), dtype=dtypes)

    shard_config = make_shards.ShardConfig(base_dir=base_dir) 

    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog)

    shard_config.write()