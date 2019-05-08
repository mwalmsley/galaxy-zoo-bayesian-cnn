"""
Not actually a test - downloads MNIST in fits form. Maybe a useful benchmark.
"""
import os

import tensorflow as tf
from astropy.io import fits
import pandas as pd
import numpy as np

from zoobot.tests import TEST_EXAMPLE_DIR

def download_to_fits():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    # x of shape (subject, height, width)
    # y of shape (label)

    # set up for active learning test
    fits_dir = os.path.join(TEST_EXAMPLE_DIR, 'mnist_fits')
    if not os.path.isdir(fits_dir):
        os.mkdir(fits_dir)

    train_fits_locs = []
    train_fits_names = []
    for im_n, im in enumerate(x_train):  # iterates over first dimension, subjects
        fits_name = 'train_im_' + str(im_n) + '.fits'
        fits_loc = os.path.join(fits_dir, fits_name)
        hdu = fits.PrimaryHDU(np.concatenate([im, im, im], axis=0))
        hdu.writeto(fits_loc, overwrite=True)
        train_fits_locs.append(fits_loc)
        train_fits_names.append(fits_name)

    train_catalog = pd.DataFrame(
        data={
            'id_str': [str(n) for n in range(len(y_train))],
            'label': y_train, 
            'fits_loc': train_fits_locs, 
            'fits_loc_relative': train_fits_names}
    )

    test_fits_locs = []
    test_fits_names = []
    for im_n, im in enumerate(x_test):  # iterates over first dimension, subjects
        fits_name = 'test_im_' + str(im_n) + '.fits'
        fits_loc = os.path.join(fits_dir, fits_name)
        hdu = fits.PrimaryHDU(np.concatenate([im, im, im], axis=0))
        hdu.writeto(fits_loc, overwrite=True)
        test_fits_locs.append(fits_loc)
        test_fits_names.append(fits_name)

    test_catalog = pd.DataFrame(
        data={
            'id_str': [str(n) for n in range(len(y_test))],
            'label': y_test, 
            'fits_loc': test_fits_locs, 
            'fits_loc_relative': test_fits_names}
    )

    train_catalog.to_csv(os.path.join(TEST_EXAMPLE_DIR, 'mnist_catalog_train.csv'), index=False)
    test_catalog.to_csv(os.path.join(TEST_EXAMPLE_DIR, 'mnist_catalog_test.csv'), index=False)


if __name__ == '__main__':
    download_to_fits()
