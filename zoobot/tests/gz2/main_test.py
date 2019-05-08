import pytest

import pandas as pd

from zoobot.get_catalogs.gz2.main import get_labels_and_images
from zoobot.tests.gz2 import get_classifications_test  # TODO make a conftest


@pytest.fixture()
def classifications():
    zoo1 = {
        'dr7objid': 'zoo1',
        'ra': 12.0,
        'dec': -1.0
    }
    zoo1.update(get_classifications_test.example_classification_data())  # TODO make a conftest

    zoo2 = {
        'dr7objid': 'zoo2',
        'ra': 15.0,
        'dec': -1.0
    }
    zoo2.update(get_classifications_test.example_classification_data())  # TODO make a conftest

    return pd.DataFrame([zoo1, zoo2])


@pytest.fixture()
def subject_manifest():
    return pd.DataFrame([
        {
            'location': 'http://s3.amazonaws.com/zoo2/1.jpg',
            'ra': 12.0,
            'dec': -1.0
        },

        {
            'location': 'http://s3.amazonaws.com/zoo2/2.jpg',
            'ra': 15.0,
            'dec': -1.0
        }
    ])


@pytest.fixture()
def png_dir(tmpdir):
    return tmpdir.mkdir('png_dir').strpath


@pytest.fixture()
def output_loc(tmpdir):
    return '{}/output.csv'.format(tmpdir.mkdir('output_dir').strpath)


def test_get_labels_and_images(classifications, subject_manifest, png_dir, output_loc):
    catalog = get_labels_and_images(classifications, subject_manifest, png_dir, output_loc, overwrite=False)
    assert not catalog.empty
