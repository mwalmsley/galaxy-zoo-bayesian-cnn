import pytest

import os
import pandas as pd

from zoobot.get_catalogs.gz2.get_classifications import get_classification_results


@pytest.fixture()
def published_data_loc(tmpdir):
    return tmpdir.mkdir('catalog_dir').strpath + '/published_data.csv'


@pytest.fixture()
def example_classification_data():
    return {

        # weighted fraction values (random)
        't01_smooth_or_features_a01_smooth_weighted_fraction': 0.1,
        't01_smooth_or_features_a02_features_or_disk_weighted_fraction': 0.8,
        't01_smooth_or_features_a03_star_or_artifact_weighted_fraction': 0.1,

        't02_edgeon_a04_yes_weighted_fraction': 0.2,
        't02_edgeon_a05_no_weighted_fraction': 0.2,  # do weighted fractions have any rules e.g. sum to 1? TODO check

        't04_spiral_a08_spiral_weighted_fraction': 0.7,
        't04_spiral_a09_no_spiral_weighted_fraction': 0.3,

        't07_rounded_a16_completely_round_weighted_fraction': 0.1,
        't07_rounded_a17_in_between_weighted_fraction': 0.1,
        't07_rounded_a18_cigar_shaped_weighted_fraction': 0.1,

        't11_arms_number_a31_1_weighted_fraction': 0.1,
        't11_arms_number_a32_2_weighted_fraction': 0.4,
        't11_arms_number_a33_3_weighted_fraction': 0.2,
        't11_arms_number_a34_4_weighted_fraction': 0.1,
        't11_arms_number_a36_more_than_4_weighted_fraction': 0.05,
        't11_arms_number_a37_cant_tell_weighted_fraction': 0.05,


        # raw count values (random)
        't01_smooth_or_features_a01_smooth_count': 12,
        't01_smooth_or_features_a02_features_or_disk_count': 14,
        't01_smooth_or_features_a03_star_or_artifact_count': 2,

        't02_edgeon_a04_yes_count': 12,
        't02_edgeon_a05_no_count': 14,

        't04_spiral_a08_spiral_count': 12,
        't04_spiral_a09_no_spiral_count': 14,

        't07_rounded_a16_completely_round_count': 12,
        't07_rounded_a17_in_between_count': 14,
        't07_rounded_a18_cigar_shaped_count': 2,

        't11_arms_number_a31_1_count': 8,
        't11_arms_number_a32_2_count': 6,
        't11_arms_number_a33_3_count': 12,
        't11_arms_number_a34_4_count': 7,
        't11_arms_number_a36_more_than_4_count': 2,
        't11_arms_number_a37_cant_tell_count': 2,
    }


@pytest.fixture()
def published_data(example_classification_data):
    zoo1 = {
            'dr7objid': 'zoo1',
            'ra': 12.0,
            'dec': -1.0
        }
    zoo1.update(example_classification_data)

    zoo2 = {
            'dr7objid': 'zoo2',
            'ra': 15.0,
            'dec': -1.0
        }
    zoo2.update(example_classification_data)

    df = pd.DataFrame([zoo1, zoo2])
    df['total_votes'] = 2
    df['total_classifications'] = 3
    return df


def test_get_classification_results(published_data, published_data_loc):
    assert not os.path.exists(published_data_loc)
    published_data.to_csv(published_data_loc)
    catalog = get_classification_results(published_data_loc)
    print(catalog['dr7objid'])
    print(published_data['dr7objid'])
    for column in published_data.columns.values:
        assert all(catalog[column] == published_data[column])
