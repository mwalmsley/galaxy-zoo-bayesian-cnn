import pandas as pd


def get_classification_results(published_data_loc, nrows=None):
    """
    Get classification data table from GZ2 published results.
    Get the weighted fraction and raw count data for each relevant question.
    Currently, this is for the questions smooth/featured, edge-on, round/cigar, spiral/not, and spiral count.
    https://data.galaxyzoo.org/ Galaxy Zoo 2 Table 1
    Direct link: http://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz

    Args:
        published_data_loc (str): file location of published GZ classifications table
        nrows (int): num. of rows to return (from head). If None, return all rows.

    Returns:
        (pd.DataFrame) DR7 ID, spiral weighted vote fraction, and vote counts for spiral/not spiral, by subject
    """

    relevant_answers = [
        't01_smooth_or_features_a01_smooth',
        't01_smooth_or_features_a02_features_or_disk',
        't01_smooth_or_features_a03_star_or_artifact',

        't02_edgeon_a04_yes',
        't02_edgeon_a05_no',

        't03_bar_a06_bar',
        't03_bar_a07_no_bar',

        't04_spiral_a08_spiral',
        't04_spiral_a09_no_spiral',

        't07_rounded_a16_completely_round',
        't07_rounded_a17_in_between',
        't07_rounded_a18_cigar_shaped',

        't11_arms_number_a31_1',
        't11_arms_number_a32_2',
        't11_arms_number_a33_3',
        't11_arms_number_a34_4',
        't11_arms_number_a36_more_than_4',
        't11_arms_number_a37_cant_tell'
    ]

    relevant_values = [
        '_weighted_fraction',
        '_count'
    ]

    useful_columns = ['dr7objid', 'ra', 'dec', 'total_classifications', 'total_votes']
    for answer in relevant_answers:
        for value in relevant_values:
            useful_columns.append("".join([answer, value]))

    df = pd.read_csv(published_data_loc, nrows=nrows, usecols=useful_columns)

    return df
