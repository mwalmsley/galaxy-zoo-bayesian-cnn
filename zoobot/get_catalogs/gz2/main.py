
import pandas as pd

from zoobot.get_catalogs.gz2.get_classifications import get_classification_results
from zoobot.get_catalogs.gz2.download_gz_from_aws import download_png_threaded
from zoobot.shared_utilities import match_galaxies_to_catalog_pandas


def get_labels_and_images(classifications, subject_manifest, png_dir, output_loc, overwrite):

    catalog, _ = match_galaxies_to_catalog_pandas(classifications, subject_manifest)
    assert len(catalog) > 0

    catalog_with_png_locs = download_png_threaded(catalog, png_dir, overwrite)
    catalog_with_png_locs.to_csv(output_loc, index=None)
    return catalog_with_png_locs  # useful to return for unit tests


if __name__ == '__main__':

    nrows = None
    overwrite = False

    catalog_dir = '/data/galaxy_zoo/gz2/catalogs'  # shared folder, for convenience
    published_data_loc = '{}/gz2_hart16.csv'.format(catalog_dir)  # volunteer labels
    subject_manifest_loc = '{}/galaxyzoo2_sandor.csv'.format(catalog_dir)  # subjects on AWS
    labels_loc = '{}/basic_regression_labels.csv'.format(catalog_dir)  # will place catalog of file list and labels here
    output_loc = '{}/gz2_classifications_and_subjects.csv'.format(catalog_dir)  # includes png_ready etc.
    png_dir = '/Volumes/alpha/gz2/png'  # will place downloaded png here (5GB or so)

    # just loads specific columns
    # classifications = get_classification_results(published_data_loc, nrows=nrows)
    classifications = pd.read_csv(published_data_loc)
    print('Published subjects with labels: {}'.format(len(classifications)))

    subject_manifest = pd.read_csv(subject_manifest_loc, nrows=nrows)
    print('AWS subjects from Sandor: {}'.format(len(subject_manifest)))

    get_labels_and_images(classifications, subject_manifest, png_dir, output_loc, overwrite=overwrite)
