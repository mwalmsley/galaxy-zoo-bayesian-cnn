class GlobalConfig():

    def __init__(self, ec2=False):
        self.ec2 = ec2

        if ec2:
            self.catalog_loc = '/home/ubuntu/data/panoptes_predictions.csv' 
            self.tfrecord_dir = '/home/ubuntu/data/'
        else:
            self.catalog_loc = '/data/galaxy_zoo/decals/panoptes/reduction/output/2018-11-05_panoptes_predictions_with_catalog.csv'
            self.tfrecord_dir = '/data/galaxy_zoo/decals/tfrecords'
