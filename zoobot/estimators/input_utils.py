import copy

import numpy as np
import pandas as pd
import tensorflow as tf

from zoobot.tfrecord.tfrecord_io import load_dataset
from zoobot.tfrecord.read_tfrecord import matrix_feature_spec, matrix_id_feature_spec, matrix_label_feature_spec, matrix_label_counts_feature_spec


class InputConfig():

    def __init__(
            self,
            name,
            tfrecord_loc,
            label_col,
            initial_size,
            final_size,
            channels,
            batch_size,
            shuffle,
            repeat,
            stratify,
            stratify_probs,
            regression=True,
            geometric_augmentation=True,
            shift_range=None,  # not implemented
            zoom=(1., 1.1),
            fill_mode=None,  # not implemented
            photographic_augmentation=True,
            max_brightness_delta=0.05,
            contrast_range=(0.95, 1.05),
            noisy_labels=True,
            greyscale=True,
            zoom_central=True
    ):

        self.name = name
        self.tfrecord_loc = tfrecord_loc
        self.label_col = label_col
        self.initial_size = initial_size
        self.final_size = final_size
        self.channels = channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.stratify = stratify
        self.stratify_probs = stratify_probs
        self.regression = regression

        if regression:
            assert not stratify
        if stratify:
            assert not regression

        self.geometric_augmentation = geometric_augmentation  # use geometric augmentations
        self.shift_range = shift_range  # not yet implemented
        self.zoom = zoom
        self.fill_mode = fill_mode  # not yet implemented, 'pad' or 'zoom'

        self.photographic_augmentation = photographic_augmentation
        self.max_brightness_delta = max_brightness_delta
        self.contrast_range = contrast_range

        self.noisy_labels = noisy_labels

        self.greyscale = greyscale
        self.zoom_central = zoom_central

    def set_stratify_probs_from_csv(self, csv_loc):
        subject_df = pd.read_csv(csv_loc)
        self.stratify_probs = [1. - subject_df[self.label_col].mean(), subject_df[self.label_col].mean()]


    # TODO move to shared utilities
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])

    def copy(self):
        return copy.deepcopy(self)


def get_input(config):
    """
    Load tfrecord as dataset. Stratify and transform_3d images as directed. Batch with queues for Estimator input.
    Args:
        config (InputConfig): Configuration object defining how 'get_input' should function  # TODO consider active class

    Returns:
        (dict) of form {'x': greyscale image batch}, as Tensor of shape [batch, size, size, 1]}
        (Tensor) categorical labels for each image
    """
    with tf.name_scope('input_{}'.format(config.name)):
        # batch_images, batch_labels = load_batches_with_labels(config)
        batch_images, batch_labels, batch_counts = load_batches_with_counts(config)
        
        preprocessed_batch_images = preprocess_batch(batch_images, config)
        # tf.shape is important to record the dynamic shape, rather than static shape
        if config.greyscale:
            assert preprocessed_batch_images['x'].shape[3] == 1
        else:
            assert preprocessed_batch_images['x'].shape[3] == 3

        joint_batch_labels = tf.stack([batch_labels, batch_counts], axis=1)
        return preprocessed_batch_images, joint_batch_labels


def make_labels_noisy(labels):
    # NEW - noisy labels
    # would like to get a volunteer response (1. or 0.), but awkward to write everything x40
    # intead, sample a label based on the observed vote fraction.
    # the expectation will be the same, and we'll run this many times.
    uniform_sample = tf.distributions.Uniform(low=1e-6, high=1.0 - 1e-6).sample(tf.shape(labels))
    # 0. if label < sample, or 1. if label > sample
    return tf.round(tf.constant(0.5) + labels - uniform_sample)


def get_batch(tfrecord_loc, feature_spec, batch_size, shuffle, repeat):
        dataset = load_dataset(tfrecord_loc, feature_spec, shuffle=shuffle)
        if shuffle:
            dataset = dataset.shuffle(5000)  # should be > len of each tfrecord
        if repeat:
            dataset = dataset.repeat(-1)  # careful, don't repeat forever for eval
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(3)  # ensure that a batch is always ready to go
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def get_images_from_batch(batch, size, channels, summary=False):

        batch_data = batch['matrix']
        batch_images = tf.reshape(
            batch_data,
            [-1, size, size, channels])  # may not get full batch at end of dataset
        assert len(batch_images.shape) == 4
        tf.summary.image('a_original', batch_images)
        # tf.summary.scalar('batch_size', tf.shape(preprocessed_batch_images['x'])[0])
        return batch_images


def get_labels_from_batch(batch, noisy_labels):
    labels = batch['label']
    if noisy_labels:
        sampled_labels = make_labels_noisy(labels)
    else:
        sampled_labels = labels
    # tf.summary.histogram('raw_labels', labels)
    # tf.summary.histogram('loaded_labels', sampled_labels)
    # tf.summary.scalar('mean_label', tf.reduce_mean(labels))
    return sampled_labels


def get_counts_from_batch(batch):
    return batch['total_votes']


def load_batches_with_labels(config):
    """
    Get batches of images and labels from tfrecord according to instructions in config
    # TODO make a single stratify parameter that expects list of floats - required to run properly anyway
    # use e.g. dataset.map(func, num_parallel_calls=n) rather than map_fn - but stratify??
    Does NOT apply augmentations or further brightness tweaks

    Args:
        config (InputConfig): instructions to load and preprocess the image and label data

    Returns:
        (tf.Tensor, tf.Tensor)
    """
    with tf.name_scope('load_batches_{}'.format(config.name)):
        feature_spec = matrix_label_feature_spec(config.initial_size, config.channels, float_label=config.regression)

        batch = get_batch(config.tfrecord_loc, feature_spec, config.batch_size, config.shuffle, config.repeat)

        batch_images = get_images_from_batch(batch, config.initial_size, config.channels, summary=True)
        batch_labels = get_labels_from_batch(batch, config.noisy_labels)
        return batch_images, batch_labels


def load_batches_with_counts(config):
    with tf.name_scope('load_batches_{}'.format(config.name)):
        feature_spec = matrix_label_counts_feature_spec()
        batch = get_batch(config.tfrecord_loc, feature_spec, config.batch_size, config.shuffle, config.repeat)
        batch_images = get_images_from_batch(batch, config.initial_size, config.channels, summary=True)
        batch_labels = get_labels_from_batch(batch, config.noisy_labels)
        batch_counts = get_counts_from_batch(batch)

        return batch_images, batch_labels, batch_counts


def load_batches_without_labels(config):
    # does not fetch id - unclear if this is important
    feature_spec = matrix_feature_spec(config.initial_size, config.channels)
    batch = get_batch(config.tfrecord_loc, feature_spec, config.batch_size, config.shuffle, config.repeat)
    return get_images_from_batch(batch, config.initial_size, config.channels, summary=True)


def load_batches_with_id_str(config):
    # does not fetch id - unclear if this is important
    feature_spec = matrix_id_feature_spec(config.initial_size, config.channels)
    batch = get_batch(config.tfrecord_loc, feature_spec, config.batch_size, config.shuffle, config.repeat)
    return get_images_from_batch(batch, config.initial_size, config.channels, summary=True), batch['id_str']


def preprocess_batch(batch_images, config):
    with tf.name_scope('preprocess'):

        assert len(batch_images.shape) == 4
        assert batch_images.shape[3] == 3  # should still have 3 channels at this point

        if config.greyscale:
            # new channel dimension of 1
            channel_images = tf.reduce_mean(batch_images, axis=3, keepdims=True)
            assert channel_images.shape[1] == config.initial_size
            assert channel_images.shape[2] == config.initial_size
            assert channel_images.shape[3] == 1
            tf.summary.image('b_greyscale', channel_images)
        else:
            channel_images = tf.identity(batch_images)

        augmented_images = augment_images(channel_images, config)
        assert augmented_images.shape[1] == config.final_size
        assert augmented_images.shape[2] == config.final_size
        tf.summary.image('c_augmented', augmented_images)

        feature_cols = {'x': augmented_images}
        return feature_cols


def stratify_images(image, label, batch_size, init_probs):
    """
    Queue examples of images/labels into roughly even True/False counts


    Args:
        image (Tensor): pixel values for one galaxy. Should be from queue e.g. iterator.get_next() to batch properly.
        label (Tensor): label for one galaxy. Should be from queue e.g. iterator.get_next() to batch properly.
        batch_size (int): size of batch to return

    Returns:
        (Tensor): pixel value batch of 1st dim length batch_size, with other dimensions set by image dimensions
        (Tensor): label batch of 1st dim length batch_size
    """

    assert init_probs is not None  # should not be called with stratify=False
    data_batch, label_batch = tf.contrib.training.stratified_sample(
        [image],
        label,
        target_probs=np.array([0.5, 0.5]),
        init_probs=init_probs,
        batch_size=batch_size,
        enqueue_many=True,  # each image/label is a single example, will be automatically batched (thanks TensorFlow!)
        queue_capacity=batch_size * 100
    )
    return data_batch, label_batch


def augment_images(images, input_config):
    """

    Args:
        images (tf.Variable):
        input_config (InputConfig):

    Returns:

    """
    if input_config.geometric_augmentation:
        images = geometric_augmentation(
            images,
            zoom=input_config.zoom,
            final_size=input_config.final_size,
            central=input_config.zoom_central)

    if input_config.photographic_augmentation:
        images = photographic_augmentation(
            images,
            max_brightness_delta=input_config.max_brightness_delta,
            contrast_range=input_config.contrast_range)

    return images


# def augment_images(images, params):
#     if params['transform']:
#         # images = tf.map_fn(
#         #     lambda image: tf.py_func(
#         #         func=functools.partial(transform_3d, params=params),
#         #         inp=[image],
#         #         Tout=tf.float32,
#         #         stateful=False,
#         #         name='augment'
#         #     ),
#         #     images)
#
#         [images] = tf.py_func(
#                 func=functools.partial(transform_3d, params=params),
#                 inp=[images[n] for n in range(images.shape[0])],
#                 Tout=tf.float32,
#                 stateful=False,
#                 name='augment')
#         images = tf.concat(images, axis=3)


def geometric_augmentation(images, zoom, final_size, central):
    """
    Runs best if image is originally significantly larger than final target size
    for example: load at 256px, rotate/flip, crop to 246px, then finally resize to 64px
    This leads to more computation, but more pixel info is preserved

    # TODO add stretch and/or shear?

    Args:
        images ():
        zoom (tuple): of form {min zoom in decimals e,g, 1.0, max zoom in decimals e.g, 1.2}
        final_size ():

    Returns:
        (Tensor): image rotated, flipped, cropped and (perhaps) normalized, shape (target_size, target_size, channels)
    """

    images = ensure_images_have_batch_dimension(images)

    assert images.shape[1] == images.shape[2]  # must be square
    assert len(zoom) == 2
    assert zoom[0] <= zoom[1]
    assert zoom[1] > 1. and zoom[1] < 10.  # catch user accidentally putting in pixel values here

    # flip functions don't support batch dimension - wrap with map_fn
    images = tf.map_fn(
        tf.image.random_flip_left_right,
        images)
    images = tf.map_fn(
        tf.image.random_flip_up_down,
        images)
    images = tf.map_fn(
        random_rotation,
        images)

    # if zoom = (1., 1.3), zoom randomly between 1x to 1.3x
    images = tf.map_fn(lambda x: crop_random_size(x, zoom=zoom, central=central), images)

    # resize to final desired size (may match crop size)
    images = tf.image.resize_images(
        images,
        tf.constant([final_size, final_size], dtype=tf.int32),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR  # only nearest neighbour works - otherwise gives noise
    )
    return images


def random_rotation(im):
    return tf.contrib.image.rotate(
        im,
        3.14 * tf.random_uniform(shape=[1]),
        interpolation='BILINEAR'
    )


def crop_random_size(im, zoom, central):
    original_width = int(im.shape[1]) # int cast allows division of Dimension
    new_width = int(original_width / np.random.uniform(zoom[0], zoom[1]))
    if central:
        lost_width = int((original_width - new_width) / 2)
        return im[lost_width:-lost_width, lost_width:-lost_width]
    else:
        cropped_shape = tf.constant([new_width, new_width, int(im.shape[2])], dtype=tf.int32)
        return tf.random_crop(im, cropped_shape)



def photographic_augmentation(images, max_brightness_delta, contrast_range):
    """
    TODO do before or after geometric?
        TODO add slight redshifting?
    TODO

    Args:
        images ():
        max_brightness_delta ():
        contrast_range ():

    Returns:

    """
    images = ensure_images_have_batch_dimension(images)

    images = tf.map_fn(
        lambda im: tf.image.random_brightness(im, max_delta=max_brightness_delta),
        images)
    images = tf.map_fn(
        lambda im: tf.image.random_contrast(im, lower=contrast_range[0], upper=contrast_range[1]),
        images)

    return images


def ensure_images_have_batch_dimension(images):
    if len(images.shape) < 3:
        raise ValueError
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=0)  # add a batch dimension
    return images


def predict_input_func(tfrecord_loc, n_galaxies, initial_size, mode='labels'):
    """Wrapper to mimic the run_estimator.py input procedure.
    Get subjects and labels from tfrecord, just like during training
    Subjects must fit in memory, as they are loaded as a single batch
    Args:
        tfrecord_loc (str): tfrecord to read subjects from. Should be test data.
        n_galaxies (int, optional): Defaults to 128. Num of galaxies to predict on, as single batch.

    Returns:
        subjects: np.array of shape (batch, x, y, channel)
        labels: np.array of shape (batch)
    """
    config = InputConfig(
        name='predict',
        tfrecord_loc=tfrecord_loc,
        label_col='label',
        stratify=False,
        shuffle=False,  # important - preserve the order
        repeat=False,
        stratify_probs=None,
        regression=True,
        geometric_augmentation=None,
        photographic_augmentation=None,
        zoom=None,
        fill_mode=None,
        batch_size=n_galaxies,
        initial_size=initial_size,
        final_size=None,
        channels=3,
        noisy_labels=False  # important - we want the actual vote fractions
    )
    if mode == 'labels':
        batch_images, batch_labels = load_batches_with_labels(config)
        id_strs = None
    elif mode == 'id_str':
        batch_images, id_strs = load_batches_with_id_str(config)
        batch_labels = None
    elif mode == 'matrix':
        batch_images = load_batches_without_labels(config)
        batch_labels = None
        id_strs = None
    else:
        raise ValueError('Predict input func. mode not recognised: {}'.format(mode))

    # don't do this! preprocessing is done at predict time, expects raw-ish images
    # preprocessed_batch_images = preprocess_batch(batch_images, config)['x']
    return batch_images, batch_labels, id_strs