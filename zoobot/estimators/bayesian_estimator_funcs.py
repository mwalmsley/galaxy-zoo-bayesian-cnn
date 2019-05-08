import logging
import sys

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants


def estimator_wrapper(features, labels, mode, params):
    # estimator model funcs are only allowed to have (features, labels, params) arguments
    # re-order the arguments internally to allow for the custom class to be passed around
    # params is really the model class
    return params.entry_point(features, labels, mode)  # must have exactly the args (features, labels)


class BayesianModel():

    def __init__(
            self,
            image_dim,
            learning_rate=0.001,
            optimizer=tf.train.AdamOptimizer,
            conv1_filters=32,
            conv1_kernel=1,
            conv1_activation=tf.nn.relu,
            conv2_filters=32,
            conv2_kernel=3,
            conv2_activation=tf.nn.relu,
            conv3_filters=16,
            conv3_kernel=3,
            conv3_activation=tf.nn.relu,
            dense1_units=128,
            dense1_dropout=0.5,
            dense1_activation=tf.nn.relu,
            predict_dropout=0.5,
            regression=False,
            log_freq=10,
    ):
        self.image_dim = image_dim
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.conv1_filters = conv1_filters
        self.conv1_kernel = conv1_kernel
        self.conv2_filters = conv2_filters
        self.conv2_kernel = conv2_kernel
        self.conv3_filters = conv3_filters
        self.conv3_kernel = conv3_kernel
        self.dense1_units = dense1_units
        self.dense1_dropout = dense1_dropout
        self.conv1_activation = conv1_activation
        self.conv2_activation = conv2_activation
        self.conv3_activation = conv3_activation
        self.dense1_activation = dense1_activation
        self.pool1_size = 2
        self.pool1_strides = 2
        self.pool2_size = 2
        self.pool2_strides = 2
        self.pool3_size = 2
        self.pool3_strides = 2
        self.padding = 'same'
        self.predict_dropout = predict_dropout  # dropout rate for predict mode
        self.regression = regression
        self.log_freq = log_freq
        self.model_fn = self.main_estimator
        # self.logging_hooks = logging_hooks(self)  # TODO strange error with passing this to estimator in params
        self.logging_hooks = [None, None, None]
        self.entry_point = self.main_estimator

    # TODO move to shared utilities
    # TODO duplicated with input_utils
    def asdict(self):
        excluded_keys = ['__dict__', '__doc__', '__module__', '__weakref__']
        return dict([(key, value) for (key, value) in self.__dict__.items() if key not in excluded_keys])


    def main_estimator(self, features, labels, mode):
        """
        Estimator wrapper function for four-layer cnn performing classification or regression
        Shows the general actions for each Estimator mode
        Details (e.g. neurons, activation funcs, etc) controlled by 'params'

        Args:
            features ():
            labels ():
            mode ():

        Returns:

        """
        response, loss = self.bayesian_regressor(features, labels, mode)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            with tf.variable_scope('predict'):
                export_outputs = {
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(response)
                }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=response, export_outputs=export_outputs)

        assert labels is not None

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope('train'):
                lr = tf.identity(self.learning_rate)
                tf.summary.scalar('learning_rate', lr)
                optimizer = self.optimizer(learning_rate=lr)

                # important to explicitly use within update_ops for batch norm to work
                # see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                logging.warning(update_ops)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(
                        loss=loss,
                        global_step=tf.train.get_global_step())
                
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        else:  # must be EVAL mode
            with tf.variable_scope('eval'):
                # Add evaluation metrics (for EVAL mode)
                eval_metric_ops = get_eval_metric_ops(self, labels, response)
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    def bayesian_regressor(self, features, labels, mode):
        """
        Model function of four-layer CNN
        Can be used in isolation or called within an estimator e.g. four_layer_binary_classifier

        Args:
            features ():
            labels ():
            mode ():
            params ():

        Returns:

        """
        dropout_rate = self.dense1_dropout
        if mode == tf.estimator.ModeKeys.PREDICT:
            dropout_rate = self.predict_dropout

        # eval mode will have a lower loss than train mode, because dropout is off
        dropout_on = (mode == tf.estimator.ModeKeys.TRAIN) or (mode == tf.estimator.ModeKeys.PREDICT)
        tf.summary.scalar('dropout_on', tf.cast(dropout_on, tf.float32))
        tf.summary.scalar('dropout_rate', dropout_rate)

        dense1 = input_to_dense(features, mode, self)  # use batch normalisation
        predictions, response = dense_to_regression(dense1, labels, dropout_on=dropout_on, dropout_rate=dropout_rate)

        # if predict mode, feedforward from dense1 SEVERAL TIMES. Save all predictions under 'all_predictions'.
        if mode == tf.estimator.ModeKeys.PREDICT:
            return response, None  # no loss, as labels not known (in general)

        else: # calculate loss for TRAIN/EVAL with binomial
            labels = tf.stop_gradient(labels)
            scalar_predictions = get_scalar_prediction(predictions)  # softmax, get the 2nd neuron
            loss = binomial_loss(labels, scalar_predictions)
            mean_loss = tf.reduce_mean(loss)
            tf.losses.add_loss(mean_loss)
            return response, mean_loss


def input_to_dense(features, mode, model):
    """

    Args:
        features ():
        mode():
        model (BayesianBinaryModel):

    Returns:

    """
    input_layer = features["x"]
    tf.summary.image('model_input', input_layer, input_layer.shape[-1])

    dropout_on = (mode == tf.estimator.ModeKeys.TRAIN) or (mode == tf.estimator.ModeKeys.PREDICT)
    # dropout_rate = model.dense1_dropout / 10.  # use a much smaller dropout on early layers (should test)
    dropout_rate = 0  # no dropout on conv layers
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=model.conv1_filters,
        kernel_size=[model.conv1_kernel, model.conv1_kernel],
        padding=model.padding,
        activation=model.conv1_activation,
        kernel_regularizer=regularizer,
        name='model/layer1/conv1')
    drop1 = tf.layers.dropout(
        inputs=conv1,
        rate=dropout_rate,
        training=dropout_on)
    conv1b = tf.layers.conv2d(
        inputs=drop1,
        filters=model.conv1_filters,
        kernel_size=[model.conv1_kernel, model.conv1_kernel],
        padding=model.padding,
        activation=model.conv1_activation,
        kernel_regularizer=regularizer,
        name='model/layer1/conv1b')
    drop1b = tf.layers.dropout(
        inputs=conv1b,
        rate=dropout_rate,
        training=dropout_on)
    pool1 = tf.layers.max_pooling2d(
        inputs=drop1b,
        pool_size=[model.pool1_size, model.pool1_size],
        strides=model.pool1_strides,
        name='model/layer1/pool1')
    

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=model.conv2_filters,
        kernel_size=[model.conv2_kernel, model.conv2_kernel],
        padding=model.padding,
        activation=model.conv2_activation,
        kernel_regularizer=regularizer,
        name='model/layer2/conv2')
    drop2 = tf.layers.dropout(
        inputs=conv2,
        rate=dropout_rate,
        training=dropout_on)
    conv2b = tf.layers.conv2d(
        inputs=drop2,
        filters=model.conv2_filters,
        kernel_size=[model.conv2_kernel, model.conv2_kernel],
        padding=model.padding,
        activation=model.conv2_activation,
        kernel_regularizer=regularizer,
        name='model/layer2/conv2b')
    drop2b = tf.layers.dropout(
        inputs=conv2b,
        rate=dropout_rate,
        training=dropout_on)
    pool2 = tf.layers.max_pooling2d(
        inputs=drop2b,
        pool_size=model.pool2_size,
        strides=model.pool2_strides,
        name='model/layer2/pool2')

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=model.conv3_filters,
        kernel_size=[model.conv3_kernel, model.conv3_kernel],
        padding=model.padding,
        activation=model.conv3_activation,
        kernel_regularizer=regularizer,
        name='model/layer3/conv3')
    drop3 = tf.layers.dropout(
        inputs=conv3,
        rate=dropout_rate,
        training=dropout_on)
    pool3 = tf.layers.max_pooling2d(
        inputs=drop3,
        pool_size=[model.pool3_size, model.pool3_size],
        strides=model.pool3_strides,
        name='model/layer3/pool3')

    # identical to conv3
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=model.conv3_filters,
        kernel_size=[model.conv3_kernel, model.conv3_kernel],
        padding=model.padding,
        activation=model.conv3_activation,
        kernel_regularizer=regularizer,
        name='model/layer4/conv4')
    drop4 = tf.layers.dropout(
        inputs=conv4,
        rate=dropout_rate,
        training=dropout_on)
    pool4 = tf.layers.max_pooling2d(
        inputs=drop4,
        pool_size=[model.pool3_size, model.pool3_size],
        strides=model.pool3_strides,
        name='model/layer4/pool4')

    """
    Flatten tensor into a batch of vectors
    Start with image_dim shape. NB, does not change with channels: just alters num of first filters.
    2 * 2 * 2 = 8 factor reduction in shape from pooling, assuming stride 2 and pool_size 2
    length ^ 2 to make shape 1D
    64 filters in final layer
    """
    pool4_flat = tf.reshape(pool4, [-1, int(model.image_dim / 16) ** 2 * model.conv3_filters], name='model/layer4/flat')

    # Dense Layer
    dense1 = tf.layers.dense(
        inputs=pool4_flat,
        units=model.dense1_units,
        activation=model.dense1_activation,
        kernel_regularizer=regularizer,
        name='model/layer4/dense1')

    return dense1


def get_scalar_prediction(prediction):
    return tf.nn.softmax(prediction)[:, 1]


def dense_to_regression(dense1, labels, dropout_on, dropout_rate):
    # helpful example: https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/examples/get_started/regression/custom_regression.py
    # Add dropout operation
    # TODO refactor out, duplication + SRP
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=dropout_rate,
        training=dropout_on)
    tf.summary.tensor_summary('dropout_summary', dropout)

    linear = tf.layers.dense(
        dropout,
        units=2,
        name='layer_after_dropout')
    tf.summary.histogram('layer_after_dropout', linear)

    prediction = linear

    scalar_prediction = get_scalar_prediction(prediction)
    tf.summary.histogram('scalar_prediction', scalar_prediction)
    response = {
        "prediction": scalar_prediction,  # softmaxed
    }
    if labels is not None:
        tf.summary.histogram('yes_votes', labels[:, 0])
        tf.summary.histogram('total_votes', labels[:, 1])
        tf.summary.histogram('observed_vote_fraction', labels[:, 0] / labels[:, 1])
        response.update({
            'labels': tf.identity(labels, name='labels'),  # these are None in predict mode
        })

    # prediction has no softmax yet, response does
    return prediction, response


def binomial_loss(labels, predictions):
    # assume labels are vote fractions and 40 people voted
    # assume predictions are softmaxed (i.e. sum to 1 in second dim)
    # TODO will need to refactor and generalise, but should change tfrecord instead
    one = tf.constant(1., dtype=tf.float32)
    # TODO may be able to use normal python types, not sure about speed
    ep = 1e-8
    epsilon = tf.constant(ep, dtype=tf.float32)

    # multiplication in tf requires floats
    yes_votes = tf.cast(labels[:, 0], tf.float32)
    total_votes = tf.cast(labels[:, 1], tf.float32)
    p_yes = tf.identity(predictions)  # fail loudly if passed out-of-range values

    # negative log likelihood
    bin_loss = -( yes_votes * tf.log(p_yes + epsilon) + (total_votes - yes_votes) * tf.log(one - p_yes + epsilon) )
    tf.summary.histogram('bin_loss', bin_loss)
    tf.summary.histogram('bin_loss_clipped', tf.clip_by_value(bin_loss, 0., 50.))
    return bin_loss


def penalty_if_not_probability(predictions):
    above_one = tf.maximum(predictions, 1.) - 1  # distance above 1
    below_zero = tf.abs(tf.minimum(predictions, 0.))  # distance below 0
    deviation_penalty = tf.reduce_sum(above_one + below_zero) # penalty for deviation in either direction
    tf.summary.histogram('deviation_penalty', deviation_penalty)
    tf.summary.histogram('deviation_penalty_clipped', tf.clip_by_value(deviation_penalty, 0., 30.))
    return deviation_penalty
    # print_op = tf.print('deviation_penalty', deviation_penalty)
    # with tf.control_dependencies([print_op]):
    #     return tf.identity(deviation_penalty)  


def get_eval_metric_ops(self, labels, predictions):
    # record distribution of predictions for tensorboard
    tf.summary.histogram('yes_votes', labels[0, :])
    tf.summary.histogram('total_votes', labels[1, :])
    assert labels.dtype == tf.int64
    assert predictions['prediction'].dtype == tf.float32
    observed_vote_fraction = tf.cast(labels[:, 0], dtype=tf.float32) / tf.cast(labels[:, 1], dtype=tf.float32)
    tf.summary.histogram('observed vote fraction', observed_vote_fraction)
    return {"rmse": tf.metrics.root_mean_squared_error(observed_vote_fraction, predictions['prediction'])}

def logging_hooks(model_config):
    train_tensors = {
        'labels': 'labels',
        # 'logits': 'logits',  may not always exist? TODO
        "probabilities": 'softmax',
        'mean_loss': 'mean_loss'
    }
    train_hook = tf.train.LoggingTensorHook(
        tensors=train_tensors, every_n_iter=model_config.log_freq)

    # eval_hook = train_hook
    eval_hook = tf.train.LoggingTensorHook(
        tensors=train_tensors, every_n_iter=model_config.log_freq)

    prediction_tensors = {}
    # [prediction_tensors.update({'sample_{}/predictions'.format(n): 'sample_{}/predictions'.format(n)}) for n in range(3)]

    prediction_hook = tf.train.LoggingTensorHook(
        tensors=prediction_tensors, every_n_iter=model_config.log_freq
    )

    return [train_hook], [eval_hook], [prediction_hook]  # estimator expects lists of logging hooks
