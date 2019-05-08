import tensorflow as tf


def dummy_model_fn(
   features,  # This is batch_features from input_fn
   labels,    # This is batch_labels from input_fn. May be None in predict mode!
   mode,      # An instance of tf.estimator.ModeKeys, see below
   params):   # Additional configuration

    net = tf.feature_column.input_layer(features, params['feature_columns'])  # must know input names/types
    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,  # convenience library for common metrics
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}  # package in dict to return
    #  can return other useful things, but they need to have the form key: array
    tf.summary.scalar('accuracy', accuracy[1])  # record metric for tensorboard

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)  # must return training operation
