import tensorflow as tf
from tensorflow.python.saved_model import signature_constants


def dummy_model_fn(
   features,  # This is batch_features from input_fn
   labels,    # This is batch_labels from input_fn. May be None in predict mode!
   mode,      # An instance of tf.estimator.ModeKeys, see below
   params):   # Additional configuration

    input_layer = features["x"]

    conv = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name='conv')

    pool = tf.layers.max_pooling2d(
        inputs=conv,
        pool_size=[2, 2],
        strides=2,
        name='pool')

    pool_flat = tf.reshape(pool, [-1, params['image_dim'] * 224], name='flat')

    # Dense Layer
    dense = tf.layers.dense(
        inputs=pool_flat,
        units=1064,
        activation=tf.nn.relu,
        name='dense')

    # Logits layer
    logits = tf.layers.dense(inputs=dense, units=2, name='logits')
    probabilities = tf.nn.softmax(logits, name="softmax_tensor")
    scores = tf.nn.softmax(logits, name='score')[:, 0]
    predicted_classes = tf.argmax(logits, 1)

    predictions = {
        "probabilities": probabilities,
        "predictions": scores,
        "classes": predicted_classes
    }

    if labels is not None:
        predictions.update({
            'labels': tf.identity(labels, name='labels'),  # these are None in predict mode
            "classes": tf.argmax(input=logits, axis=1, name='classes'),
        })

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput({
                'predictions': predictions['probabilities']
                })
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,  # convenience library for common metrics
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}  # package in dict to return
    tf.summary.scalar('accuracy', accuracy[1])  # record metric for tensorboard

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)  # must return training operation