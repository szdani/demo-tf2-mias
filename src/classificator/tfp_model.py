

import os
import tensorflow as tf
import tensorflow_probability as tfp

tf.compat.v1.disable_eager_execution()

import datetime

from .pipeline import classificator_pipeline
from . import metrics

LOG_DIR = '../../logs/tfp_classif/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
BATCH_SIZE = 5
NUMBER_OF_DATA = 322
STEPS_PER_EPOCH = int(NUMBER_OF_DATA / BATCH_SIZE)
BETA = 0.7

# Model
kernel_divergence_fn = (lambda q, p, ignore: BETA * tfp.distributions.kl_divergence(q, p)/NUMBER_OF_DATA)

inputs = tf.keras.Input(shape=(512, 512, 1))
x = tf.keras.layers.concatenate([inputs, inputs, inputs])
x = tf.keras.applications.inception_v3.InceptionV3(include_top=False)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(125)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(56)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tfp.layers.DenseFlipout(2, activation='softmax', kernel_divergence_fn=kernel_divergence_fn)(x)
model = tf.keras.Model(inputs=inputs, outputs=x)
kl = tf.math.reduce_sum(model.losses[0])

optimizer = tf.keras.optimizers.Adam(0.01)

kl_metrics = metrics.simple_tensor_wrapper(kl, beta=1/(NUMBER_OF_DATA))

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', kl_metrics])

# Callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
tfboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True, write_images=True)

# Data set
tf_records_path = os.path.join('../../data/tf_records/', 'classificator_0/*.tfrecords')
dataset = classificator_pipeline(tf_records_path,
                                 target_size=(512,512),
                                 batch_size=BATCH_SIZE,
                                 io_parallel_calls=2,
                                 file_parsing_parallelism=2,
                                 augmentation_parallelism=2)

# Training
model.fit(dataset, epochs=300, steps_per_epoch=STEPS_PER_EPOCH, verbose=1, callbacks=[reduce_lr, tfboard])
