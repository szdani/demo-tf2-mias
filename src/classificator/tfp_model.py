

import os
import tensorflow as tf
import tensorflow_probability as tfp

tf.compat.v1.disable_eager_execution()

import datetime

from .pipeline import classificator_pipeline
from . import metrics

LOG_DIR = '../../logs/tfp_classif/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Model
inputs = tf.keras.Input(shape=(512, 512, 1))
x = tf.keras.layers.concatenate([inputs, inputs, inputs])
x = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights=None)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(125)(x)
x = tf.keras.layers.Dense(56)(x)
x = tfp.layers.DenseFlipout(2, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=x)
kl = tf.math.reduce_sum(model.losses[0])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=metrics.kl_and_binary_crossentropy(kl, beta=0.8),
              metrics=['accuracy', 'binary_crossentropy', metrics.simple_tensor_wrapper(kl)])

# Callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
tfboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True, write_images=True)

# Data set
tf_records_path = os.path.join('../../data/tf_records/', 'classificator_0/*.tfrecords')
dataset = classificator_pipeline(tf_records_path,
                                 reshape_size=(512,512),
                                 batch_size=5,
                                 io_parallel_calls=8,
                                 file_parsing_parallelism=8,
                                 augmentation_parallelism=6)

# Training
model.fit(dataset, epochs=300, steps_per_epoch=64, verbose=2, callbacks=[reduce_lr, tfboard])
