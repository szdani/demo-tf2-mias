from .pipeline import classificator_pipeline

import os
import tensorflow as tf
import tensorflow_probability as tfp

import datetime

LOG_DIR = '../../logs/classif/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights=None,
                                                       input_shape=(512,512,1), classes=2)

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
tfboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True, write_images=True)

tf_records_path = os.path.join('../../data/tf_records/', 'classificator_0/*.tfrecords')
dataset = classificator_pipeline(tf_records_path,
                                 target_size=(512,512),
                                 batch_size=5,
                                 io_parallel_calls=2,
                                 file_parsing_parallelism=1,
                                 augmentation_parallelism=2)


model.fit(dataset, epochs=300, steps_per_epoch=200, verbose=2, callbacks=[reduce_lr, tfboard])
