from .pipeline import classificator_pipeline
from . import tf_record

import os
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64*64))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])


tf_records_path = os.path.join('../../data/tf_records/', 'classificator_0/*.tfrecords')
dataset = classificator_pipeline(tf_records_path,
                                 batch_size=5,
                                 io_parallel_calls=2,
                                 file_parsing_parallelism=1,
                                 augmentation_parallelism=2)

model.fit(dataset, epochs=30, steps_per_epoch=15)
