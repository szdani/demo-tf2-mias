from .pipeline import generator_pipeline

import os
import tensorflow as tf
import datetime


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, activation='relu', input_shape=(128,128,1)))
model.add(tf.keras.layers.Conv2D(filters=16, activation='relu', kernel_size=3))
model.add(tf.keras.layers.Conv2D(filters=32, activation='relu', kernel_size=5, strides=1))
model.add(tf.keras.layers.Conv2D(filters=32, activation='relu', kernel_size=5, strides=2))
model.add(tf.keras.layers.Conv2D(filters=32, activation='relu', kernel_size=5, strides=2))
model.add(tf.keras.layers.Conv2D(filters=3, activation='relu', kernel_size=5, strides=1))

model.add(tf.keras.layers.Conv2DTranspose(filters=16, activation='relu', kernel_size=5, strides=1))
model.add(tf.keras.layers.Conv2DTranspose(filters=32, activation='relu', kernel_size=5, strides=2))
model.add(tf.keras.layers.Conv2DTranspose(filters=32, activation='relu', kernel_size=5, strides=2))
model.add(tf.keras.layers.Conv2DTranspose(filters=16, activation='relu', kernel_size=5, strides=1))
model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=1))
model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4))
#model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mse'])


tf_records_path = os.path.join('data/tf_records/', 'classificator_0/*.tfrecords')
dataset = generator_pipeline(tf_records_path,
                                 reshape_size=(128,128),
                                 batch_size=25,
                                 io_parallel_calls=2,
                                 file_parsing_parallelism=1,
                                 augmentation_parallelism=4)

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



model.fit(dataset, epochs=40, steps_per_epoch=64, callbacks=[tensorboard_callback])

pass
for r in dataset:
    r = model.predict(r)
    import matplotlib.pyplot as plt
    print(r.shape)
    plt.imshow(r[0,:,:,0])
    plt.show()
    break