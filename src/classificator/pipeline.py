from . import tf_record

import tensorflow as tf

import os


def classificator_pipeline(tf_records_path, batch_size=1, io_parallel_calls=1, file_parsing_parallelism=1,
                           augmentation_parallelism=1):
    print("Reading from " + tf_records_path)

    filenames = tf.data.Dataset.list_files(tf_records_path, shuffle=True)
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP', num_parallel_reads=io_parallel_calls)
    dataset = dataset.map(map_func=tf_record.ClassificatorTFRecordHandler.parse_tf_record,
                         num_parallel_calls=file_parsing_parallelism)

    # Cast to float
    cast = lambda x, y: (tf.cast(x, tf.dtypes.float32), tf.cast(y, tf.dtypes.float32))
    dataset = dataset.map(cast, num_parallel_calls=augmentation_parallelism)

    # Clip into [0, 300]
    clip = lambda x, y: (tf.clip_by_value(x, 0.0, 300.0), y)
    dataset = dataset.map(clip, num_parallel_calls=augmentation_parallelism)

    # Augment
    random_brightness = lambda x, y : (tf.image.random_brightness(x, 1.), y)
    dataset = dataset.map(random_brightness, num_parallel_calls=augmentation_parallelism)

    gamma = lambda x, y : (tf.image.adjust_gamma(x, 1, 1), y)
    dataset = dataset.map(gamma, num_parallel_calls=augmentation_parallelism)

    # map_and_batch also available but our batch size is < 100
    dataset = dataset.batch(batch_size)

    return dataset



#tf.enable_eager_execution()
tf_records_path = os.path.join(tf_record.TF_RECORDS_PATH, 'classificator_0/*.tfrecords')
dataset = classificator_pipeline(tf_records_path)

img = iter(dataset)
img = next(img)


import matplotlib.pyplot as plt

print(img[0].shape)
print(img[1].shape)
plt.imshow(img[0][0], cmap='Greys_r')
plt.show()