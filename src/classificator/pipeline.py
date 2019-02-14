from . import tf_record

import tensorflow as tf

import os


def classificator_pipeline(tf_records_path, batch_size=1, io_parallel_calls=1, file_parsing_parallelism=1):
    print("Reading from " + tf_records_path)

    filenames = tf.data.Dataset.list_files(tf_records_path)
    dataset = filenames.interleave(tf.data.TFRecordDataset, cycle_length=1, num_parallel_calls=io_parallel_calls)
    dataset = dataset.map(map_func=tf_record.ClassificatorTFRecordHandler.parse_tf_record,
                          num_parallel_calls=file_parsing_parallelism)
    # map_and_batch also available but our batch size is < 100
    dataset = dataset.batch(batch_size)
    print(dataset)



#tf.enable_eager_execution()
tf_records_path = os.path.join(tf_record.TF_RECORDS_PATH, 'classificator_0/*.tfrecords')
dataset = classificator_pipeline(tf_records_path)

