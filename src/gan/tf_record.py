import os

import tensorflow as tf
import pandas as pd
import cv2

DATA_PATH = 'data/img/'
CSV_PATH = 'data/info.csv'
TF_RECORDS_PATH = 'data/tf_records/'

COLUMN_PATH = 'path'
COLUMN_TARGET = 'target'
COLUMN_ID = 'id'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class ClassificatorTFRecordHandler():
    def __init__(self, name, csv_source=CSV_PATH, img_source=DATA_PATH, destination_directory=TF_RECORDS_PATH):
        self.destination = os.path.join(destination_directory, name)
        self.csv_source = csv_source
        self.img_source = img_source

    def _read_meta_data(self):
        return pd.read_csv(self.csv_source, sep=' ')

    def _create_binary_classification_table(self, table):
        # Returns pandas table ['id', COLUMN_TARGET, COLUMN_PATH]
        # COLUMN_TARGET = 1 if abnormality is NORMAL or 0 if it isn't

        new_table = table[['id']]
        new_table[COLUMN_TARGET] = (table['abnormality'] == 'NORM').astype(int)
        new_table[COLUMN_PATH] = table['id'].apply(lambda id: os.path.join(DATA_PATH, id + '.pgm'))

        return new_table

    def _read_pgm_image(self, img_path):
        # Returns grayscale image! FLAG for cv2: https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#Mat%20imread(const%20String&%20filename,%20int%20flags)
        return cv2.imread(img_path, 0)

    def _create_directory(self):
        try:
            # Create target Directory
            os.mkdir(self.destination)
            print("Directory ", self.destination, " Created ")
        except Exception:
            print("Directory ", self.destination, " already exists or cannot be created")
            return

    def create_dataset(self):
        examples = self._create_binary_classification_table(self._read_meta_data())

        self._create_directory()
        print('TF Records will be created under ' + self.destination)

        shape = None
        for idx, row in examples.iterrows():
            print(str(idx),'/' , str(len(examples)))
            destination = os.path.join(self.destination, row['id'] + '.tfrecords')
            # We can create a compressed writer with TFRecordCompressionType
            with tf.io.TFRecordWriter(destination, tf.io.TFRecordCompressionType.GZIP) as writer:

                raw_img = self._read_pgm_image(row[COLUMN_PATH])
                if shape != raw_img.shape and shape != None:
                    # Currently only one dimension is supported!
                    raise ValueError('Image differs from previously parsed image! ' + shape, raw_img.shape)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(raw_img.shape[0]),
                    'width': _int64_feature(raw_img.shape[1]),
                    'image_raw': _int64_array_feature(raw_img.flatten()),
                    'mask_raw': _int64_feature(row[COLUMN_TARGET])}))

                writer.write(example.SerializeToString())

    @staticmethod
    def parse_tf_record(tf_record):
        keys_to_features = {'height': tf.io.VarLenFeature(tf.int64),
                            'width': tf.io.VarLenFeature(tf.int64),
                            'image_raw': tf.io.VarLenFeature(tf.int64),
                            'mask_raw': tf.io.VarLenFeature(tf.int64)}
        print(tf_record)

        parsed_features = tf.io.parse_single_example(tf_record, keys_to_features)

        reshaped = tf.sparse.reshape(parsed_features['image_raw'],
                                                         (parsed_features['height'].values[0], parsed_features['width']
                                                          .values[0]))
        reshaped = tf.reshape(parsed_features['image_raw'].values,(1024,1024))
        return reshaped, parsed_features['mask_raw'].values[0]
