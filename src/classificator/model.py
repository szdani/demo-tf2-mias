from .pipeline import classificator_pipeline
from . import tf_record

import os
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024*1024))



#tf_records_path = os.path.join(tf_record.TF_RECORDS_PATH, 'classificator_0/*.tfrecords')
#print(tf_records_path)
#dataset = classificator_pipeline(tf_records_path)

#model.compile()
