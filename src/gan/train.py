import os

from . import model
from .pipeline import generator_pipeline

BATCH_SIZE = 25
EPOCHS = 80
TF_RECORDS_PATH = os.path.join('data/tf_records/', 'classificator_0/*.tfrecords')

if __name__ == "__main__":
    dataset = generator_pipeline(TF_RECORDS_PATH,
                                 reshape_size=(128, 128),
                                 batch_size=BATCH_SIZE,
                                 io_parallel_calls=3,
                                 file_parsing_parallelism=3,
                                 augmentation_parallelism=4)

    gan = model.GANTrainer()
    gan.summary()
    gan.train(dataset, epochs=EPOCHS)
    gan.save()


