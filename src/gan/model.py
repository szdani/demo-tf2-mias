from .pipeline import generator_pipeline

import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import sys

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*16, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 16)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(18, (2, 2), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.Dense(128))
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)):
    real_loss = loss(tf.ones_like(real_output), real_output)
    fake_loss = loss(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)):
    return loss(tf.ones_like(fake_output), fake_output)

@tf.function
def _train_step(images, batch_size, generator, discriminator, generator_optimizer, discriminator_optimizer,
                noise_dim, train_loss_generator, train_loss_discriminator):
    """This function signature is just ugly.

    :param images:
    :return:
    """
    noise = tf.random.normal([batch_size, noise_dim])

    # Create GradientTape instance to 'track' every computational point in the graph
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Get gradients from the 'tape'
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Update weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Metrics
    train_loss_generator(gen_loss)
    train_loss_discriminator(disc_loss)

class GANTrainer():

    def __init__(self, batch_size = 20, noise_dim=100, base_log_dir='logs/gradient_tape/'):
        self.base_log_dir = base_log_dir
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        # Make models
        self.discriminator = make_discriminator_model()
        self.generator = make_generator_model()

        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        # Metrics for training
        self.train_loss_generator = tf.keras.metrics.Mean('train_loss_generator', dtype=tf.float32)
        self.train_loss_discriminator = tf.keras.metrics.Mean('train_loss_discriminator', dtype=tf.float32)
        ## Fixed seed for Tensorboard
        self.seed_noise = tf.random.normal([1, self.noise_dim])

    def summary(self, line_length=100):
        self.generator.summary(line_length)
        self.discriminator.summary(line_length)

    def save(self):
        tf.saved_model.save(self.generator, self.base_log_dir + '/gen/model')
        tf.saved_model.save(self.discriminator, self.base_log_dir + '/dis/model')

    def train(self, dataset, epochs=10, steps_per_epoch=100):
        """Train the GAN for given epochs. After every 'steps_per_epoch' training step it will create an image with
        the generator and with the same seed number.

        :param dataset: Tensorflow dataset.
        :param epochs: Int: Number of epochs to train the models. Default value is 10.
        :return: None
        """

        # Logging - created here because the actual 'train' call and the creation of the class can be very different.
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir_generator = self.base_log_dir + current_time + '/gen'
        train_log_dir_discriminator = self.base_log_dir + current_time + '/dis'
        train_summary_writer_generator = tf.summary.create_file_writer(train_log_dir_generator)
        train_summary_writer_discriminator = tf.summary.create_file_writer(train_log_dir_discriminator)

        # Actual Training
        idx = 0
        epoch = 0
        for image_batch in dataset:
            _train_step(image_batch, self.batch_size, self.generator, self.discriminator, self.generator_optimizer,
                            self.discriminator_optimizer, self.noise_dim, self.train_loss_generator,
                            self.train_loss_discriminator)

            with train_summary_writer_generator.as_default():
                tf.summary.scalar('train_loss_generator', self.train_loss_generator.result(), step=idx)
            with train_summary_writer_discriminator.as_default():
                tf.summary.scalar('train_loss_discriminator', self.train_loss_discriminator.result(), step=idx)

            idx = idx + 1

            if idx % steps_per_epoch == 0:
                epoch = epoch + 1
                generated_image = self.generator(self.seed_noise, training=False)
                with train_summary_writer_generator.as_default():
                    tf.summary.image("Generator Images", generated_image, step=epoch)

                template = 'Epoch {}, G Loss: {}, D Loss: {}'
                print(template.format(epoch,
                                      self.train_loss_generator.result(),
                                      self.train_loss_discriminator.result()))

                self.train_loss_generator.reset_states()
                self.train_loss_discriminator.reset_states()

            if epoch == epochs:
                print('Training ended!')
                return
