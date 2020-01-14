import tensorflow as tf


def kl_and_binary_crossentropy(kl, beta=1.0):
    def loss(target, output):
        result = tf.math.add(tf.keras.backend.binary_crossentropy(target, output), beta*kl)
        return result
    return loss

def simple_tensor_wrapper(tensor):
    def loss(_, __):
        return tensor
    return loss
