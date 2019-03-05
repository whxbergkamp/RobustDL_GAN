import tensorflow as tf
from adversarial_networks.models import (
    simple_cnn_mnist,
)

generator_dict = {
    'simple_cnn_mnist': simple_cnn_mnist.generator,
}

discriminator_dict = {
    'simple_cnn_mnist': simple_cnn_mnist.discriminator,
}

def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)
