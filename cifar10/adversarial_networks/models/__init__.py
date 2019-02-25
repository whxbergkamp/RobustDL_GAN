import tensorflow as tf
from adversarial_networks.models import (
    resnet18_2,
    resnet18_2_wide,
)

generator_dict = {
    'resnet18_2': resnet18_2.generator,
    'resnet18_2_wide': resnet18_2_wide.generator,
}

discriminator_dict = {
    'resnet18_2': resnet18_2.discriminator,
    'resnet18_2_wide': resnet18_2_wide.discriminator,
}

def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)
