
import tensorflow as tf
from tensorflow.contrib import slim
import logging

logger = logging.getLogger(__name__)

def generator(x, f_dim, output_size, c_dim, is_training=True):
    ngf = f_dim

    net = tf.layers.conv2d(x, filters=ngf, kernel_size=5, strides=2, padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2))
    net = tf.layers.batch_normalization(net, training=True)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, filters=ngf*2, kernel_size=5, strides=2, padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2))
    net = tf.layers.batch_normalization(net, training=True)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d_transpose(net, filters=ngf, kernel_size=5, strides=2, padding='same', use_bias=False, kernel_initializer=tf.variance_scaling_initializer(scale=2))
    net = tf.layers.batch_normalization(net, training=True)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d_transpose(net, filters=c_dim, kernel_size=5, strides=2, padding='same', use_bias=False, kernel_initializer=tf.variance_scaling_initializer(scale=2))
    
    net = tf.nn.tanh(net)

    return net


def discriminator(x, f_dim, output_size, c_dim, is_training=True):

    net = slim.conv2d(x, 32, [5, 5], [1, 1], activation_fn=tf.nn.relu, scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], [1, 1], activation_fn=tf.nn.relu, scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = tf.reshape(net, [-1, 7*7*64])
    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc3')

    return net
