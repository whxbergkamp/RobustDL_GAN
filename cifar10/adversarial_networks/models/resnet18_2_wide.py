# Implementation of ResNet modified from 
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf
from tensorflow.contrib import slim
import logging

logger = logging.getLogger(__name__)


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)



def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
    data_format):
  """
  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


# take a look at data format channel-first or not 
def resnet_v1(inputs, training, data_format):
    # for GPU; channels_first
    if data_format=='channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=16, kernel_size=3,
        strides=1, data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = block_layer(
          inputs=inputs, filters=160, bottleneck=False,
          block_fn=_building_block_v1, blocks=3,
          strides=1, training=training,
          name='block_layer_D{}'.format(1), data_format=data_format)
    inputs = block_layer(
          inputs=inputs, filters=320, bottleneck=False,
          block_fn=_building_block_v1, blocks=3,
          strides=2, training=training,
          name='block_layer_D{}'.format(2), data_format=data_format)
    inputs = block_layer(
          inputs=inputs, filters=640, bottleneck=False,
          block_fn=_building_block_v1, blocks=3,
          strides=2, training=training,
          name='block_layer_D{}'.format(3), data_format=data_format)

    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=8,
        strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')

    inputs = tf.reshape(inputs, [-1, 640])
    inputs = tf.layers.dense(inputs=inputs, units=10, kernel_initializer=tf.variance_scaling_initializer())
    inputs = tf.identity(inputs, 'final_dense')
    return inputs




def generator(x, f_dim, output_size, c_dim, is_training=True):
    ngf = f_dim
    inputs = x
    data_format='channels_first'

    # for GPU; channels_first
    if data_format=='channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(inputs, filters=ngf, kernel_size=3, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, is_training, data_format)
    inputs = tf.nn.relu(inputs)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        inputs = conv2d_fixed_padding(inputs, filters=ngf*mult*2, kernel_size=3, strides=2, data_format=data_format)
        inputs = batch_norm(inputs, is_training, data_format)
        inputs = tf.nn.relu(inputs)

    mult = 2**n_downsampling
    inputs = block_layer(
          inputs=inputs, filters=ngf*mult, bottleneck=False,
          block_fn=_building_block_v1, blocks=6,
          strides=1, training=is_training,
          name='block_layer_G{}'.format(1), data_format=data_format)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling-i)
        # fix this 2d transpose
        inputs = tf.layers.conv2d_transpose(inputs, filters=int(ngf*mult/2), kernel_size=3, strides=(2,2), 
                 padding='same', kernel_initializer=tf.truncated_normal_initializer(0.0, 0.02), data_format=data_format, use_bias=False)
        inputs = batch_norm(inputs, is_training, data_format)
        inputs = tf.nn.relu(inputs)

    inputs = tf.layers.conv2d(inputs, filters=c_dim, kernel_size=3, strides=1,
      padding='same', use_bias=True, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.02), data_format=data_format)
    inputs = tf.nn.tanh(inputs)

    if data_format=='channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    return inputs






def discriminator(x, f_dim, output_size, c_dim, is_training=True):
    # Network

    return resnet_v1(x, is_training, 'channels_first')
