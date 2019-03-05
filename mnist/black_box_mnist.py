# wrapper script for black box attacks

import os
import argparse

from adversarial_networks.run_attacks_black_box_mnist import black_box_attacks
import pprint
from adversarial_networks import models
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser(description='Black box attack on model 1 using model 2.')
parser.add_argument('--output-size', default=64, type=int, help='Size of samples.')
parser.add_argument('--c-dim', default=1, type=int, help='Number of channels.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--d-architecture1', default='simple_cnn_mnist', type=str, help='Architecture for discriminator 1.')
parser.add_argument('--d-architecture2', default='simple_cnn_mnist', type=str, help='Architecture for discriminator 2.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')
parser.add_argument('--dataset', default='mnist', type=str, help='Which data set to use.')
parser.add_argument('--model-file1', default='./model1.ckpt', type=str, help='checkpoint file for model 1.')
parser.add_argument('--model-file2', default='./model2.ckpt', type=str, help='checkpoint file for model 2.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--test-size', default=10000, type=int, help='Size of test set.')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')

parser.add_argument('--epsilon', default=0.6, type=float, help='Epsilon for perturbation.')


def main():
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    # test data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    # mnist only
    # https://gist.github.com/noahfl/0b244346d4ad2501718bbb226be16b1e

    test_image, test_label = mnist.test.images, mnist.test.labels
    test_image = tf.reshape(test_image, [10000, 28, 28, 1])
    test_image_batch = tf.train.shuffle_batch(
        [test_image, test_label],
        batch_size=args.batch_size,
        enqueue_many=True,
        num_threads=16,
        capacity=10000 + 3 * args.batch_size,
        min_after_dequeue=10000
    )

    config = vars(args)


    discriminator1 = models.get_discriminator(args.d_architecture1, scope='discriminator1', 
        output_size=args.output_size, c_dim=args.c_dim, f_dim=args.df_dim, is_training=True)    
    discriminator2 = models.get_discriminator(args.d_architecture2, scope='discriminator2', 
        output_size=args.output_size, c_dim=args.c_dim, f_dim=args.df_dim, is_training=True)

    black_box_attacks(discriminator1, discriminator2, args.model_file1, args.model_file2, test_image_batch, config)


if __name__ == '__main__':
    main()

