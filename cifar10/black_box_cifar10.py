# wrapper script for black box attacks

import os
import argparse

from adversarial_networks.run_attacks_black_box_cifar10 import black_box_attacks
from adversarial_networks.inputs import (
    get_filename_queue,
    get_input_image, get_input_cifar10,
    create_batch
)
import pprint
from adversarial_networks import models
import tensorflow as tf



parser = argparse.ArgumentParser(description='Black box attack on model 1 using model 2.')
parser.add_argument('--output-size', default=64, type=int, help='Size of samples.')
parser.add_argument('--c-dim', default=3, type=int, help='Number of channels.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--d-architecture1', default='resnet18_2', type=str, help='Architecture for discriminator 1.')
parser.add_argument('--d-architecture2', default='resnet18_2', type=str, help='Architecture for discriminator 2.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')
parser.add_argument('--dataset', default='cifar-10', type=str, help='Which data set to use.')
parser.add_argument('--model-file1', default='./model1.ckpt', type=str, help='checkpoint file for model 1.')
parser.add_argument('--model-file2', default='./model2.ckpt', type=str, help='checkpoint file for model 2.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--test-size', default=10000, type=int, help='Size of test set.')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')

parser.add_argument('--epsilon', default=0.0625, type=float, help='Epsilon for perturbation.')    # 0.0625=8/256*2 (for range [-1,1])


def main():
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    # Test data
    test_filename_queue = get_filename_queue(
        split_file=os.path.join(args.data_dir, 'splits', args.dataset, 'test.lst'),
        data_dir=os.path.join(args.data_dir, args.dataset)
    )

    test_image, test_label = get_input_cifar10(test_filename_queue)

    test_image_batch = tf.train.shuffle_batch(
        [test_image, test_label],
        batch_size=args.batch_size,
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

