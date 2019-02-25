# wrapper script for generating static adversarial examples

import os
import argparse

from adversarial_networks.gen_static_adversarial_examples import gen_adv_examples
from adversarial_networks.inputs import (
    get_filename_queue,
    get_input_image, get_input_cifar10,
    create_batch
)
import pprint
from adversarial_networks import models
import tensorflow as tf



parser = argparse.ArgumentParser(description='Black box attack on model 1 using model 2.')
parser.add_argument('--c-dim', default=3, type=int, help='Number of channels.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--d-architecture', default='resnet18_2', type=str, help='Architecture for discriminator.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')
parser.add_argument('--dataset', default='cifar-10', type=str, help='Which data set to use.')
parser.add_argument('--model-file', default='./model_std_cifar10.ckpt', type=str, help='checkpoint file for model.')
parser.add_argument('--sample-file', default='./adv_std_cifar10.npz', type=str, help='sample file for adv examples.')
parser.add_argument('--test-sample-file', default='./adv_std_cifar10_test.npz', type=str, help='test sample file for adv examples.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--train-size', default=50000, type=int, help='Size of test set.')
parser.add_argument('--test-size', default=10000, type=int, help='Size of test set.')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')
parser.add_argument('--split', default='train', type=str, help='Which split to use.')

parser.add_argument('--epsilon', default=0.0625, type=float, help='Epsilon for perturbation.')


def main():
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    # Data
    filename_queue = get_filename_queue(
        split_file=os.path.join(args.data_dir, 'splits', args.dataset, args.split + '.lst'),
        data_dir=os.path.join(args.data_dir, args.dataset)
    )
    # test data
    test_filename_queue = get_filename_queue(
        split_file=os.path.join(args.data_dir, 'splits', args.dataset, 'test.lst'),
        data_dir=os.path.join(args.data_dir, args.dataset)
    )

    image, label = get_input_cifar10(filename_queue)
    output_size = 32
    c_dim = 3

    test_image, test_label = get_input_cifar10(test_filename_queue)

    image_batch = create_batch([image, label], batch_size=args.batch_size,
        num_preprocess_threads=16, min_queue_examples=10000)

    test_image_batch = tf.train.shuffle_batch(
        [test_image, test_label],
        batch_size=args.batch_size,
        num_threads=16,
        capacity=10000 + 3 * args.batch_size,
        min_after_dequeue=10000
    )


    config = vars(args)


    discriminator = models.get_discriminator(args.d_architecture, scope='discriminator', 
        output_size=output_size, c_dim=args.c_dim, f_dim=args.df_dim, is_training=True)    

    gen_adv_examples(discriminator, args.model_file, image_batch, test_image_batch, config)


if __name__ == '__main__':
    main()

