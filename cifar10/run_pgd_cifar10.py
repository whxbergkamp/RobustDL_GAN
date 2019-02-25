# experiment on PGD adversarial training method
import os
import argparse

from adversarial_networks.train_pgd_cifar10 import train
from adversarial_networks.inputs import (
    get_filename_queue,
    get_input_image, get_input_cifar10,
    create_batch
)
import pprint
from adversarial_networks import models
import tensorflow as tf


parser = argparse.ArgumentParser(description='Train adversarial PGD model.')
# Architecture
parser.add_argument('--c-dim', default=3, type=int, help='Number of channels.')
parser.add_argument('--z-dim', default=512, type=int, help='Dimensionality of the latent space.')
parser.add_argument('--gf-dim', default=64, type=int, help='Number of filters to use for generator.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--d-architecture', default='resnet18_2', type=str, help='Architecture for discriminator.')

# Training
parser.add_argument('--nsteps', default=200000, type=int, help='Number of steps to run training.')
parser.add_argument('--ntest', default=500, type=int, help='How often to run tests.')
parser.add_argument('--learning-rate', default=0.1, type=float, help='Learning rate for the model.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
# hw added:
parser.add_argument('--train-size', default=50000, type=int, help='Train dataset size.')
parser.add_argument('--test-size', default=10000, type=int, help='Test dataset size.')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')
parser.add_argument('--print-iter', default=100, type=int, help='Number of iterations to print results.')
parser.add_argument('--save-iter', default=20000, type=int, help='Number of iterations to save model.')

parser.add_argument('--log-dir', default='./logs_pgd_cifar10', type=str, help='Where to store log and checkpoint files.')
parser.add_argument('--model-file', default='./model_pgd_cifar10.ckpt', type=str, help='checkpoint file for pgd model.')
parser.add_argument('--random-seed', default='20180526', type=str, help='random seed for tf')
parser.add_argument('--weight-decay', default=1E-4, type=float, help='weight decay parameter')
parser.add_argument('--dataset', default='cifar-10', type=str, help='Which data set to use.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')
parser.add_argument('--split', default='train', type=str, help='Which split to use.')
parser.add_argument('--epsilon', default=0.0625, type=float, help='Epsilon for perturbation.')    # 0.0625=8/256*2 (for range [-1,1])


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
        capacity=10000 + 3 * args.batch_size,         #use parameter replace 10000
        min_after_dequeue=10000
    )


    config = vars(args)

    discriminator = models.get_discriminator(args.d_architecture,
        output_size=output_size, c_dim=args.c_dim, f_dim=args.df_dim)

    train(discriminator, image_batch, test_image_batch, config)


if __name__ == '__main__':
    main()
