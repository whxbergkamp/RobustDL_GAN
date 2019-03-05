# experiment on our adversarial network
import argparse

from adversarial_networks.train_gan_mnist import train
import pprint
from adversarial_networks import models
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser(description='Train adversarial-network-based robust model.')
# Architecture
parser.add_argument('--c-dim', default=1, type=int, help='Number of channels.')
parser.add_argument('--z-dim', default=512, type=int, help='Dimensionality of the latent space.')
parser.add_argument('--gf-dim', default=64, type=int, help='Number of filters to use for generator.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--reg-param', default=10., type=float, help='Regularization parameter.')
parser.add_argument('--g-architecture', default='simple_cnn_mnist', type=str, help='Architecture for generator.')
parser.add_argument('--d-architecture', default='simple_cnn_mnist', type=str, help='Architecture for discriminator.')

# Training
parser.add_argument('--nsteps', default=200000, type=int, help='Number of steps to run training.')
parser.add_argument('--ntest', default=500, type=int, help='How often to run tests.')
parser.add_argument('--learning-rate', default=0.01, type=float, help='Learning rate for the model.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--log-dir', default='./logs_mnist_gan', type=str, help='Where to store log and checkpoint files.')
parser.add_argument('--model-fileD', default='./modelD_mnist_alt.ckpt', type=str, help='checkpoint file for fgs model.')
parser.add_argument('--model-fileG', default='./modelG_mnist_alt.ckpt', type=str, help='checkpoint file for fgs model.')
parser.add_argument('--random-seed', default='20180526', type=str, help='random seed for tf')
parser.add_argument('--weight-decay', default=1E-5, type=float, help='weight decay parameter')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')
parser.add_argument('--gamma', default=1E-2, type=int, help='Regularization parameter for gradient regularization')
parser.add_argument('--train-size', default=55000, type=int, help='Size of training set.')
parser.add_argument('--test-size', default=10000, type=int, help='Size of test set.')

parser.add_argument('--dataset', default='mnist', type=str, help='Which data set to use.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')

parser.add_argument('--epsilon', default=0.6, type=float, help='Epsilon for perturbation.')


def main():
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    # Data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    # mnist only
    # https://gist.github.com/noahfl/0b244346d4ad2501718bbb226be16b1e
    image, label = mnist.train.images, mnist.train.labels
    image = tf.reshape(image, [55000, 28, 28, 1])
    image_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=args.batch_size,
        enqueue_many=True, 
        num_threads=16,
        capacity=10000 + 3 * args.batch_size,
        min_after_dequeue=10000
    )

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

    generator = models.get_generator(args.g_architecture,
        output_size=64, c_dim=args.c_dim, f_dim=args.gf_dim)

    discriminator = models.get_discriminator(args.d_architecture,
        output_size=64, c_dim=args.c_dim, f_dim=args.df_dim)

    train(generator, discriminator, image_batch, test_image_batch, config)


if __name__ == '__main__':
    main()
