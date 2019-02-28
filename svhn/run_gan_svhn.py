# experiment on our adversarial network
import numpy as np
import argparse

from adversarial_networks.train_gan_svhn import train
import pprint
from adversarial_networks import models
import tensorflow as tf

import scipy.io as sio

parser = argparse.ArgumentParser(description='Train adversarial-network-based robust model.')
# Architecture
parser.add_argument('--c-dim', default=3, type=int, help='Number of channels.')
parser.add_argument('--z-dim', default=512, type=int, help='Dimensionality of the latent space.')
parser.add_argument('--gf-dim', default=64, type=int, help='Number of filters to use for generator.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--g-architecture', default='resnet18', type=str, help='Architecture for generator.')
parser.add_argument('--d-architecture', default='resnet18', type=str, help='Architecture for discriminator.')

# Training
parser.add_argument('--nsteps', default=200000, type=int, help='Number of steps to run training.')
parser.add_argument('--ntest', default=500, type=int, help='How often to run tests.')
parser.add_argument('--learning-rate', default=0.01, type=float, help='Learning rate for the model.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--log-dir', default='./logs_svhn_alt', type=str, help='Where to store log and checkpoint files.')
parser.add_argument('--model-fileD', default='./modelD_svhn_alt.ckpt', type=str, help='checkpoint file for D model.')
parser.add_argument('--model-fileG', default='./modelG_svhn_alt.ckpt', type=str, help='checkpoint file for G model.')
parser.add_argument('--random-seed', default='20180526', type=str, help='random seed for tf')
parser.add_argument('--weight-decay', default=1E-4, type=float, help='weight decay parameter')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')
parser.add_argument('--gamma', default=1E-2, type=int, help='Regularization parameter for gradient regularization')


parser.add_argument('--dataset', default='svhn', type=str, help='Which data set to use.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')

parser.add_argument('--epsilon', default=0.1, type=float, help='Epsilon for perturbation.')


def main():
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))


    # SVHN
    mat_content = sio.loadmat('./SVHN/train_32x32.mat')
    label_npy = mat_content['y']
    label_npy[label_npy==10] = 0
    image_npy = mat_content['X'].astype(np.float32)/256.0

    # extra data
    extra_content = sio.loadmat('./SVHN/extra_32x32.mat')
    extra_label_npy = extra_content['y']
    extra_label_npy[extra_label_npy==10] = 0
    extra_image_npy = extra_content['X'].astype(np.float32)/256.0

    idx = np.random.choice(531131, 80000)
    extra_label_npy = extra_label_npy[idx]
    extra_image_npy = extra_image_npy[:,:,:,idx]

    image_combined = np.concatenate((image_npy, extra_image_npy), axis=3)
    image_combined = np.transpose(image_combined, (3,0,1,2))
    label_combined = np.concatenate((label_npy, extra_label_npy), axis=0)


    # 26032 examples
    mat_content = sio.loadmat('./SVHN/test_32x32.mat')
    test_label_npy = mat_content['y']
    test_label_npy[test_label_npy==10] = 0
    test_image_npy = mat_content['X'].astype(np.float32)/256.0
    test_image_npy = np.transpose(test_image_npy, (3,0,1,2))


    config = vars(args)

    generator = models.get_generator(args.g_architecture,
        output_size=64, c_dim=args.c_dim, f_dim=args.gf_dim)

    discriminator = models.get_discriminator(args.d_architecture,
        output_size=64, c_dim=args.c_dim, f_dim=args.df_dim)

    train(generator, discriminator, config, image_combined, label_combined, test_image_npy, test_label_npy)


if __name__ == '__main__':
    main()
