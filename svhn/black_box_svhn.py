# wrapper script for black box attacks

import os
import argparse

from adversarial_networks.run_attacks_black_box_svhn import black_box_attacks
import pprint
from adversarial_networks import models
import tensorflow as tf

import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser(description='Black box attack on model 1 using model 2.')
parser.add_argument('--output-size', default=64, type=int, help='Size of samples.')
parser.add_argument('--c-dim', default=3, type=int, help='Number of channels.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--d-architecture1', default='resnet18', type=str, help='Architecture for discriminator 1.')
parser.add_argument('--d-architecture2', default='resnet18', type=str, help='Architecture for discriminator 2.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')
parser.add_argument('--dataset', default='cifar-10', type=str, help='Which data set to use.')
parser.add_argument('--model-file1', default='./model1.ckpt', type=str, help='checkpoint file for model 1.')
parser.add_argument('--model-file2', default='./model2.ckpt', type=str, help='checkpoint file for model 2.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')

parser.add_argument('--epsilon', default=0.1, type=float, help='Epsilon for perturbation.')


def main():
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    # SVHN
    # 26032 examples
    mat_content = sio.loadmat('./SVHN/test_32x32.mat')
    test_label_npy = mat_content['y']
    test_label_npy[test_label_npy==10] = 0
    test_image_npy = mat_content['X'].astype(np.float32)/256.0
    test_image_npy = np.transpose(test_image_npy, (3,0,1,2))


    config = vars(args)

    discriminator1 = models.get_discriminator(args.d_architecture1, scope='discriminator1', 
        output_size=args.output_size, c_dim=args.c_dim, f_dim=args.df_dim, is_training=True)    
    discriminator2 = models.get_discriminator(args.d_architecture2, scope='discriminator2', 
        output_size=args.output_size, c_dim=args.c_dim, f_dim=args.df_dim, is_training=True)

    black_box_attacks(discriminator1, discriminator2, args.model_file1, args.model_file2, test_image_npy, 
        test_label_npy, config)


if __name__ == '__main__':
    main()

