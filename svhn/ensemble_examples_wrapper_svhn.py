# wrapper script for black box attacks

import os
import argparse

from adversarial_networks.gen_static_adversarial_examples_svhn import gen_adv_examples
import pprint
from adversarial_networks import models
import tensorflow as tf

import scipy.io as sio
import numpy as np

parser = argparse.ArgumentParser(description='Black box attack on model 1 using model 2.')
parser.add_argument('--c-dim', default=3, type=int, help='Number of channels.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--d-architecture', default='resnet18', type=str, help='Architecture for discriminator.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')
parser.add_argument('--dataset', default='svhn', type=str, help='Which data set to use.')
parser.add_argument('--model-file', default='./model_std_svhn.ckpt', type=str, help='checkpoint file for model.')
parser.add_argument('--sample-file', default='./adv_svhn_std.npz', type=str, help='sample file for adv examples.')
parser.add_argument('--test-sample-file', default='./adv_svhn_std_test.npz', type=str, help='test sample file for adv examples.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')

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


    discriminator = models.get_discriminator(args.d_architecture, scope='discriminator', 
        output_size=64, c_dim=args.c_dim, f_dim=args.df_dim, is_training=True)    

    gen_adv_examples(discriminator, args.model_file, config, image_combined, label_combined, test_image_npy, test_label_npy)


if __name__ == '__main__':
    main()

