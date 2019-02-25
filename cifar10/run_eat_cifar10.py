# run ensemble adversarial training (EAT) using previously generated adversarial examples

import os
import scipy.misc
import numpy as np
import argparse

from adversarial_networks.train_eat_standard_cifar10 import train
from adversarial_networks.inputs import (
    get_filename_queue,
    get_input_image, get_input_cifar10,
    create_batch
)
import pprint
from adversarial_networks import models
import tensorflow as tf

parser = argparse.ArgumentParser(description='Train EAT model.')
# Architecture
parser.add_argument('--c-dim', default=3, type=int, help='Number of channels.')
parser.add_argument('--z-dim', default=512, type=int, help='Dimensionality of the latent space.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--d-architecture', default='resnet18_2', type=str, help='Architecture for discriminator.')

# Training
parser.add_argument('--nsteps', default=200000, type=int, help='Number of steps to run training.')
parser.add_argument('--ntest', default=500, type=int, help='How often to run tests.')
parser.add_argument('--learning-rate', default=0.1, type=float, help='Learning rate for the model.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--class-num', default=10, type=int, help='Number of dataset classes.')
parser.add_argument('--pgd-iter', default=10, type=int, help='Number of pgd iterations when generate pgd adversarial example.')

parser.add_argument('--log-dir', default='./logs_eat_cifar10', type=str, help='Where to store log and checkpoint files.')
parser.add_argument('--model-file', default='./model_eat_cifar10.ckpt', type=str, help='checkpoint file for fgs model.')
parser.add_argument('--random-seed', default='20180526', type=str, help='random seed for tf')
parser.add_argument('--weight-decay', default=1E-4, type=float, help='weight decay parameter')

parser.add_argument('--eat-train-data', default='./adv_std_cifar10.npz', type=str, help='Training data for EAT in numpy format.')
parser.add_argument('--eat-test-data', default='./adv_std_cifar10_test.npz', type=str, help='Test data for EAT in numpy format.')

parser.add_argument('--epsilon', default=0.0625, type=float, help='Epsilon for perturbation.')    # 0.0625=8/256*2 (for range [-1,1])



def main():
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    config = vars(args)

    # EAT Data
    mat_content = np.load(config['eat_train_data'])
    label_npy = mat_content['labels'].astype(np.int)
    image_npy = mat_content['examples'].astype(np.float32)

    # test data
    mat_content = np.load(config['eat_test_data'])
    test_label_npy = mat_content['labels'].astype(np.int)
    test_image_npy = mat_content['examples'].astype(np.float32)


    discriminator = models.get_discriminator(args.d_architecture,
        output_size=32, c_dim=args.c_dim, f_dim=args.df_dim)

    train(discriminator, config, image_npy, label_npy, test_image_npy, test_label_npy)


if __name__ == '__main__':
    main()
