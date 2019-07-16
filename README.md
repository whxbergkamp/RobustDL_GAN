# Robust Deep Learning with Adversarial Networks

This is the code accompanying the paper [A Direct Approach to Robust Deep Learning Using Adversarial Networks](https://openreview.net/forum?id=S1lIMn05F7&noteId=H1lmqBBmxV). 

```
@INPROCEEDINGS{Wang2019ICLR,
  title={A Direct Approach to Robust Deep Learning Using Adversarial Networks},
  author={Huaxia Wang and Chun-Nam Yu},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```


## Getting Started

The code is divided into 4 subdirectories based on experiments on 4 different datasets: MNIST, SVHN, CIFAR-10 and CIFAR-100. 
The subdirectories share substantial amount of code and differ mostly in data processing, neural network model definitions, and training parameter settings. 
The experiments were run with Tensorflow 1.10. 
We include code for training standard models, adversarial training with projected gradient descent, our adversarial networks approach, and our implementation of ensemble adversarial training. 


### Data Preparation

For the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset download the cropped digits (Format 2) and put them in a folder called **SVHN**. 


For the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) downlaod the binary version of the data and extract in the **data** folder. 

For the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) downlaod the binary version of the data and extract in the **cifar100_expr** folder. 
## Running the code

We give examples on how to run the code through the CIFAR-10 dataset. The commands for running the experiments on other datasets are similar. 

### Training the models


To run training on standard (undefended) neural network models: 
```
python run_standard_cifar10.py --learning-rate=0.1 --weight-decay=1E-4 --batch-size=64 --model-file=model_std_cifar10.ckpt 
```
The details of these parameters and command line options can be found in the script **run_standard_cifar.py** and are largely self-explanatory. They can also be directly set in the scripts. Similarly to run adversarial training with PGD, use the script **run_pgd_cifar.py**. 

```
python run_pgd_cifar10.py --learning-rate=0.1 --weight-decay=1E-5 --epsilon=0.0625 --batch-size=64 --model-file=model_pgd_cifar10.ckpt 
```

The neural network architecture definitions, for both the generative and discriminative networks, are stored in the folder 'adversarial_networks/models/'. 
New architectures can be defined and stored in that folder. 


To run training with adversarial networks: 
```
python run_gan_cifar10.py --learning-rate=0.1 --weight-decay=1E-5 --epsilon=0.0625 --batch-size=64 --model-fileD=modelD_gan_cifar10.ckpt --model-fileG=modelG_gan_cifar10.ckpt 
```


### Evaluating the models using black box and white box attacks

```
python black_box_cifar10.py --model-file1=model1.ckpt --model-file2=model2.ckpt --epsilon=0.0625
```

The script evaluates **model1.ckpt** using adversarial examples generated from **model2.ckpt** with FGS and PGD (black box attacks). 
The model files **model1.ckpt** and **model2.ckpt** are generated from the training commands above. 
To perform white box attacks just use the same model file for **model-file1** and **model-file2**. 


### Training the ensemble adversarial training (EAT) models

Training ensemble adversarial training (EAT) models takes two steps. 
First we need to generate a dataset augmented with adversarial examples: 

```
python ensemble_examples_wrapper.py --model-file=model.ckpt --sample-file=adv_cifar10.npz --test-sample-file=adv_cifar10_test.npz --epsilon=0.0625
```

This script generates adversarial examples using FGS, PGD, and least likely class with a previously trained model **model.ckpt**, and saves the generated examples along with original data in **adv_cifar10.npz** for training and **adv_cifar10_test.npz** for testing. 

To train the actual EAT model, run
```
python run_eat_cifar10.py --eat-train-data=adv_cifar10.npz --eat-test-data=adv_cifar10_test.npz --model-file=model_eat_cifar10.ckpt
```



## Authors

* Huaxia Wang

* Chun-Nam Yu


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* The original version of this code was based on [The Numerics of GANs](https://github.com/LMescheder/TheNumericsOfGANs) . 

