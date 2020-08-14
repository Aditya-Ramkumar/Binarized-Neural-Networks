# Binarized-Neural-Networks
This repository contains a Larq based implementation of the networks described in [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830) Larq is an open-source deep learning library for training neural networks with extremely low precision weights and activations.


## Requirements
* Python 3.7, numpy
* [Tensorflow](https://github.com/tensorflow/tensorflow)
* [Larq](https://github.com/larq/larq)

## BinaryNet MNIST MLP
This implementation achieves a test accuracy of **97.1%** after training for 20 epochs.

## Binary LeNet 
A binarized version of LeNet5 is trained for 10 epochs and achieves a test accuracy of **96.93%.**        
