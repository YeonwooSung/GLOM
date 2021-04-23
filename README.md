# GLOM

PyTorch implementation of [GLOM](https://arxiv.org/abs/2102.12627)

## 1. Overview

An implementation of Geoffrey Hinton's paper "How to represent part-whole hierarchies in a neural network" for MNIST Dataset.

## 2. Implementation

Three Types of networks per layer of vectors

    1. Top-Down Network
    2. Bottom-up Network
    3. Attention on the same layer Network

### 2 - 1. Intro to State

There is an initial state that all three types of network outputs get added to after every time step. The bottom layer of the state is the input vector where the MNIST pixel data is kept and doesn't get anything added to it to retain the MNIST pixel data. The top layer of the state is the output layer where the loss function is applied and trained to be the one-hot MNIST target vector.
