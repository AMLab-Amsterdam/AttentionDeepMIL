Attention-based Deep Multiple Instance Learning
================================================

by Maximilian Ilse (<ilse.maximilian@gmail.com>) and Jakub M. Tomczak (<jakubmkt@gmail.com >), add Max???

Overview
--------

PyTorch implementation of our proposed Attention-based Multiple Instance Learning (MIL) architecture, see <https://arxiv.org/pdf/1802.04712.pdf>.


Installation
------------

Installing Pytorch 0.3.1, using pip or conda, should resolve all dependencies. Tested with Python 2.7, but should work with 3.x as well. CPU, GPU anything goes.


Contents
--------

The code can be used to run the MNIST-BAGS experiment, see Section 4.2 and Figure 1 in <https://arxiv.org/pdf/1802.04712.pdf>.
In order to have a small and concise experimental setup, the code has the following limitation:
+ Mean bag length parameter shouldn't be much larger than 10, for larger numbers the training dataset will become unbalanced very quick. You can run the data loader on its own to check, see __main__ part of dataloader.py
+ No validation set is used during training, no early stopping


How to Use
----------
dataloader.py: Generates training and test set by combining multiple MNIST images to bags. A bag is given a positive label if it contains one or more images with the label specified by the variable target_number.
If run as main, it computes the ratio of positive bags as well as the mean, max and min value for the number per instances in a bag.

main.py: Trains a small CNN with the Adam optimization algorithm.
The training takes 20 epoches. Last, the accuracy and loss of the model on the test set is computed.
In addition, a subset of the bags labels and instance labels are printed.

model.py: The model is a modified LeNet-5, see <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>.
The Attention-based MIL pooling is located before the last layer of the model.
The objective function is the negative log-likelihood of the Bernoulli distribution ???.


Questions and Issues
--------------------

If you find any bugs or have any questions about this code please contact ???. We cannot guarantee any support for
this software.
