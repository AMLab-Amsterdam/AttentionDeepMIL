Attention-based Deep Multiple Instance Learning
================================================

by Maximilian Ilse (<ilse.maximilian@gmail.com>), Jakub M. Tomczak (<jakubmkt@gmail.com>) and Max Welling

Overview
--------

PyTorch implementation of our paper "Attention-based Deep Multiple Instance Learning":
* Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. arXiv preprint arXiv:1802.04712. [link](https://arxiv.org/pdf/1802.04712.pdf).


Installation
------------

Installing Pytorch 0.3.1, using pip or conda, should resolve all dependencies.
Tested with Python 2.7, but should work with 3.x as well.
Tested on both CPU and GPU.


Content
--------

The code can be used to run the MNIST-BAGS experiment, see Section 4.2 and Figure 1 in our [paper](https://arxiv.org/pdf/1802.04712.pdf).
In order to have a small and concise experimental setup, the code has the following limitation:
+ Mean bag length parameter shouldn't be much larger than 10, for larger numbers the training dataset will become unbalanced very quickly. You can run the data loader on its own to check, see __main__ part of dataloader.py
+ No validation set is used during training, no early stopping

__NOTE__: In order to run experiments on the histopathology datasets, please download datasets [Breast Cancer](http://bioimage.ucsb.edu/research/bio-segmentation) and [Colon Cancer](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/). In the histopathology experiments we used a similar model to the model in `model.py`, please see the [paper](https://arxiv.org/pdf/1802.04712.pdf) for details.


How to Use
----------
`dataloader.py`: Generates training and test set by combining multiple MNIST images to bags. A bag is given a positive label if it contains one or more images with the label specified by the variable target_number.
If run as main, it computes the ratio of positive bags as well as the mean, max and min value for the number per instances in a bag.

`mnist_bags_loader.py`: Added the original data loader we used in the experiments. It can handle any bag length without the dataset becoming unbalanced. It is most probably not the most efficient way to create the bags. Furthermore it is only test for the case that the target number is ‘9’.

`main.py`: Trains a small CNN with the Adam optimization algorithm.
The training takes 20 epochs. Last, the accuracy and loss of the model on the test set is computed.
In addition, a subset of the bags labels and instance labels are printed.

`model.py`: The model is a modified LeNet-5, see <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>.
The Attention-based MIL pooling is located before the last layer of the model.
The objective function is the negative log-likelihood of the Bernoulli distribution.


Questions and Issues
--------------------

If you find any bugs or have any questions about this code please contact Maximilian or Jakub. We cannot guarantee any support for this software.

Citation
--------------------

Please cite our paper if you use this code in your research:
```
@article{ITW:2018,
  title={Attention-based Deep Multiple Instance Learning},
  author={Ilse, Maximilian and Tomczak, Jakub M and Welling, Max},
  journal={arXiv preprint arXiv:1802.04712},
  year={2018}
}
```

Acknowledgments
--------------------

The work conducted by Maximilian Ilse was funded by the Nederlandse Organisatie voor Wetenschappelijk Onderzoek (Grant DLMedIa: Deep Learning for Medical Image Analysis).

The work conducted by Jakub Tomczak was funded by the European Commission within the Marie Skodowska-Curie Individual Fellowship (Grant No. 702666, ”Deep learning and Bayesian inference for medical imaging”).
