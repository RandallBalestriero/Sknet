Welcome to Sknet
================

Sknet is a lightweight python library built upon Tensorflow.
Sknet levereages the low level Tensorflow ops at its core for performances
and GPU computing while prividing its own high level methods.
The library is built with the deep learning pipeline in mind:

    - :ref:`datasets-label`: mostly refurbishes tensorflow_datasets plus additional dataset s.a. cifar10 and imagenet
    - :ref:`preprocess-label`: standard pre-processing techniques s.a. zca whitening and standardization
    - :ref:`networks-label` : some pre defined deep networks such as resnets, densenets, CNNs, built upon  :ref:`layers-label`
    - :ref:`layers-label`: medium-level building blocks of any deep network and are created from the low level :ref:`ops-label`.
    - :ref:`ops-label`: mostly refurbishes and augment the basic tensorflow ops.
    - :ref:`optimizers-label`: collection of gradient descent flavors to do optimization s.a. adam or sgd
    - :ref:`schedules-label`: time-varying functions used for example to have adaptive learning rates s.a. stepwise or adaptive

The above blocks can be considered as refurbishing and augmenting some of the key modules proivded
by Tensorflow to allow completeness of the Sknet library. In addition of the above, two crucial 
components are further introduced in Sknet:

    - sknet.workplace : this class manages the the acutal execution of ops


This toolbox is oriented for research and education, and any projects
do not requiring multi-GPU computing.
We briefly describe here the fundamentals of the toolbox. The project is on _GitHub.

  
User Guide
----------

The Sknet user guide explains how to install Sknet, how to build and train
neural networks, and how to contribute to the library as a developer.

.. toctree::
  :maxdepth: 2

  user/installation
  user/dataset
  user/preprocess
  user/network
  user/schedule
  user/pipeline

API Reference
-------------

If you are looking for information on a specific function:

.. toctree::
  :maxdepth: 2

  modules/base
  modules/dataset
  modules/ops
  modules/layers
  modules/network
  modules/optimize

Quickstart
==========


CIFAR100 with LeNet5
--------------------


SVHN with Custom Model
----------------------





About this documentation
========================

Lots of documentation can be found online [#f1]_


Indices, tables, Index
----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




.. rubric:: Footnotes

.. [#f1] '<https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#simple-tables>'_




.. _GitHub: https://github.com/RandallBalestriero/sknet.git
.. _Python: https://www.python.org/download/releases/3.0/
.. _Tensorflow: https://www.tensorflow.org/
.. _Anaconda: https://www.anaconda.com/
