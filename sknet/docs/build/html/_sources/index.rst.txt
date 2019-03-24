Welcome to sk-net (in construction)
===================================

Sknet is a lightweight library to build and train neural networks in Tensorflow.
This library aims ot fulfill two major goals:

    - provide an out-of-the-box solution for practicioner to reproduce results
    and build upon previous work with a comon working environment
    - provide tools and simple methods following the research direction of
    the development team around max-affine splines to interpret deep neural
    networks
    
This toolbox is oriented for research and education, for any scale projects
do not requiring multi-GPU computing.
We briefly describe here the fundamentals of the toolbox. The project is on _GitHub.


Sknet way ot working
--------------------

The library is built with the deep learning pipeline in mind. That is, it provides
multiple blocks being combined to solve a task. Those blocks are:

    - dataset : any collection of inputs or (input-output) pairs
    
    - pre-processing (optional): pre-processing that can be applied onto any
    dataset for increased performances s.a. zca whitening
    
    - models : a fully describe DNN form input to output, in term of 
    layers s.a. LeNet5
    
    - layers : low-level building blocks of any DNN s.a. dense or conv2D
    
    - optimizer : any updating policy applied onto the learnable weights 
    of a model s.a. adam or sgd
    
    - learning rate schedules: combined with a loss function, a model and 
    a dataset, the learning rate scheduler plays a crucial role to 
    guarantee best performances s.a. stepwise or adaptive
    
    - trainer : a higher-level method assembling those independent blocks 
    into a trainable pipeline, also containing the tensorflow session


In each block, any user can augment the library with their own creation.

  
User Guide
----------

The Sknet user guide explains how to install Sknet, how to build and train
neural networks, and how to contribute to the library as a
developer.

.. toctree::
  :maxdepth: 2

  user/installation
  user/dataset
  user/preprocess
  user/model
  user/layer

API Reference
-------------

If you are looking for information on a specific function:

.. toctree::
  :maxdepth: 2

  modules/dataset
  modules/layer
  modules/model
  

Quickstart
==========


CIFAR100 with LeNet5
--------------------


SVHN with Custom Model
----------------------





About this documentation
========================

Lots of documentation can be found online [#f1]_


.. rubric:: Footnotes

.. [#f1] '<https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#simple-tables>'_


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/RandallBalestriero/sknet.git
.. _Python: https://www.python.org/download/releases/3.0/
.. _Tensorflow: https://www.tensorflow.org/
.. _Anaconda: https://www.anaconda.com/
