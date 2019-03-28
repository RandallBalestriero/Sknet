Welcome to Sknet (in construction)
===================================

Sknet is a lightweight library built mainly upon numpy and tensorflow.
The aim is to provide a fully independent and self content toolbox. Similarly
to matplotlib, the user can either run some state-of-the-art deep learning
methods on the most comon dataset with just a few lines of codes and no prerequisites;
but also accees any block and any subtleties of the provided methods to
tweak, experiment, or improve upon them. As such, this library aims to
fulfill two major goals:

    - provide an out-of-the-box solution for practicioner allowing to reproduce 
      results and build upon previous work with a comon working environment without
      any prerequisites or additional ressources
    - Allow easy access and modification of any of the provided methods to allow
      anyone to experiment and improve upon existing methods

The above makes this toolbox oriented for any party interested in trying deep
learning methods for some specific tasks to researchers in need of a qualitative
and self content toolbox. By providing to the user enough transparency and 
flexibility to implement their own creation without requiring to redo all the
other parts of the pipeline we hope to 

    - allow fast and easy experimenting on any part of the deep learning
      pipeline
    - allow anyone to easily validate their idea without requiring time
      ressources in coding parts independent from the idea to test
    - ensure that provided and used methods follow the guidelines used by
      the developers and practicioners

    
This toolbox is oriented for research and education, and any projects
do not requiring multi-GPU computing.
We briefly describe here the fundamentals of the toolbox. The project is on _GitHub.


Sknet way of working
--------------------

The library is built with the deep learning pipeline in mind. That is, it provides
multiple blocks which are highly customizable. Those blocks are then combined into a 
pipeline to solve a task. Those blocks are:

    - :ref:`dataloading-label`: any collection of inputs or (input-output) pairs
    - :ref:`preprocess-label` (optional): pre-processing that can be applied onto any
      dataset for increased performances s.a. zca whitening
    - :ref:`model-label`: a fully describe DNN form input to output, in term of 
      layers s.a. LeNet5
    - :ref:`layer-label`: low-level building blocks of any DNN s.a. dense or conv2D
    - optimizer : any updating policy applied onto the learnable weights 
      of a model s.a. adam or sgd
    - learning rate schedules: combined with a loss function, a model and 
      a dataset, the learning rate scheduler plays a crucial role to 
      guarantee best performances s.a. stepwise or adaptive
    - trainer : a higher-level method assembling those independent blocks 
      into a trainable pipeline, also containing the tensorflow session

  
User Guide
----------

The Sknet user guide explains how to install Sknet, how to build and train
neural networks, and how to contribute to the library as a developer.

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
  modules/model
  modules/layer
  modules/schedule

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
