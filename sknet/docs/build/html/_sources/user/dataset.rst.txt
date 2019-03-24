Data Loading
============

Dataset
-------

Sknet provides comon dataset out-of-the-box in the _sknet.dataset module as

- sknet.dataset.mnist

- sknet.dataset.fashionmnist

- sknet.dataset.svhn

- sknet.dataset.cifar10

- sknet.dataset.cifar100

In order to load the train/valid/test set one simply 
calls the load function as in::

	train_set,valid_set,test_set = sknet.dataset.mnist.load()
	
where each set contains first the inputs and then the labels as in::

	train_x,train_y = train_set
	valid_x,valid_y = valid_set
	test_x,test_y   = test_set

To access the image shape, the data format or additional dataset 
specific attributes simply do::

	mnist_image_shape = sknet.dataset.mnist.image_shape #(1,28,28)
	mnist_data_format = sknet.dataset.mnist.data_format # 'NCHW'

The standard format is 'NCHW'. But the 
library supports transparently the 'NHWC'.

All the pre-coded dataset will be loaded from the path given by the 
environment variable DATASET_PATH. If needs be, the dataset will first
be downloaded into this path prior loading (at first utilisation of
sknet for example). The saved dataset are in compressed format.


Adding a dataset
----------------
Each dataset has its correpsonding file in sknet/dataset/dataset_name.py
in which all the attributes and the load function are defined.
To add a dataset, one needs to create a .py file with name being the one
of the dataset. In this file one needs

- load function
- image_shape, data_format attributes


Code Example
------------

Running the code _quickstart_data_loading.py

.. literalinclude:: /quickstart_data_loading.py
    :encoding: latin-1
    :language: python

 will generate the following figure

.. figure:: ./test_loading.png
   :scale: 50 %
   :alt: dataset figure

   Preloaded dataset from sknet.


running for the first time will produce::

Loading MNIST
Dataset MNIST loaded in 0.71 s.
Loading FASHION-MNIST
Dataset FASHION-MNIST loaded in 0.33 s.
Loading SVHN
Dataset SVHN loaded in 2.74 s.
Loading CIFAR10
Dataset CIFAR10 loaded in 2.37 s.
Loading CIFAR100
Dataset CIFAR100 loaded in 2.39 s.


.. _quickstart_data_loading.py: https://github.org/RandallBalestriero/sknet/quickstart_data_loading.py
.. _sknet.dataset: https://github.org/RandallBalestriero/sknet/sknet/dataset
