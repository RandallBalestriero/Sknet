.. _dataloading-label:

Dataset
=======

Dataset format
--------------

Each dataset has its own attributes and specificities. To allow such
flexibility in Python we represent each dataset in a class inheriting from
a :mod:`dict` object. That is, each dataset is a Python dictionnary with
some key attributes s.a. :py:data:`datum_shape` or :py:data:`n_classes`.
We provide a simple example using the MNIST dataset:

.. literalinclude:: ../../../../examples/quickstart_base.py
   :lines: 60-94




We refer to the last section for details and how to create its own dataset.

Pre-loaded Datasets
-------------------

Sknet provides comon dataset out-of-the-box in the :mod:`sknet.dataset` module as

- :py:meth:`sknet.dataset.load_mnist`

- :py:meth:`sknet.dataset.load_fashionmnist`

- :py:meth:`sknet.dataset.load_svhn`

- :py:meth:`sknet.dataset.load_cifar10`

- :py:meth:`sknet.dataset.load_cifar100`

- :py:meth:`sknet.dataset.load_stl10`

- :py:meth:`sknet.dataset.load_warblr`

- :py:meth:`sknet.dataset.load_freefield1010`

In order to work with a dataset one simply calls the 
dataset loading function and retreives the dataset as output
with any pre-imposed data splitting already applied.


All the pre-coded dataset will be loaded from the path given by the 
environment variable :envvar:`DATASET_PATH`, or an alternative path given at 
the function call. If needs be, the dataset will
be downloaded into this path, prior loading (at first utilisation of
sknet for example). The saved dataset are in compressed format.


Open a pre-loaded dataset
-------------------------

Running the code _quickstart_data_loading.py


.. literalinclude:: ../../../../examples/quickstart_data_loading.py
    :encoding: latin-1
    :language: python

will generate the following figure

.. rubric:: mnist

.. image:: https://i.imgur.com/Zri9DXy.png

.. rubric:: fashionmnist

.. image:: https://i.imgur.com/UxGv0Yc.png

.. rubric:: svhn

.. image:: https://i.imgur.com/4kfgD9a.png

.. rubric:: cifar10

.. image:: https://i.imgur.com/rSRJXAm.png

.. rubric:: cifar100 (superclass,class)

.. image:: https://i.imgur.com/htPYkpn.png 

.. rubric:: stl10

.. image:: https://i.imgur.com/w4HlyjK.png

.. rubric:: warblr

.. image:: https://i.imgur.com/LlHsFIZ.png

.. rubric:: freefield1010

.. image:: https://i.imgur.com/aleEcQa.png


Running for the first time will produce the following output and 
download all the dataset not already present in the default path:


.. include:: ../_static/data_loading_output_1.txt
   :literal:

running for the next times will produce:

.. include:: ../_static/data_loading_output_2.txt
   :literal:


Custom dataset
--------------

The :class:`sknet.dataset.Dataset` class is general and can be used with 
any user's own dataset. In fact, the dataset loading function simply 
automate the data loading process and then set them as a dataset.
Here is an example assuming the user has already loaded its dataset
into the working python script

.. literalinclude:: ../../../../examples/quickstart_base.py
   :lines: 95-129

Dataset split
-------------

A :class:`sknet.dataset.Dataset` object provides many convenient methods s.a.
data splitting as

.. literalinclude:: ../../../../examples/quickstart_base.py
   :lines: 131-160



