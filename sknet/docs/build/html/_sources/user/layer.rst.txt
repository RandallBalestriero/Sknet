.. _layer-label:

Layer
=====

All the layers use the following approach.
They can be initialized given an input shape or an input layer. Excpet for the 
:class:`sknet.layer.Input` which can take only an input shape or an input shape and
a tensorflow variable. This flexibility allow anyone to use those object to create
new models. 

Special
-------

Input 
Output
Reshape
Lambda
ExpandDim


Data Augmentation
-----------------

We provide some simple data augmentation tools that can be combined, they are commutative layers:

- random flip

- random crop

- dropout (additive and multiplicative)


for example, running the following code

.. literalinclude:: ../../../../quickstart_perturb.py


.. rubric:: Crop 2626

.. image:: https://i.imgur.com/NQOfAaD.png

.. rubric:: axis reverse (width)

.. image:: https://i.imgur.com/wXeqh9q.png

.. rubric:: Uniform (0,1) additive

.. image:: https://i.imgur.com/dffPCZs.png

.. rubric:: Dropout (0.5)
  
.. image:: https://i.imgur.com/ED5KQLy.png

.. rubric:: Gaussian Additive (0,1)

.. image:: https://i.imgur.com/53pBlq0.png

.. rubric:: Rotation 90

.. image:: https://i.imgur.com/CtEnVya.png

.. rubric:: Uniform Multiplicative

.. image:: https://i.imgur.com/UwRlhvo.png



Transform
---------




Pool
----



