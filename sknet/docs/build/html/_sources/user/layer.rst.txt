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


Transform
---------




Pool
----



