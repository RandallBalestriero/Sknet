:mod:`sknet.layer`
==================

.. automodule:: sknet.layer

.. toctree::
    :hidden:

    layer/base
    layer/io
    layer/dense
    layer/conv
    layer/pool
    layer/special
    layer/normalize
    layer/perturb
    layer/shape


All the :py:class:`sknet.layer` inherit from this base class. Its documentation
presents all the details about thegeneral behaviors of the layers.

.. autosummary::
    :nosignatures:

    Layer

.. rubric:: :doc:`layer/io`

.. autosummary::
    :nosignatures:

    Input
    
.. rubric:: :doc:`layer/dense`

.. autosummary::
    :nosignatures:

    Dense
   
.. rubric:: :doc:`layer/conv`

.. autosummary::
    :nosignatures:

    Conv2D
 
.. rubric:: :doc:`layer/pool`

.. autosummary::
    :nosignatures:

    Pool
    GlobalSpatialPool
    
.. rubric:: :doc:`layer/special`

.. autosummary::
    :nosignatures:

    Activation
    Spectrogram
    LambdaFunction


.. rubric:: :doc:`layer/perturb`

.. autosummary::
    :nosignatures:

    Dropout
    Gaussian
    Uniform
    RandomCrop
    RandomRot90
    RandomAxisReverse

.. rubric:: :doc:`layer/shape`

.. autosummary::
    :nosignatures:

    Reshape
    Stack
    Concat
    ExpandDim
    Merge
 
