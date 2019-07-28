:mod:`sknet.ops`
==================

.. automodule:: sknet.ops

.. toctree::
    :hidden:

    ops/base
    ops/dense
    ops/conv
    ops/pool
    ops/special
    ops/normalize
    ops/perturb
    ops/shape


All the :py:class:`sknet.ops` inherit from this base class. Its documentation
presents all the details about thegeneral behaviors of the layers.

.. autosummary::
    :nosignatures:

    Op

    
.. rubric:: :doc:`ops/dense`

.. autosummary::
    :nosignatures:

    Dense
   
.. rubric:: :doc:`ops/conv`

.. autosummary::
    :nosignatures:

    Conv2D
    SplineWaveletTransform
 
.. rubric:: :doc:`ops/pool`

.. autosummary::
    :nosignatures:

    Pool
    
.. rubric:: :doc:`ops/special`

.. autosummary::
    :nosignatures:

    Activation
    Spectrogram
    LambdaFunction

.. rubric:: :doc:`ops/normalize`

.. autosummary::
    :nosignatures:

     BatchNorm


.. rubric:: :doc:`ops/perturb`

.. autosummary::
    :nosignatures:

    Dropout
    Gaussian
    Uniform
    RandomCrop
    RandomRot90
    RandomAxisReverse

.. rubric:: :doc:`ops/shape`

.. autosummary::
    :nosignatures:

    Reshape
    Stack
    Concat
    ExpandDim
    Merge
 
