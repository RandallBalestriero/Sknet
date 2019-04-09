:mod:`sknet.optimize`
=====================

.. automodule:: sknet.optimize.schedule

.. toctree::
    :hidden:

    optimize/schedule
    optimize/optimizer
    optimize/loss


.. rubric:: :doc:`optimize/schedule`

.. autosummary::
    :nosignatures:

    constant
    linear
    exponential
    stepwise
    adaptive
    
.. automodule:: sknet.optimize.optimizer

.. rubric:: :doc:`optimize/optimizer`

.. autosummary::
    :nosignatures:

    Adam

.. automodule:: sknet.optimize.loss

.. rubric:: :doc:`optimize/loss`

All the losses are classes which inherit from :py:class:`tf.Tensor` and thus behave
as such (as standard tensorflow variables). Thus one can do the 
following for example ::

    loss1 = l2_norm(W1)
    loss2 = l2_norm(W2)
    loss3 = crossentropy_logits(p=None,q=network_output)
    loss  = loss1+loss2

as such in the above, the variables also contrain the loss attributes such as
``loss3.p`` which will return the placeholder created to compute the loss (as
:py:data:`p` was not provided.


.. autosummary::
    :nosignatures:

    accuracy
    crossentropy_logits
    l2_norm
 
