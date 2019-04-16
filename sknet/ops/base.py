#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from .. import Tensor

class Op(Tensor):
    """Op class used as parent of any implemented layer.

    :py:class:`sknet` relies extremely heavily on the :py:class:`sknet.layer`
    modules. The layers store the computations, the losses, the penalties,
    and everything that might be needed for deep learning applications. 
    As such, understanding the parent class :py:class:`sknet.layer.Op`
    from which all the layers inherit is crucial.

    .. rubric:: Ops are tensors

    The :py:class:`sknet.layer.Op` class inherits from 
    :py:class:`sknet.ObservedTensor` for two reasons. First, it
    allows the layer to be treater as a tensor in the computational
    graph and thus to be used as a standard variable for computation, such
    as ::

        # Set some arbitrary layer given some previous_layer
        some_layer = sknet.layer.Dense(previous,units = 20)
        # the variable some_layer is a Op, treated as a
        # standard tf.Tensor and thus can be used as a variable
        some_stats = tf.reduce_sum(some_layer/0.1+4,0)

    this allows great flexibility in the code as well as for customized
    computations without requiring to access to some attribute or method
    from the instanciated layer.

    Another implication is the ability to use the standard tensorflow
    Tensor methods such as (to get the output shape) ``some_layer.shape``
    which will return a tensorflow shape, or ``some_layer.shape.as_list()``
    to get it as a standard listo f int.

    .. rubric:: Op variables and variable functions:

    All the variables of a layer (let say :py:data:`W` and :py:data:`b` 
    for example) are access via ``layer_instance._W`` and 
    ``layer_instance._b``.  While those are the variables of the layer (given
    at initialization of created), the ones used for the forward computation
    are ``layer_instance.W`` and ``layer_instance.b``. In fact, those are the
    ones that have the (optional) weight function applied onto them. For
    example, ``layer_instance.W=W_func(layer_instance._W)``. In fact, one can
    pass any arbitrary function to be applied onto any of the variables of the
    layer, the only requirements is that the induces parameter has the required
    shape for the forward pass. For example, if one simply wants to impose
    nonnegative weights for the slope of a :py:class:`sknet.layer.Dense` 
    layer ::
    
        layer = sknet.layer.Dense(previous_layer,units=20, W_func=tf.nn.relu) 
        
    or with a more conventional case of imposing unit
    norm (exact, not as a penalty) to the weights ::

        # define the function to be applied onto the weights to have unit norm
        renorm_func = lambda W:W/tf.sqrt(tf.reduce_sum(tf.square(W)))
        layer = sknet.layer.Dense(previous_layer,units=20,W_func=renorm_func)

    .. rubric:: Variables initialization:

    The _variables are initialized by the layer based on the values given
    for :py:data:`W` and :py:data:`b` during initialization or created is
    none is passed. In ALL cases, if
    the passed argument is an actual variable with values, s.a. a 
    :py:class:`np.ndarray`, a :py:class:`tf.Tensor` etc, then it is considered
    as a given value and not an initialization to the learnable parameter. If a
    function is given, then the constructor calls it with the shape of the
    parameter in question as input to obtain an actual initial value for the
    learnable parameter. This makes it easy to set a parameter learnable or not
    as for example::

        # Random normal (gaussian) initialization, learnable weights,
        # the layer W will be turned into a trainable tf.Variable
        sknet.layer.Dense(previous_layer,units=10,W=tf.random_normal)
        # Random normal weights that won't be learned, the passed value
        # is used as the actual parameter without any transformation
        sknet.layer.Dense(previous_layer,units=10,W=tf.random_normal((764,10)))
        # this allow to use the following case where one pre-determines a 
        # special form for the weights, and thus do not want this to be used
        # just as an initializer of a tf.Variable, but the actual weights to
        # be used by the layer
        W1 = tf.Variable(tf.random_normal((764,10))
        W2 = tf.Variable(tf.random_normal((764,10))
        layer = sknet.layer.Dense(previous_layer,units=10,W=W1*W2)

    Note that if the input shape of the layer is unknown, one can retreive it 
    via ``previous_layer.shape``. This also allows to use a 
    :py:class:`sknet.ObservedVariable` as a layer variable, allowing to set
    manually some values of the variable values hwen desired. For example::

        # set some weights that won't be learned
        fixed_W = tf.random_normal((764,10))
        # set the standard learnable weights
        learnable_W = tf.Variable(tf.random_normal((764,10)))
        # create the W variables used by the layer as an ObservedTensor
        W = sknet.ObservedTensor(learnable_W,observed=True,observation=fixed_W)
        # create the layer
        layer = sknet.layer.Dense(previous_layer,units=10,W=W)
        # this layer will thus either use the (possibly learned) value from
        # learnable_W or the fixed (random but not learned) value from fixed_W
        # depending on the value feed to W.teacher_forcing


    recall that :py:data:`layer` which is a layer also behaves as a tensor 
    and thus can be used directly as the term to be used for hte loss
    computation. In this case, we did not create a target variable for the
    :py:data:`p` distribution and thus set it to :py:data:`None` to allow
    the loss to create it itself. For training, one would have to
    feed the true labels for ``layer.cross_entropy.p`` since it is now
    a placeholder.


    Parameters
    ----------

    input : tf.Tensor or tf.Op
        the input to the layer, can eb any arbitrary variable or layer instance
        this allows the user to feed directly a tensorflow variable as input

    """
    def __init__(self,input, deterministic=None, deterministic_behavior=True,
                            **kwargs):
        
        # Link this tensor to its input
        self._input = input

        # set deterministic
        if deterministic_behavior:
            if deterministic is None:
                self.deterministic = tf.placeholder(tf.bool,
                                                name='deterministic')
            else:
                self.deterministic = deterministic
        # compute the layer output based on the layer forward method
        output = self.forward(input,self.deterministic)
        # initialize the variable
        super().__init__(output)


    @property
    def input(self):
        return self._input

    @property
    def params(self):
        if hasattr(self,'_params'):
            return self._params
        else:
            return []
    @property
    def updates(self):
        if hasattr(self,'_updates'):
            return self._updates
        else:
            return []



