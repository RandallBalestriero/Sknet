import tensorflow as tf
import numpy as np
from . import Op
from .. import Variable

class BatchNorm(Op):
    """applies batch-normalization onto the input
    Given some :py:data:`axis`, applies batch-normalization onto the 
    input on those axis. This applies udring training a moment based
    rneormalization, and during testing, the moving average of the
    moments encoutered during training. The moments are computed per batch

    To remove the learnable scaling or offset one can set 
    :py:data:`gamma=False` and/or :py:data:`beta=False`. 
    Even though one can let them as :py:data:`True` and give a constant
    value, this is no recommander as the implementation optimizes the
    computation otherwise.

    :param incoming: input shape or incoming layer
    :type incoming: shape or :class:`Op`
    :param axis: the axis to normalize
    :type axis: tuple or list of ints
    :param training: variable representing the state of the model, if
                     training or testing time
    :type training: tf.bool
    :param beta_initializer: initializer of the beta parameter
    :type beta_initializer: initializer or array
    :param gamma_initializer: initializer of the gamma parameter
    :type gamma_initializer: initializer or array
    :param name: name of the layer
    :type name: str
    :param epsilon: (optional) the epsilon constant to add
                    to the renormalization
    :type epsilon: scalar
    :param decay: the decay to use for the exponential
                  moving average to compute the test time
                  moments form the training batches ones
    :type decay: scalar
    """

    name='BatchNorm'
    deterministic_behavior = True

    def __init__(self,incoming,axis,deterministic=None, b = tf.zeros,
                W=tf.ones, W_func=tf.identity, b_func=tf.identity,
                name='bn_layer', epsilon=1e-4, decay=0.9, **kwargs):

        with tf.variable_scope("bn_layer") as scope:
            self.scope_name = scope.original_name_scope
            # set attributes
            self.epsilon = epsilon
            if np.isscalar(axis):
                self.axis = [axis]
            else:
                self.axis = axis

            # Infer the shape of the parameters, it is 1 for the axis that are
            # being normalized over and the same as the input shape for 
            # the others
            in_shape = incoming.shape.as_list()
            shape_= [s if i not in self.axis else 1
                                        for i,s in enumerate(in_shape)]


            # Initialization gamma
            if W is None:
                self._W = 1
            elif callable(W):
                self._W = Variable(W(shape_),name=name+'_beta')
            else:
                self._W  = W
            self.W = W_func(self._W) if W is not None else self._W
            self.add_param(self._W)

            # Initialization beta
            if b is None:
                self._b = 0
            elif callable(b):
                self._b = Variable(b(shape_),name=name+'_beta')
            else:
                self._b = b
            self.b  = b_func(self._b) if b is not None else self._b
            self.add_param(self._b)
            if decay=='AVG':
                decay_ = tf.Variable(np.float32(1.),trainable=False,name='t')
                self.ema  = tf.train.ExponentialMovingAverage(decay=1/decay_)
            else:
                self.ema  = tf.train.ExponentialMovingAverage(decay=decay)

            super().__init__(incoming,deterministic)
            if decay=='AVG':
                self._updates.append(tf.assign_add(decay_,1.))

    def forward(self,input, deterministic, **kwargs):

        # batch statistics
        mean_,var_ = tf.nn.moments(input,axes=self.axis,keep_dims=True)

        # update of the moving averages and op collection
        ema_op = self.ema.apply([mean_,var_])
        self._updates = [ema_op]
#        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,ema_op)

        # function, context dependent
        def not_deter():
            return mean_,var_
        def deter():
            return self.ema.average(mean_),self.ema.average(var_)

        # get stat based on context
        use_mean,use_var = tf.cond(deterministic,deter,not_deter)
        use_std = tf.sqrt(use_var)+self.epsilon

        # we also compute those quantities that allow to rewrite the output as
        # A*input+b, this might be of use for research pruposes
        output = self.W*(input-use_mean)/use_std+self.b
        self.b = -(self.W*use_mean)/use_std+self.b
        self.A = self.W/use_std
        return output

    def backward(self,input):
        return input*self.A

