import tensorflow as tf
from . import Layer

import numpy as np

from . import Layer

from .. import Variable

class BatchNorm(Layer):
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
    :type incoming: shape or :class:`Layer`
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
    def __init__(self,incoming,axis,deterministic=None, beta = tf.zeros,
                gamma=tf.ones, beta_func=tf.identity, gamma_func=tf.identity,
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
        	# being normalized over and the same as the input shape for the others
            in_shape = incoming.shape.as_list()
            shape_= [s if i not in self.axis else 1 for i,s in enumerate(in_shape)]
    
    
            # Initialization beta
            if type(beta)!=Variable:
                beta = Variable(beta,name=name+'_beta')
            self._beta  = beta(shape_)
            self.beta   = beta_func(self._beta)
    
            # Initialization gamma
            if type(gamma)!=Variable:
                gamma = Variable(gamma,name=name+'_beta')
            self._gamma = gamma(shape_)
            self.gamma  = gamma_func(self._gamma)
    
            self.ema  = tf.train.ExponentialMovingAverage(decay=decay)
    
            super().__init__(incoming,deterministic)

    def forward(self,input, deterministic, **kwargs):

        # batch statistics
        mean_,var_ = tf.nn.moments(input,axes=self.axis,keep_dims=True)

        # update of the moving averages and op collection
        ema_op     = self.ema.apply([mean_,var_])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,ema_op)

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
        output = self.gamma*(input-use_mean)/use_std+self.beta
        self.b = -(self.gamma*use_mean)/use_std+self.beta
        self.A = self.gamma/use_std
        return output

