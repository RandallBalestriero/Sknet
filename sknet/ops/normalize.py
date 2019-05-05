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

    name = 'BatchNormOp'
    deterministic_behavior = True

    def __init__(self,incoming,axis,deterministic=None, b = tf.zeros,
                W=tf.ones, W_func=tf.identity, b_func=tf.identity,
                name='bn_layer', epsilon=1e-4, decay=0.9, trainable_W = True,
                trainable_b = True, **kwargs):

        with tf.variable_scope(type(self).name) as scope:
            self.name    = scope.original_name_scope
            self.epsilon = epsilon
            self.decay   = decay
            self.axis    = [axis] if np.isscalar(axis) else axis

            # Infer the shape of the parameters, it is 1 for the axis that are
            # being normalized over and the same as the input shape for 
            # the others
            in_shape = incoming.shape.as_list()
            shape_   = [s if i not in self.axis else 1
                                        for i,s in enumerate(in_shape)]


            # Initialization W (a.k.a gamma)
            if callable(W):
                self._W = Variable(W(shape_), trainable=trainable_W, name='W')
            else:
                self._W = Variable(W, trainable=trainable_W, name='W')
            self.add_param(self._W)
            self.W = W_func(self._W)

            # Initialization b (a.k.a beta)
            if callable(b):
                self._b = Variable(b(shape_), trainable=trainable_b, name='b')
            else:
                self._b = Variable(b, trainable=trainable_b, name='b')
            self.add_param(self._b)
            self.b  = b_func(self._b)

            # Steps
            self.steps = Variable(tf.zeros((1,)),trainable=False,name='step')
            self.add_param(self.steps)
            self._updates.append(tf.assign_add(self.steps,1.))

            super().__init__(incoming,deterministic)

    def forward(self,input, deterministic, **kwargs):

        # batch statistics
        mean_,var_ = tf.nn.moments(input,axes=self.axis,keep_dims=True)

        # update of the moving averages and updates/params collection
        if self.decay=='AVG':
            mean_ema,mean_ema_op=EMA(mean_, self.steps)
            var_ema,var_ema_op  =EMA(var_, self.steps)
        else:
            mean_ema,mean_ema_op=EMA(mean_, self.decay, self.steps)
            var_ema,var_ema_op  =EMA(var_, self.decay, self.steps)
        self._updates.append(mean_ema_op)
        self._updates.append(var_ema_op)
        self.add_param(mean_ema)
        self.add_param(var_ema)

        # function, context dependent to get stat to use
        use_mean,use_var = tf.cond(deterministic,lambda :mean_,var_,
                                                 lambda :mean_ema,var_ema)
        use_std = tf.sqrt(use_var)+self.epsilon

        # we also compute those quantities that allow to rewrite the output as
        # A*input+b, this might be needed for some implementation/models
        output = self.W*(input-use_mean)/use_std+self.b
        self.b = -(self.W*use_mean)/use_std+self.b
        self.A = self.W/use_std
        return output

    def backward(self,input):
        return input*self.A

