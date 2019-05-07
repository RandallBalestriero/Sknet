import tensorflow as tf
import numpy as np
from . import Op
from .. import Variable, EMA, ONE_INT32, ZERO_INT32

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

    _name_ = 'BatchNormOp'
    deterministic_behavior = True

    def __init__(self, incoming, axis, deterministic=None, epsilon=1e-4, 
                decay=0.9, W=tf.ones, b=tf.zeros, W_func=tf.identity, 
                b_func=tf.identity, trainable_W=True, trainable_b=True, 
                **kwargs):

        with tf.variable_scope(self._name_) as scope:
            self._name   = scope.original_name_scope
            self.epsilon = tf.constant(epsilon)
            self.decay   = tf.constant(decay)
            self.axis    = [axis] if np.isscalar(axis) else axis

            # Infer the shape of the parameters, it is 1 for the axis that are
            # being normalized over and the same as the input shape for 
            # the others
            in_shape = incoming.shape.as_list()
            shape_   = [s if i not in self.axis else 1
                                        for i,s in enumerate(in_shape)]

            # Initialization W (a.k.a gamma)
            if callable(W):
                self._W = tf.Variable(W(shape_), trainable=trainable_W,name='W')
            else:
                self._W = tf.Variable(W, trainable=trainable_W, name='W')
            self.W = W_func(self._W)

            # Initialization b (a.k.a beta)
            if callable(b):
                self._b = tf.Variable(b(shape_), trainable=trainable_b,name='b')
            else:
                self._b = tf.Variable(b, trainable=trainable_b, name='b')
            self.b = b_func(self._b)

            # Steps
            self.steps = tf.Variable(-ONE_INT32, trainable=False, name='step')
            
            super().__init__(incoming,deterministic)

    def forward(self,input, deterministic, *args, **kwargs):

        step = tf.assign_add(self.steps, ONE_INT32)
        self._updates.append(step)

        # batch statistics
        mean_,var_ = tf.nn.moments(input, axes=self.axis, keep_dims=True)

        # update of the moving averages and updates/params collection
        if self.decay=='AVG':
            m_ema, mean_ema_op = EMA(mean_, tf.cast(step+ONE_INT32, tf.float32))
            v_ema, var_ema_op  = EMA(var_, tf.cast(step+ONE_INT32, tf.float32))
        else:
            m_ema, mean_ema_op = EMA(mean_, self.decay, step)
            v_ema, var_ema_op  = EMA(var_, self.decay, step)
        self._updates.append(mean_ema_op)
        self._updates.append(var_ema_op)

        # function, context dependent to get stat to use
        use_mean,use_var = tf.cond(deterministic, lambda :[m_ema,v_ema],
                                                  lambda :[mean_,var_])
        use_std = tf.sqrt(use_var)+self.epsilon

        # we also compute those quantities that allow to rewrite the output as
        # A*input+b, this might be needed for some implementation/models
        output = self.W*(input-use_mean)/use_std+self.b
        self.b = -(self.W*use_mean)/use_std+self.b
        self.A = self.W/use_std
        return output

    def backward(self,input):
        return input*self.A

