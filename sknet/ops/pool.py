#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from . import Op

# ToDo: PoolOp: optimize s.t. if it is just channel pooling you dont reshape


class GlobalPool2D(Op):
    """Pooling layer over spatial and/or channel dimensions.

    Example of use::

        # (3,3) max pooling with (3,3) stride
        # All ther below are equivalent
        PoolOp(previous_layer, window=(3,3), stride=(3,3))
        PoolOp(previous_layer, window=(3,3))
        PoolOp(previous_layer, window=(1,3,3), stride=(1,3,3))
        PoolOp(previous_layer, window=(1,3,3))
        # Channel pooling (only)
        PoolOp(previous_layer, window=(4,1,1))
        # Channel (with overlap) and Spatial Pooling
        PoolOp(previous_layer, window=(4,2,2), stride=(2,2,2))


    Each output position :math:'[z]_{n,i,j,c}' results form pooling 
    over the corresponding region in the input

    Parameters
    ----------

    incoming : tf.Tensor or sknet.Op
        The incoming tensor or layer instance

    window : list of int
        The size of the pooling window

    stride : list of int (default=window)
        The stride of the pooling

    """

    name = 'GlobalPool2D'
    deterministic_behavior = False

    def __init__(self, incoming, pool_type='AVG',
                            keepdims=False,*args, **kwargs):
        with tf.variable_scope("Pool") as scope:
            self.scope_name = scope.original_name_scope
            self.pool_type = pool_type
            self.keepdims = keepdims
            super().__init__(incoming)

    def forward(self, input, *args, **kwargs):
        # This is the standard spatial pooling case
        if self.pool_type=='AVG':
            return tf.reduce_mean(input,[2,3],keepdims=self.keepdims)
        elif self.pool_type=='MAX':
            return tf.reduce_max(input,[2,3],keepdims=self.keepdims)
        else:
            return self.pool_type(input)
    def backward(self,input):
        return tf.gradient(self,self.input,input)[0]



class Pool2D(Op):
    """Pooling layer over spatial dimensions.
        
    Example of use::

        # (3,3) max pooling with (3,3) stride
        # All ther below are equivalent
        Pool2D(previous_layer, window_shape=(3,3), strides=(3,3))
        Pool2D(previous_layer, window_shape=(3,3))
        # Spatial Pooling with overlap
        Pool2D(previous_layer, window_shape=(5,5), strides=(2,2))

    Each output position :math:'[z]_{n,c,i,j}' results form pooling 
    over the corresponding region in the input.

    Parameters
    ----------

    incoming : tf.Tensor or sknet.Op
        The incoming tensor or layer instance

    window_shape : list of int
        The size of the pooling window

    strides : list of int (default=window)
        The stride of the pooling

    pool_type : str
        The pooling to use, can be :var:`"MAX"` or :var:`"AVG"`.

    padding : str
        The padding to use, cane be :var:`"VALID"` or :var:`"SAME"`

    """

    _name_ = 'Pool2DOp'
    deterministic_behavior = False

    def __init__(self, incoming, window_shape, strides=None, pool_type='MAX',
                    padding='VALID', *args, **kwargs):
        assert(len(window_shape)==len(incoming.shape)-2)
        with tf.variable_scope(self._name_) as scope:
            self._name        = scope.original_name_scope
            self.pool_type    = pool_type
            self.padding      = padding
            self.window_shape = window_shape
            self.strides      = window_shape if strides is None else strides
            super().__init__(incoming)

    def forward(self, input, *args, **kwargs):
        output = tf.nn.pool(input,window_shape=self.window_shape,
                    strides=self.strides, pooling_type=self.pool_type,
                    padding=self.padding, data_format='NCHW')

        # Set-up the the VQ
        if self.pool_type=='MAX':
            mask = tf.gradients(output,input,tf.ones_like(output))[0]
            self.mask = tf.cast(mask,tf.bool)
        return output

    def backward(self,input):
        return tf.gradient(self, self.input, input)[0]


