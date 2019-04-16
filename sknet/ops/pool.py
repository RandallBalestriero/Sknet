#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from . import Op

# ToDo: PoolOp: optimize s.t. if it is just channel pooling you dont reshape


class Pool(Op):
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
    def __init__(self, incoming, window_shape, strides=None, pool_type='MAX',
                    padding='VALID', *args, **kwargs):
        with tf.variable_scope("Pool") as scope:
            self.scope_name = scope.original_name_scope
            self.pool_type = pool_type
            self.padding   = padding
            assert(len(window_shape)==len(incoming.shape)-1)
            if strides is None:
                strides = window_shape
            else:
                for i,w in enumerate(window_shape):
                    if w==-1:
                        assert(strides[i]==1)
            # DO THE CASES WHERE -1 IS PRESENT
            self.total_axis = [i for i,w in enumerate(window_shape) if w==-1]
            self.total      = len(self.total_axis)>0
            if self.total:
                self.total_func = tf.reduce_mean if pool_type=="AVG" else tf.reduce_max
                self.window_shape = [1 if i in self.total_axis else w for
                                            i,w in enumerate(window_shape)]
                self.strides = [1 if i in self.total_axis else w for
                                            i,w in enumerate(strides)]
            else:
                self.window_shape = window_shape
                self.strides      = strides
            super().__init__(incoming)

    def forward(self,input,*args,**kwargs):
        # This is the standard spatial pooling case
        if self.total:
            input = self.total_func(input,axis=self.total_axis,keepdims=True)
        # pooling also occurs on channel axis
        data_format = ["NCW","NCHW","NCDHW"]
        if self.window_shape[0]>1:
            data_format = data_format[len(input.shape)-2]
            input = tf.expand_dims(input,1)
            output = tf.nn.pool(input,window_shape=self.window_shape,
                    strides=self.strides, pooling_type=self.pool_type,
                    padding=self.padding, data_format=data_format)[:,0]
        else:
            print(self.window_shape,self.strides)
            data_format = data_format[len(input.shape)-3]
            output = tf.nn.pool(input,window_shape=self.window_shape[1:],
                    strides=self.strides[1:], pooling_type=self.pool_type, 
                    padding=self.padding, data_format=data_format)

        # Set-up the the VQ
        if self.pool_type=='AVG':
            self.VQ = None
        else:
            self.VQ = tf.gradients(output,input,tf.ones_like(output))[0]
        return output








