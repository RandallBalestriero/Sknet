#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from . import Layer

# ToDo: PoolLayer: optimize s.t. if it is just channel pooling you dont reshape


class Pool(Layer):
    """Pooling layer over spatial and/or channel dimensions.
        
    Example of use::

        # (3,3) max pooling with (3,3) strides
        # All ther below are equivalent
        PoolLayer(previous_layer, windows=(3,3), strides=(3,3),data_format='NCHW')
        PoolLayer(previous_layer, windows=(3,3),data_format='NCHW')
        PoolLayer(previous_layer, windows=(1,3,3), strides=(1,3,3),data_format='NCHW')
        PoolLayer(previous_layer, windows=(1,3,3),data_format='NCHW')

    Each output position :math:'[z]_{n,i,j,c}' results form pooling 
    over the corresponding region in the input

    :param incoming: input layer or input shape
    :type incoming: ranet.layer or 1D-vector of ints    
    :param windows: shape of the pooling window
    :type windows: 1D-vector of ints
    """
    def __init__(self, incoming, windows, strides=None, pool_type='MAX', 
            padding='VALID'):
        super().__init__(incoming)
        self.pool_type = pool_type
        self.padding = padding
        if strides is not None:
            assert(len(strides)==len(windows))
        else:
            strides = windows
        if self.data_format=='NCHW' and windows[0]==1:
            self.strides = strides[1:]
            self.windows = windows[1:]
        if self.data_format=='NHWC' and windows[2]==1:
            self.strides = strides[:-1]
            self.windows = windows[:-1]
        if self.data_format=='NCHW':
            if len(self.windows)==2:
                h = 1+(self.in_shape[2]-self.windows[0])//self.strides[0]
                w = 1+(self.in_shape[3]-self.windows[1])//self.strides[1]
                self.out_shape = [self.in_shape[0],self.in_shape[1],h,w]
            else:
                c = 1+(self.in_shape[1]-self.windows[0])//self.strides[0]
                h = 1+(self.in_shape[2]-self.windows[1])//self.strides[1]
                w = 1+(self.in_shape[3]-self.windows[2])//self.strides[2]
                self.out_shape = [self.in_shape[0],c,h,w]
        else:
            if len(self.windows)==2:
                h = 1+(self.in_shape[1]-self.windows[0])//self.strides[0]
                w = 1+(self.in_shape[2]-self.windows[1])//self.strides[1]
                self.out_shape = [self.in_shape[0],self.in_shape[1],h,w]
            else:
                c = 1+(self.in_shape[3]-self.windows[2])//self.strides[2]
                h = 1+(self.in_shape[1]-self.windows[0])//self.strides[0]
                w = 1+(self.in_shape[2]-self.windows[1])//self.strides[1]
                self.out_shape = [self.in_shape[0],c,h,w]
        if self.given_input:
            self.forward(incoming.output)
    def forward(self,input,**kwargs):
        if len(self.windows)==2:
            # This is the standard spatial pooling case
            self.output = tf.nn.pool(input,window_shape=self.windows,
                    strides=self.strides, pooling_type=self.pool_type, 
                    padding=self.padding, data_format=self.data_format)
        else:
            self.output = tf.nn.pool(tf.expand_dims(input,-1),
                    window_shape=self.windows,strides=self.strides, 
                    pooling_type=self.pool_type, padding=self.padding, 
                    data_format=self.data_format)[...,0]
        # Set-up the the VQ
        if self.pool_type=='AVG':
            self.VQ = None
        else:
            self.VQ = tf.gradients(self.output,input,tf.ones_like(self.output))[0]
        return self.output


class GlobalSpatialPool(Layer):
    """pool over all spatial dimension
    This layer implements a global spatial pooling removing all spatial
    dimension present in the input.

    :param incoming: the input shape or an incoming layer
    :type incoming: shape or :class:`Layer`
    :param pool_type: the type of pooling to apply, can be either 
                      :py:data:`"MAX"`, or :py:data:`"AVG"`
    :type pool_type: str
    """
    def __init__(self,incoming, pool_type='AVG'):
        super().__init__(incoming)
        self.pool_type = pool_type
        if self.data_format=='NCHW':
            self.axis = [2,3]
        else:
            self.axis = [1,2]
        # Set up output shape
        if self.data_format=='NCHW':
            self.out_shape = [self.in_shape[0],self.in_shape[1],1,1]
        else:
            self.out_shape = [self.in_shape[0],1,1,self.in_shape[1]]
        if self.given_input:
            self.forward(incoming.output)
    def forward(self,input,**kwargs):
        if self.pool_type=='AVG':
            self.output = tf.reduce_mean(input,axis=self.axis,keep_dims=True)
            self.VQ = None
        else:
            self.output = tf.reduce_max(input,axis=self.axis,keep_dims=True)
            self.VQ = tf.gradients(self.output,input,tf.ones_like(self.output))[0]
        return self.output







