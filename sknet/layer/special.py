#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Layer


class input(Layer):
    def __init__(self,in_shape,x=None, data_format='NCHW'):
        """
        2D pooling layer
        Performs 2D mean- or max-pooling over the two trailing axes of a 4D input
        tensor. This is an alternative implementation which uses
        ``theano.sandbox.cuda.dnn.dnn_pool`` directly.
        Parameters
        ----------
        incoming : a :class:`Layer` instance or tuple
            The layer feeding into this layer, or the expected input shape.
        pool_size : integer or iterable
            The length of the pooling region in each dimension. If an integer, it
            is promoted to a square pooling region. If an iterable, it should have
            two elements.
        stride : integer, iterable or ``None``
            The strides between sucessive pooling regions in each dimension.
            If ``None`` then ``stride = pool_size``.
        pad : integer or iterable
            Number of elements to be added on each side of the input
            in each dimension. Each value must be less than
            the corresponding stride.
        ignore_border : bool (default: True)
            This implementation never includes partial pooling regions, so this
            argument must always be set to True. It exists only to make sure the
            interface is compatible with :class:`lasagne.layers.MaxPool2DLayer`.
        mode : string
            Pooling mode, one of 'max', 'average_inc_pad' or 'average_exc_pad'.
            Defaults to 'max'.
        **kwargs
            Any additional keyword arguments are passed to the :class:`Layer`
            superclass.
        Notes
        -----
        The value used to pad the input is chosen to be less than
        the minimum of the input, so that the output of each pooling region
        always corresponds to some element in the unpadded input region.
        This is a drop-in replacement for :class:`lasagne.layers.MaxPool2DLayer`.
        Its interface is the same, except it does not support the ``ignore_border``
        argument."""
        self.data_format = data_format
        self.out_shape   = in_shape
        if x is not None:
            self.given_input = True
            self.forward(x)
        else:
            self.given_input = False
    def forward(self,input,**kwargs):
        self.output = input
        return input



class output(Layer):
    def __init__(self, incoming, classes,
                init_W = tfl.xavier_initializer(uniform=True), name=''):
        super().__init__(incoming)
        
        # Set up the input, flatten if needed
        if len(self.in_shape)>2:
            self.flatten_input = True
            flat_dim = np.prod(self.in_shape[1:])
        else:
            self.flatten_input = False
            flat_dim = self.in_shape[1]
            
        # Output Shape
        self.out_shape = (self.in_shape[0],classes)
        
        # initialize Parameters
        self.W = tf.Variable(init_W((flat_dim,classes)), 
                            name='denselayer_W_'+name, 
                            trainable=True)
        self.b = tf.Variable(tf.zeros((1,classes)),
                            name='denselayer_b_'+name, 
                            trainable=True)
        if self.given_input:
            self.forward(incoming.output)
#    def backward(self,output):
#        """output is of shape (batch,n_output)
#        return of this function is of shape [(batch,in_dim),(batch,1)]"""
#        # use tf.nn.conv2d_backprop_input for conv2D
#        A = tf.reshape(tf.matmul(output*self.mask*scaling,self.W,transpose_b=True),incoming.out_shape)
#        B = tf.matmul(output*self.mask,self.b,transpose_b=True)
#        return A,B
    def forward(self,input,**kwargs):
        if self.flatten_input:
            input = tf.layers.flatten(input)
        self.S      = tf.matmul(input,self.W)+self.b
        self.VQ     = tf.argmax(self.S,1)
        self.output = self.S
        return self.output

