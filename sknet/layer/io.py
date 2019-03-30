#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Layer
from ..utils import init_variable as init_var
import time

class Input(Layer):
    """Input layer, first of any model.
    This layer provides the interface between an input shape
    the layer format which can then
    be fed to any following :class:`Layer` class"""
    def __init__(self, input_shape, data_format='NCHW', input=None, 
            input_dtype=tf.float32, name=''):
        """Initialize the layer with the given parameters

        :param input_shape: shape of the input to the layer
        :type input_shape: tuple of int
        :param data_format: data formatting of the input, default 
                            is :py:data:`'NCHW'`, possible values are
                            :py:data:`'NCHW', 'NHWC', 'NCT', 'NTC'` where
                            the last two ones are for time serie inputs
        :type data_format: str
        """
        self.input_dtype  = input_dtype
        self.data_format  = data_format
        self.output_shape = input_shape
        self.input_shape  = input_shape

        if input is not None:
            self.given_input = True
            self.input = input
        else:
            self.given_input = False
        
        self.name = name
    def forward(self,input=None,deterministic=None,_input=None, **kwargs):
        """Perform forward pass given an input Tensorflow variable
        
        :param input: An input variable
        :type input: Tensor
        """
        if input is None:
            self.output = _input[self]
        else:
            self.output = input
        return self.output



class OutputClassifier(Layer):
    """Output layer for classification tasks.
    This layer implement a linear classifiers. It corresponds to a
    fully connected layer with softmax nonlinearity
    """
    def __init__(self, incoming, classes,
                init_W = tfl.xavier_initializer(uniform=True),
                init_b = tf.zeros, name=''):
        """Initialize the class
        :param incoming: the incoming layer or shape
        :type incoming: Layer of tuple of int
        :param classes: the number of classes
        :type classes: int
        :param init_W: initializer for the weight
        :type init_W: initializer or Tensor or array
        :param init_b: initializer for the bias
        :type init_b: initializer or Tensor or array
        """
        super().__init__(incoming)
        self.classes = classes
        self.init_W  = init_W
        self.init_b  = init_b
        self.initialized = False
        # Set up the input, flatten if needed
        if len(self.input_shape)>2:
            self.flatten_input = True
            self.flat_dim      = np.prod(self.input_shape[1:])
        else:
            self.flatten_input = False
            self.flat_dim      = self.input_shape[1]
            
        # Output Shape
        self.output_shape = (self.input_shape[0],classes)

        #If already with a variable, init the weights
        if self.given_input:
            self.initialize_variables()
    def initialize_variables(self):
        # initialize Parameters
        if not self.initialized:
            self.W = init_var(self.init_W,(self.flat_dim,self.classes), 
                            name='denselayer_W_'+name, 
                            trainable=True)
            self.b = init_var(self.init_b,(1,self.classes),
                            name='denselayer_b_'+name, 
                            trainable=True)
            self.initialized = True
#    def backward(self,output):
#        """output is of shape (batch,n_output)
#        return of this function is of shape [(batch,in_dim),(batch,1)]"""
#        # use tf.nn.conv2d_backprop_input for conv2D
#        A = tf.reshape(tf.matmul(output*self.mask*scaling,self.W,transpose_b=True),incoming.output_shape)
#        B = tf.matmul(output*self.mask,self.b,transpose_b=True)
#        return A,B
    def forward(self,input=None, deterministic=None, _input=None, **kwargs):
        """given na input tensor, produces the output, this method should be used
        when propagating an input through layers manually"""
        if input is None:
            input = self.incoming.forward(deterministic=deterministic,_input=_input)
        if self.flatten_input:
            input = tf.layers.flatten(input)
        self.output = tf.matmul(input,self.W)+self.b
        return self.output

