#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Layer


class Input(Layer):
    """Input layer, first of any model.
    This layer provides the interface between an input shape
    or an input variable and the layer format which can then
    be fed to any following Layer class"""
    def __init__(self,in_shape,x=None, data_format='NCHW'):
        """Initialize the layer with the given parameters

        :param in_shape: shape of the input to the layer
        :type in_shape: tuple of int
        :param x: (optional) an input Tensorflow variable. if
                  creating a model manually in a code, it is
                  safer to leave this None
        :type x: NoneType or Tensor
        :param data_format: data formatting of the input, default 
                            is :py:data:`'NCHW'`, possible values are
                            :py:data:`'NCHW', 'NHWC', 'NCT', 'NTC'` where
                            the last two ones are for time serie inputs
        :type data_format: str
        """
        self.data_format = data_format
        self.out_shape   = in_shape
        if x is not None:
            self.given_input = True
            self.forward(x)
        else:
            self.given_input = False
    def forward(self,input,**kwargs):
        """Perform forward pass given an input Tensorflow variable
        
        :param input: An input variable
        :type input: Tensor
        """
        self.output = input
        return input



class OutputClassifier(Layer):
    """Output layer for classification tasks.
    This layer implement a linear classifiers. It corresponds to a
    fully connected layer with softmax nonlinearity
    """
    def __init__(self, incoming, classes,
                init_W = tfl.xavier_initializer(uniform=True), name=''):
        """Initialize the class
        :param incoming: the incoming layer or shape
        :type incoming: Layer of tuple of int
        :param classes: the number of classes
        :type classes: int
        :param init_W:initializer for the weight
        :type init_W: initializer or Tensor or array
        """
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

