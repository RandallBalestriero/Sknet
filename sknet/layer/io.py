#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Layer


class Input(Layer):
    def __init__(self,in_shape,x=None, data_format='NCHW'):
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



class Output(Layer):
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

