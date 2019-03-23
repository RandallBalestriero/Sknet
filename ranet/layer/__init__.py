#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf

__all__ = [
        "normalization",
        "pool",
        "augment",
        "transform",
        "nonlinearity"]




class Input:
    def __init__(self,input_shape,x, data_format='NCHW'):
        self.output       = x
        self.data_format  = data_format
        self.output_shape = input_shape




class Output:
    def __init__(self, incoming, classes,
                init_W = tf.contrib.layers.xavier_initializer(uniform=True), name=''):
        # Set up the input, flatten if needed
        if(len(incoming.output_shape)>2):
            inputf = tf.layers.flatten(incoming.output)
            in_dim = prod(incoming.output_shape[1:])
        else:
            inputf = incoming.output
            in_dim = incoming.output_shape[1]
        # Output Shape
        self.output_shape = (incoming.output_shape[0],classes)
        # Param Inits
        self.W            = tf.Variable(init_W((in_dim,classes)),
                                name='denselayer_W_'+name, trainable=True)
        self.bias         = tf.Variable(tf.zeros((1,classes)),
                                name='denselayer_b_'+name, trainable=True)
        Wx                = tf.matmul(inputf,self.W)
        self.S            = Wx+self.bias
        self.VQ           = tf.argmax(self.S,1)
        self.output       = self.S
#    def backward(self,output):
#        """output is of shape (batch,n_output)
#        return of this function is of shape [(batch,in_dim),(batch,1)]"""
#        # use tf.nn.conv2d_backprop_input for conv2D
#        A = tf.reshape(tf.matmul(output*self.mask*scaling,self.W,transpose_b=True),incoming.output_shape)
#        B = tf.matmul(output*self.mask,self.b,transpose_b=True)
#        return A,B



from . import *
