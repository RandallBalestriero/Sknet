#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Layer



class Reshape(Layer):
    def __init__(self,incoming,new_shape,**kwargs):
        super().__init__(incoming)
        self.out_shape = new_shape
        self.reshape   = lambda x:tf.reshape(x,new_shape)
        if self.given_input:
            self.forward(incoming.output)
    def forward(self,input,**kwargs):
        self.output = self.reshape(input)
        self.VQ = None
        return self.output


class Stack(Layer):
    def __init__(self,incomings,axis,**kwargs):
        super().__init__(incomings[0])
        N = len(incomings)
        self.out_shape = [s if i not in axis else N for i,s in enumerate(self.in_shape)]
        self.axis      = axis
        self.stack     = lambda xs:tf.stack(xs,axis)
        if self.given_input:
            self.forward([incoming.output for incoming in incomings])
    def forward(self,input,**kwargs):
        self.output = self.stack(input)
        self.VQ     = None
        return self.output


class Concat(Layer):
    def __init__(self,incomings,axis,**kwargs):
        super().__init__(incomings[0])
        N = len(incomings)
        self.out_shape = [s if i not in axis else N*s for i,s in enumerate(self.in_shape)]
        self.axis      = axis
        self.concat    = lambda xs:tf.concat(xs,axis)
        if self.given_input:
            self.forward([incoming.output for incoming in incomings])
    def forward(self,inputs,**kwargs):
        self.output = self.concat(inputs)
        self.VQ     = None
        return self.output



class Merge(Layer):
    def __init__(self,incomings,op,**kwargs):
        super().__init__(incomings[0])
        N = len(incomings)
        self.out_shape = self.in_shape
        self.op        = op
        self.merge     = lambda xs:op(xs)
        if self.given_input:
            self.forward([incoming.output for incoming in incomings])
    def forward(self,inputs,**kwargs):
        self.output = self.merge(inputs)
        self.VQ     = None
        return self.output




class ExpandDim(Layer):
    def __init__(self,incoming,axis,**kwargs):
        super().__init__(incomings[0])
        N = len(incoming)
        if not isscalar(axis):
            assert len(axis)==1
            axis = axis[0]
        self.out_shape = self.in_shape.insert(1,axis)
        self.axis      = axis
        self.expand_dim= lambda x:tf.expand_dims(x,axis)
        if self.given_input:
            self.forward(incoming.output)
    def forward(self,input,**kwargs):
        self.output = self.expand_dim(input)
        self.VQ     = None
        return self.output





