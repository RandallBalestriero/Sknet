#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Layer



class Reshape(Layer):
    """reshape of the input accoridng to a given shape.
    This layer allows to transform the shape of an input
    by reshaping the input according to a given shape
    
    :param incoming: input shape of incoming :class:`Layer` object
    :type incoming: Layer or tuple of int
    :param new_shape: the shape to use to reshape the input
    :type new_shape: tuple of int
    """
    def __init__(self,incoming,new_shape,**kwargs):
        super().__init__(incoming)
        self.out_shape = new_shape
        self.reshape   = lambda x:tf.reshape(x,new_shape)
        if self.given_input:
            self.forward(incoming.output)
    def forward(self,input=None, training=None, **kwargs):
        if input is None:
            input = self.incoming.forward(training=training)
        self.output = self.reshape(input)
        return self.output


class Stack(Layer):
    """stack multiple inputs into one.
    This layer stacks multiple inputs into one by stacking them along
    a specified axis.

    :param incomings: list of :class:`Layer` instances or shapes
    :type incomings: list
    :param axis: axis along which the inputs are stacked
    :type axis: int
    """
    def __init__(self,incomings,axis,**kwargs):
        super().__init__(incomings[0])
        N = len(incomings)
        self.out_shape = [s if i not in axis else N for i,s in enumerate(self.in_shape)]
        self.axis      = axis
        self.stack     = lambda xs:tf.stack(xs,axis)
        if self.given_input:
            self.forward([incoming.output for incoming in incomings])
    def forward(self,input=None, training=None, **kwargs):
        if input is None:
            input = self.incoming.forward(training=training)
        self.output = self.stack(input)
        self.VQ     = None
        return self.output


class Concat(Layer):
    """concatenate multiple inputs into one according to a given axis.
    This layer allows to take multiple inputs and concatenates them
    along a given :py:data:`axis`.
    The inputs must have same shape for all the dimension other
    than :py:data:`axis`.

    :param incomings: the list of :class:`Layer` or input shapes
                      to be concatenated
    :type incomings: list
    :param axis: the axis to concatenate over with
    :type axis: int
    """
    def __init__(self,incomings,axis,**kwargs):
        super().__init__(incomings[0])
        N              = len(incomings)
        self.incomings = incomings
        self.out_shape = [s if i not in axis else N*s for i,s in enumerate(self.in_shape)]
        self.axis      = axis
        self.concat    = lambda xs:tf.concat(xs,axis)
        if self.given_input:
            self.forward([incoming.output for incoming in incomings])
    def forward(self,inputs=None, training=None, **kwargs):
        if inputs is None:
            inputs = [incoming.forward(training=training) 
                            for incoming in self.incomings]
        self.output = self.concat(inputs)
        return self.output



class Merge(Layer):
    """merge multiple inputs into one according to some function.
    This layer allows to take multiple inputs and aggregate them (merge them)
    into a single one via some function with signature 
    func(list_of_tensors)->tensor or according to elementwise sum
    or elementwise maximum. The shape of the inputs must be the same.

    Example of use::

        a_shape = [10,1,32,32]
        b_shape = [10,1,32,32]
        # element wise sum
        Merge([a_shape,b_shape],func="SUM")
        # with a predefined function for example performing softmax
        # pooling over the tensors
        def my_func(lots):
            stacked = tf.nn.softmax(tf.stack(lots,-1))
            return tf.reduce_sum(stacked,-1)
        Merge([a_shape,b_shape],func=my_func)

    :param incoming: list of input shapes or instances of :class:`Layer`
    :type incoming: list
    :param func: function to use to merge the inputs can be either 
                 a function of :py:data:`"MAX"`, :py:data:`"MIN"`, 
                 :py:data:`"AVG"`, :py:data:`"SUM"`
    :type func: func or str
    """
    def __init__(self,incomings,func,**kwargs):
        super().__init__(incomings[0])
        self.incomings = incomings
        self.N         = np.float32(len(incomings))
        self.out_shape = self.in_shape
        self.func      = func
        if self.given_input:
            self.forward([incoming.output for incoming in incomings])
    def forward(self,inputs=None, training=None, **kwargs):
        if inputs is None:
            inputs = [incoming.forward(training=training) 
                            for incoming in self.incomings]
        if type(self.func)==str:
            if self.func=='SUM':
                self.output = tf.add_n(inputs)
            elif self.func=='AVG':
                self.output = tf.add_n(inputs)/self.N
            elif self.func=='MAX':
                self.output = tf.reduce_max(tf.stack(inputs,0),0)
            elif self.func=='MIN':
                self.output = tf.reduce_min(tf.stack(inputs,0),0)
        else:
            self.output = self.func(inputs)
        return self.output




class ExpandDim(Layer):
    """add an extra dimension in the input
    This layer allows to add an extra dimension in the input specified
    by :py:data:`axis`

    example of use::

        input_shape = [10,32,32]
        layer = ExpandDim(input_shape,1)
        layer.out_shape #[10,1,32,32]

    :param incoming: the input shape of the layer
    :type incoming: shape or :class:`Layer` instance
    :param axis: the axis to expand dim on
    :type axis: int
    """
    def __init__(self,incoming,axis,**kwargs):
        super().__init__(incomings[0])
        N = len(incoming)
        if not isscalar(axis):
            assert len(axis)==1
            axis = axis[0]
        self.out_shape = self.in_shape.insert(1,axis)
        self.axis      = axis
        self.expand_dim= lambda x:tf.expand_dims(x,axis)
    def forward(self,input, training=None, **kwargs):
        if input is None:
            input = self.incoming.foward(training=training)
        self.output = self.expand_dim(input)
        return self.output





