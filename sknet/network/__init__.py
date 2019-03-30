#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from .. import layer

__all__=["cnn"]


class Network:
    def __init__(self, layers=None, name = 'model', **kwargs):
        if layers is None:
            self.layers = self.get_layers(**kwargs)
        else:
            self.layers  = layers
        self.name         = name
    def get_layers(self):
        return self.layers
    def initialize_variables(self):
        # ensure that all layers are linked or not with a given input layer
        # also check if one and only one output layer
        print('Initializing the model')
        given_input = list()
        outputs = 0
        for layer_ in self.layers:
            given_input.append(layer_.given_input)
        if len(set(given_input))>1:
            print("Not all given layers are correctly linked")
            exit()

        # now create all the variables
        self.deterministic = tf.placeholder(tf.bool,name='deterministic')
        self.input         = list()
        self._input        = dict()
        # initialize all the layer variables
        # and create the inputs if needed
        for layer_ in self.layers:
            layer_.initialize_variables()
            if isinstance(layer_,layer.Input):
                if layer_.given_input:
                    self.input = layer_input
                else:
                    self.input.append(tf.placeholder(layer_.input_dtype, shape=layer_.input_shape,name='x'))
                self._input[layer_] = self.input[-1]

        # remove the list type if only one input
        if len(self.input)==1:
            self.input = self.input[0]

        # now get the output layer
        # if we have to use the last, we directly extract it
        # otherwise we search for the output layer in the given
        # collection of layers
        output_layer = self.layers[-1]

        # now we get the output from the output layer
        # if laready computed because the input was given then
        # we just get the layer.output otherwise use the newly
        # created inputs to get the layer forward method
        if layer_.given_input:
            self.output = layer_.output
        else:
            self.output = layer_.forward(deterministic=self.deterministic,_input=self._input)




from . import *


