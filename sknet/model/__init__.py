#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from .. import layer
__all__=[
        "cnn",
        "minimal"]
#        "resnet",
 #       "dense"]



class Model:
    def __init__(self,input_shape, classes=10, data_format='NCHW', **kwargs):
        """Start the Foo.

        :param qux: The first argument to initialize class.
        :type qux: string
        :param spam: Spam me yes or no...
        :type spam: bool

        """
        self.classes      = classes
        self.input_shape  = input_shape
        self.data_format  = data_format
    def get_layers(self, input_variable, training):
        """get the layers.

        :param qux: The first argument to initialize class.
        :type qux: string
        :param spam: Spam me yes or no...
        :type spam: bool

        """
        pass



class Custom(Model):
    def __init__(self,layers):
        self.layers = layers
    def get_layers(self, input_variable, training):
        for l in self.layers:
            input_variable = l.forward(input_variable,training)
        return self.layers

from . import *


