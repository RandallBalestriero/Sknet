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
        self.classes      = classes
        self.input_shape  = input_shape
        self.data_format  = data_format
    def get_layers(self, input_variable, training):
        pass



class Custom(Model):
    def __init__(self,layers):
        self.layers = layers
    def get_layers(self, input_variable, training):
        for l in self.layers:
            input_variable = l.forward(input_variable,training)
        return self.layers

from . import *


