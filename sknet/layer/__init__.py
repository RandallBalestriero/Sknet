#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Layer:
    def __init__(self,incoming,**kwargs):
        # Case where incoming is a layer
        if isinstance(incoming,Layer):
            self.input_shape = incoming.output_shape
            self.data_format = incoming.data_format
            self.known_input = True
            self.incoming    = incoming
            self.given_input = incoming.given_input
        else:
            self.input_shape = incoming
            self.known_input = False
            self.data_format = kwargs["data_format"]
            self.given_input = False
    def initialize_variables(self):
        pass
    def forward(self,input,training=None):
        pass
    def backward(self,output):
        pass




from .pool import *
from .perturb import *
from .normalize import *
from .conv import *
from .dense import *
from .shape import *
from .io import *
from .special import *
from .meta import *
