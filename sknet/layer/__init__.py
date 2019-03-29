#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Layer:
    def __init__(self,incoming,**kwargs):
        # By default, assumes no input was given
        if isinstance(incoming,Layer):
            self.in_shape    = incoming.out_shape
            self.given_input = incoming.given_input
            self.data_format = incoming.data_format
        else:
            self.in_shape    = incoming
            self.given_input = False
    def forward(self,input,training=None):
        pass
    def backward(self,output):
        pass


from .pool import *
from .augment import *
from .normalize import *
from .conv import *
from .dense import *
from .shape import *
from .io import *
from .special import *

