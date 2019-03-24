#!/usr/bin/env python
# -*- coding: utf-8 -*-

#__all__ = [
#        "special",
#        "normalize",
#        "pool",
#        "augment",
#        "transform",
#        "nonlinearity"]


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
from .transform import *
from .special import *
