#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from . import Model
from .. import layer

class augmentation(Model):
    def __init__(self,input_shape,crop,crop_size,left_right,
            up_down,rot90,data_format,**kwargs):
        """Start the minimal.

        :param qux: The first argument to initialize class.
        :type qux: string
        :param spam: Spam me yes or no...
        :type spam: bool

        """
        super().__init__(input_shape,1,**kwargs)
        self.name      = '-model(augmentation)'
        self.crop      = crop
        self.crop_size = crop_size
        self.data_format = data_format
        self.left_right= left_right
        self.up_down   = up_down
        self.rot90     = rot90
    def get_layers(self,input_variable,training):
        """Start the Foo.

        :param qux: The first argument to initialize class.
        :type qux: string
        :param spam: Spam me yes or no...
        :type spam: bool

        """
        dnn = [layer.Input(self.input_shape,input_variable, data_format=self.data_format)]
        if self.crop:
            dnn.append(layer.RandomCrop(dnn[-1],training=training,crop_size=self.crop_size))
        if self.left_right:
            dnn.append(layer.RandomFlipLeftRight(dnn[-1],training=training))
        if self.up_down:
            dnn.append(layer.RandomFlipUpDown(dnn[-1],training=training))
        if self.rot90:
            dnn.append(layer.RandomRot90(dnn[-1],training=training))
        return dnn




