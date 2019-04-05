#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from .. import layer


class Network:
    def __init__(self, layers=None, name = 'model', **kwargs):
        if layers is None:
            self.layers = self.get_layers(**kwargs)
        else:
            self.layers = layers
        self.name     = name
#        self.init_values()
#        self.reset_op = tf.group([layer.reset_op for layer in self.layers])
    def __getitem__(self,key):
        return self.layers[key]
    def __len__(self):
        return len(self.layers)
    def get_layers(self):
        """To implement model specific list of layers 
        that has to be passed to the network
        """
        pass
    @property
    def loss(self):
        layer_losses = list()
        for layer in self.layers:
            if hasattr(layer,'losses'):
                layer_losses+=layer.losses
        return tf.add_n([layer_loss.loss for layer_loss in layer_losses])
    def set_deterministic(self,value,session=None):
        for layer in self:
            if hasattr(layer,'set_deterministic'):
                layer.set_deterministic(value,session)
    def init_values(self):
        inputs   = list()
        infered_observed = list()
        for layer_ in self.layers:
            if isinstance(layer_,layer.Input):
                inputs.append(layer_.input)
            else:
                if layer_.observed:
                    infered_observed.append([layer_,layer_.observation])

        if len(inputs)==1:
            self.input = inputs[0]
        else:
            self.input = inputs
        self.output = self.layers[-1]
        self.infered_observed = infered_observed



from .cnn import *


