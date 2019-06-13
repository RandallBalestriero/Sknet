#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from ..ops import Op

class Network:

    def __init__(self, layers=[], name='model'):
        self.name = name
        self.layers = layers
        self._deter_dict = dict()
    def __getitem__(self, key):
        if isinstance(key, slice):
            return Network(layers=self.layers[key], name='sub'+self.name)
        return self.layers[key]

    def __len__(self):
        return len(self.layers)

    def append(self, item):
        """append an additional layer to the current network"""
        if hasattr(item, '__len__'):
            item = list(item)
            self.layers += item
        else:
            self.layers.append(item)
        self.update_deter_dict()
    def as_list(self):
        """return the layers as a list"""
        return [layer for layer in self]

    def update_deter_dict(self):
        """gather all the per layer deterministic variables and
        create a dictionary mapping those variables to value"""
        for layer in self.layers:
            if isinstance(layer, Op):
                self._deter_dict.update(layer.deter_dict(True))

    def deter_dict(self, value):
        for key in self._deter_dict.keys():
            self._deter_dict[key] = value
        return self._deter_dict

    @property
    def shape(self):
        """return the list of shapes of the feature maps for all the
        layers currently in the network."""
        return [i.get_shape().as_list() for i in self]

    @property
    def reset_variables_op(self, group=True):
        """gather all the reset variables op of each of the layers
        and group them into a single op if group is True, or
        return the list of operations"""
        var = []
        for layer in self:
            if hasattr(layer, 'variables'):
                var.append(layer.reset_variables_op)
        if group:
            return tf.group(*var)
        return var

    def variables(self, trainable=True):
        """return all the variables of the network
        which are trainable only or all"""
        var = list()
        for layer in self:
            if hasattr(layer, 'variables'):
                var += layer.variables(trainable)
        return var

    def backward(self, tensor):
        """feed the tensor backward in the network by
        successively calling each layer backward method,
        from the last layer to the first one. Usefull when
        doing backpropagation to get the gradient w.r.t. the input"""
        ops = self.as_list()[::-1]
        for op in ops:
            tensor = op.backward(tensor)
        return tensor

    @property
    def updates(self):
        """gather all the network updates that are
        present in the layer (for example the passive
        update of the batch-normalization layer"""
        updates = []
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates
        return updates
