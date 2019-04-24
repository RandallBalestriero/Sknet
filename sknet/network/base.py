#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..ops import Op

class Network:
    def __init__(self, layers=[], name = 'model',**kwargs):
        self.name       = name
        self.layers     = layers
    def __getitem__(self,key):
        if isinstance(key,slice):
            return Network(layers=self.layers[key],name='sub'+self.name)
        return self.layers[key]
    def __len__(self):
        return len(self.layers)
    def append(self,item):
        self.layers.append(item)
    def deter_dict(self,value):
        couple = list()
        for layer in self.layers:
            if hasattr(layer,'deterministic'):
                if isinstance(layer,Op):
                    couple.append((layer.deterministic,value))
                else:
                    for d in layer.deterministic:
                        couple.append((d,value))
        return dict(couple)
    @property
    def shape(self):
        s = [i.get_shape().as_list() for i in self]
        return s
    @property
    def params(self):
        params = []
        for layer in self.layers:
            params+=layer.params
        return params
    @property
    def updates(self):
        updates = []
        for layer in self.layers:
            updates+=layer.updates
        return updates





