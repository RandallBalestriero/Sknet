#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np

from .. import layer

from .. import optimize



def get_tensor_dependencies(tensor):

    # If a tensor is passed in, get its op
    try:
        tensor_op = tensor.op
    except:
        tensor_op = tensor

    # Recursively analyze inputs
    dependencies = []
    for inp in tensor_op.inputs:
        new_d = get_tensor_dependencies(inp)
        non_repeated = [d for d in new_d if d not in dependencies]
        dependencies = [*dependencies, *non_repeated]

    # If we've reached the "end", return the op's name
    if tensor_op.type == 'Placeholder':
        dependencies = [tensor.name]

    # Return a list of tensor op names
    return dependencies



class Network:
    def __init__(self, layers=None, name = 'model',dataset = None ,**kwargs):
        if layers is None:
            self.layers = self.get_layers(**kwargs)
        else:
            self.layers = layers
        self.name       = name
        self.dataset    = dataset
        self.batch_size = layers[0].shape.as_list()[0]
    def __getitem__(self,key):
        return self.layers[key]
    def __len__(self):
        return len(self.layers)
    def deter_dict(self,value):
        return dict([(layer.deterministic,value) 
                for layer in self.layers if hasattr(layer,'deterministic')])



from .cnn import *


