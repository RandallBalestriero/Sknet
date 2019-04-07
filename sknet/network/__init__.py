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
        dependencies = [tensor_op]

    # Return a list of tensor op names
    return dependencies



class Network:
    def __init__(self, layers=None, name = 'model', **kwargs):
        if layers is None:
            self.layers = self.get_layers(**kwargs)
        else:
            self.layers = layers
        self.name     = name
#        self.losses   = 
#        self.init_values()
#        self.reset_op = tf.group([layer.reset_op for layer in self.layers])
        self.losses = dict()
        self.variables = dict()
    def __getitem__(self,key):
        return self.layers[key]
    def __len__(self):
        return len(self.layers)
    def get_layers(self):
        """To implement model specific list of layers 
        that has to be passed to the network
        """
        pass
    def add_linkage(self,linkage):
        self._linkage = linkage
    @property
    def linkage(self):
        return self._linkage
    def add_loss(self,loss,name,optimizer,schedule=0.01):
        if np.isscalar(schedule):
            schedule = optimize.schedule.constant(schedule)
        self.losses[name]=(loss,name,optimizer,schedule)
    def add_variable(self,variable,name):
        self.variables[name]=variable

#    @property
#    def loss(self):
#        layer_losses = list()
#        for layer in self.layers:
#            if hasattr(layer,'losses'):
#                layer_losses+=layer.losses
#        return tf.add_n([layer_loss.loss for layer_loss in layer_losses])
    def set_deterministic(self,value,session=None):
        for layer in self:
            if hasattr(layer,'set_deterministic'):
                layer.set_deterministic(value,session)
    def _init_dependencies(self):
        # init the dependencies
        dict_ = list()
        for var in :
            if isinstance(var,tf.Tensor):
                dict_.append((var,get_tensor_dependencies(var)))
            else:
                dependencies = list()
                for inner_var in var:
                    dependencies.append(get_tensor_dependencies(var))
                # ensure unique
                dependencies = list(set(dependencies))
                dict_.append((var,dependencies))
        self._dependencies = dict(dict_)

    def init_values(self):
        # init the dependencies
        self._init_dependencies()


        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        for loss in self.losses.values():
            with tf.control_dependencies(update_op):
                loss_    = loss[0]
                train_op = loss[-2].minimize(loss_)

                dependencies = get_tensor_dependencies(loss_)





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


