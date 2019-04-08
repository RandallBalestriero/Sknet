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
        self.name     = name
        self.dataset  = dataset
        self.batch_size = layers[0].shape.as_list()[0]
#        self.losses   = 
#        self.init_values()
#        self.reset_op = tf.group([layer.reset_op for layer in self.layers])
        self.losses = dict()
        self.variables = dict()
        self._dependencies = dict()
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
    def get_input_for(self,var,indices,context):
        """ given a placeholder, batch indices and a set_name (context)
        return the data corresponding to be fed for this placeholder
        extract from the given set and indices

        Parameters
        ----------

        var : tf.placeholder
            a placeholder, present in self.placeholders and that has been
            linked with a corresponding dataset in ``self.linkage``

        indices : list or array of int
            the indices of the data to be extracted

        context : the set to extract the data from
        """

        # first get the name
        try:
            var_name = self.linkage[var]
        except:
            print('beware,',var,' will need to be provided as part of')
            print('additional, linkage, it was not given with linkage,')
            print('probably because it is not a dataset based value')
        # then extract this var form the correct context and indices
        return self.dataset[var_name][context][indices]

    def set_deterministic(self,value,session=None):
        for layer in self:
            if hasattr(layer,'set_deterministic'):
                layer.set_deterministic(value,session)
    def init_dependencies(self):
        # init the dependencies
        dict_ = list()
        for var in self.variables.values():
            if isinstance(var,tf.Tensor):
                dict_.append((var,get_tensor_dependencies(var)))
                dependencies = get_tensor_dependencies(var)
            else:
                var_ = var.control_inputs
                dependencies = []
                for v in var_:
                    dependencies.append(get_tensor_dependencies(v))
                dependencies = list(set([item for sublist in dependencies for item in sublist]))
            dict_.append((var,dependencies))
        self._dependencies = dict(dict_)
    @property
    def dependencies(self):
        return self._dependencies

    def init_values(self):
        # init the dependencies
        self._init_dependencies()







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


