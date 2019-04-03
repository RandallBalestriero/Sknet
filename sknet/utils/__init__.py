#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

__all__ = [
        "plotting",
        "geometry"]



def init_variable(var_or_func,var_shape,name=None,trainable=True, as_var=True):
    if type(var_or_func)==np.ndarray:
        assert(np.isequal(var_of_func.shape,var_shape))
        if as_var:
            return tf.Variable(var_or_func,name=name,trainable=trainable)
        else:
            return var_or_func
    elif callable(var_or_func):
        if as_var:
            return tf.Variable(var_or_func(var_shape),name=name,trainable=trainable)
        else:
            return var_or_func(var_shape)
    else:
        if as_var:
            return tf.Variable(var_or_func,name=name,trainable=False)
        else:
            return var_or_func



def to_one_hot(labels,K=None):
    if K is None:
        K=int(np.max(labels)+np.min(labels))
    matrix = np.zeros((len(labels),K),dtype='float32')
    matrix[range(len(labels)),labels]=1
    return matrix


from . import *
from .pipeline import *

