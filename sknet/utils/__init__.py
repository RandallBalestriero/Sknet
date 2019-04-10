#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

__all__ = [
        "plotting",
        "geometry"]



def init_variable(var_or_func,var_shape,name=None):
    if callable(var_or_func):
        return tf.Variable(var_or_func(var_shape),name=name,trainable=True)
    else:
        return var_or_func



def to_one_hot(labels,K=None):
    if K is None:
        K=int(np.max(labels)+np.min(labels))
    matrix = np.zeros((len(labels),K),dtype='float32')
    matrix[range(len(labels)),labels]=1
    return matrix


from . import *
from .workplace import *

