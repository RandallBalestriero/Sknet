#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
        "plotting",
        "geometry",
        "trainer"]




def init_variable(var_or_func,var_shape,name=None,trainable=True):
    if type(var_or_func)==np.ndarray:
        assert(np.isequal(var_of_func.shape,var_shape))
        return tf.Variable(var_or_func,name=name,trainable=trainable)
    elif callable(var_or_func):
        return tf.Variable(var_or_func(var_shape),name=name,trainable=trainable)
    else:
        return tf.Variable(var_or_func,name=name,trainable=False)

from . import *
