#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from . import Layer

from .. import ops

class Conv2D(Layer):
    _ops = [ops.Conv2D,ops.BatchNorm,ops.Activation]

class Conv2DPool(Layer):
    _ops = [ops.Conv2D,ops.BatchNorm,ops.Activation,ops.Pool2D]



def custom_layer(*operators):
    class custom(Layer):
        _ops = operators
    return custom

