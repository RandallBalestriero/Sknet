#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from . import ops

def Conv2D(input, filters, axis, nonlinearity, strides=1):
    conv = ops.Conv2D(input, filters=filters, strides=strides, b=None)
    bn = ops.BatchNorm(conv, axis)
    nonlinearity = ops.Activation(bn, nonlinearity)
    return [conv, bn, nonlinearity]

def Conv2DPool(input, filters, axis, nonlinearity, pool_shape, strides=1):
    conv = ops.Conv2D(input, filters=filters, strides=strides, b=None)
    bn = ops.BatchNorm(conv, axis)
    nonlinearity = ops.Activation(bn, nonlinearity)
    pool = ops.Pool2D(nonlinearity, pool_shape)
    return [conv, bn, nonlinearity, pool]


