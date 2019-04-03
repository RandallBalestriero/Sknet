#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Layer
from ..utils import init_variable as init_var
import time

class Input(Layer):
    """Input layer, first of any model.
    This layer provides the interface between an input shape
    the layer format which can then
    be fed to any following :class:`Layer` class"""
    def __init__(self, input_shape, var_name=None, data_format='NCHW', input=None,
            input_dtype=tf.float32):
        """Initialize the layer with the given parameters

        :param input_shape: shape of the input to the layer
        :type input_shape: tuple of int
        :param data_format: data formatting of the input, default 
                            is :py:data:`'NCHW'`, possible values are
                            :py:data:`'NCHW', 'NHWC', 'NCT', 'NTC'` where
                            the last two ones are for time serie inputs
        :type data_format: str
        """

        if input is None:
            input = tf.placeholder(input_dtype,shape=input_shape,name=var_name)
        else:
            var_name = input.name

        super().__init__(input, deterministic=None, observed=False, 
                        observation=None, teacher_forcing=None,data_format=data_format)
        self._observation = input
        self._var_name    = var_name
    def forward(self,input,deterministic=None, **kwargs):
        """Perform forward pass given an input Tensorflow variable

        Parameters
        ----------
        input: tf.Tensor
            An input variable

        Returns
        -------
        output: tf.Tensor
            Same as input
        """
        return input




