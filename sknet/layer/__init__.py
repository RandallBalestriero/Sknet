#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from .. import Tensor


class Layer(Tensor):
    def __init__(self,input, deterministic=None, observed=False, 
                        observation=None, teacher_forcing=None, **kwargs):
        # Case where input is a layer
        self._input = input
        if isinstance(input,Layer):
            self._data_format = input.data_format
        else:
            self._data_format = kwargs["data_format"]
        # deterministic
        if deterministic is None:
            self._deterministic = tf.Variable(True,name='deterministic',trainable=False)
        else:
            self._deterministic = deterministic
        self._set_deterministic     = tf.assign(self._deterministic,True)
        self._set_not_deterministic = tf.assign(self._deterministic,False)
        self._reset_variables = []
        output = self.forward(input,deterministic)
        super().__init__(output, observed=observed, observation=observation,
                            teacher_forcing=teacher_forcing)

    def forward(self,input,training=None,**kwargs):
        pass

    def backward(self,output):
        pass

    def set_deterministic(self,value, session=None):
        if session is None:
            session = tf.get_default_session()
        if value:
            session.run(self._set_deterministic)
        else:
            session.run(self._set_not_deterministic)

    @property
    def deterministic(self):
        return self._deterministic

    @property
    def reset_variables(self):
        return self._reset_variables

    @property
    def data_format(self):
        return self._data_format

    @property
    def input(self):
        return self._input


from .pool import *
from .perturb import *
from .normalize import *
from .conv import *
from .dense import *
from .shape import *
from .io import *
from .special import *
#from .meta import *
