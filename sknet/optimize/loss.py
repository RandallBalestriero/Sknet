#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def classification(layer,**kwargs):
    return tf.reduce_mean(tf.losses.softmax_cross_entropy(layer.observation,layer,**kwargs))

def accuracy(layer,**kwargs):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer.observation,1),tf.argmax(layer,1)),tf.float32))

        


