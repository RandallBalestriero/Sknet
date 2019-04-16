#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

__all__ = [
        "plotting",
        "geometry"]

def count_number_of_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


                       
def case(integer,tensors,start=0):
    cond = tf.equal(integer-start,0)
    if len(tensors)==1:
        return tensors[0]
    if len(tensors)==2:
        return tf.cond(cond,lambda :tensors[0],lambda :tensors[1])
    else:
        follow = case(integer,tensors[1:],start+1)
        return tf.cond(cond,lambda :tensors[0],lambda :follow)


def to_one_hot(labels,K=None):
    if K is None:
        K=int(np.max(labels)+np.min(labels))
    matrix = np.zeros((len(labels),K),dtype='float32')
    matrix[range(len(labels)),labels]=1
    return matrix


from . import *
from .workplace import *

