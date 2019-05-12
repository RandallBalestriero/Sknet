#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm

__all__ = [
        "plotting",
        "geometry"]


def repeat(tensor,values):
    tensor_shape = tensor.shape.as_list()
    assert len(values)==len(tensor_shape)
    #
    augmented_shape = sum([[s,1] for s in tensor_shape],[])
    augmented = tf.reshape(tensor,augmented_shape)
    #
    tilings   = sum([[1,v] for v in values],[])
    tiled     = tf.tile(augmented, tilings)
    #
    output_shape = [v*s for v,s in zip(values,tensor_shape)]
    output       = tf.reshape(tiled,output_shape)
    return output


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def count_number_of_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])



def hermite_interp(t, knots, m, p):
    """
    Parameters
    ----------

    t : 1d vector
        represents time sampling

    knots : matrix
        a collection of knots of dimension (N_FILTERS,N_KNOTS) that share
        the same time sampling

    m : matrix
        the value of the function at the knots (shared across filters)

    p : matrix
        the value of the derivative of the function at the knots
        (shared across filters)

    Returns
    -------

    f : 2d array
        the interpolated filters

    """

    # the matrix sovling the linear system of equation and giving
    # the solution for the polynomial coefficients based on the
    # imposed values of the function and its derivative at the
    # region boundaries
    M = tf.constant(np.array([[1, 0,-3, 2],
                              [0, 0, 3,-2],
                              [0, 1,-2, 1],
                              [0, 0,-1, 1]]).astype('float32'))

    # Concatenate the coefficients of left and right position of the
    # region over the first dimensiononto knots 0:-1 and 1:end
    # dimension (I B R 4)
    y  = tf.stack([m[...,:-1], m[...,1:], p[...,:-1], p[...,1:]], axis=-1)
    ym = tf.einsum('ijab,bc->ijac', y, M)

    # create the time sampling versions to be between 0 and 1 for each interval
    # thus having a tensor of shape (N_FILTERS,N_REGIONS,TIME_SAMPLING)
    # first make it start at 0 and then end at 1
    knots   = tf.expand_dims(knots,2)
    t_unit  = (t-knots[:,:-1])/(knots[:,1:]-knots[:,:-1]+1e-8)
    # then remove everything that is not between 0 and 1
    mask    = tf.cast(tf.logical_and(tf.greater_equal(t_unit, 0.), 
                              tf.less(t_unit, 1.)), tf.float32) #(J*Q R time)

    # create all the powers for the interpolation formula (J*Q R time 4)
    t_p = tf.concat([tf.expand_dims(mask,-1), tf.expand_dims(t_unit*mask,-1),
                         tf.pow(tf.expand_dims(t_unit*mask,3), [2,3])],-1)

    filters = tf.reduce_sum(tf.expand_dims(ym,3)*t_p,[2,4])
    return filters





                       
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

