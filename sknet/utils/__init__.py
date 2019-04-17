#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm

__all__ = [
        "plotting",
        "geometry"]




class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def count_number_of_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])



def hermite_interp(t, knots, m, p, real=False):
    """
    Parameters
    ----------

    t : 1d vector
        represents time sampling

    knots : matrix
        a collection of knots of dimension (N_FILTERS,N_KNOTS) that share
        the same time sampling

    m : 1d vector
        the value of the function at the knots (shared across filters)

    p : 1d vector
        the value of the derivative of the function at the knots
        (shared across filters)

    real : bool
        is the filter is real or complex (in which case the m and p parameters
        are 2d arrays)

    Returns
    -------

    f : 2d array
        the interpolated filters

    """

    # Create it here for graph bugs
    M = tf.constant(np.array([[1, 0,-3, 2],
                              [0, 0, 3,-2],
                              [0, 1,-2, 1],
                              [0, 0,-1, 1]]).astype('float32'))

    # Concatenate coefficients onto knots 0:-1 and 1:end
    mm = tf.stack([m[:,:-1], m[:,1:]], axis=-1)  # (2/0 KNOTS-1 2)
    pp = tf.stack([p[:,:-1], p[:,1:]], axis=-1)  # (2/0 KNOTS-1 2)
    y  = tf.concat([mm, pp], axis=-1)            # (2 KNOTS-1 4)

    if real:
        ym = tf.matmul(y,M) # (KNOTS-1 4)
    else:
        ym = tf.einsum('iab,bc->iac',y, M)        # (2 KNOTS-1 4)

    # create the time sampling versions to be between 0 and 1 for each interval
    # thus having a tensor of shape (N_FILTERS,N_REGIONS,TIME_SAMPLING)
    # first make it start at 0
    t_zero  = (t-tf.expand_dims(knots[:,:-1],2))
    # then make it end at 1
    t_unit  = t_zero/tf.expand_dims(knots[:,1:]-knots[:,:-1],2)

    # then remove everything that is not between 0 and 1
    mask = tf.cast(tf.logical_and(tf.greater_equal(t_unit, 0.), 
                                        tf.less(t_unit, 1.)), tf.float32)

    # create all the powers for the interpolation formula
    t_p  = tf.pow(tf.expand_dims(t_unit,-1), [0,1,2,3])
    if real:
        return tf.einsum('irf,srtf->ist',ym,t_p*tf.expand_dims(mask,-1))
    else:
        return tf.einsum('irf,srtf->ist',ym,t_p*tf.expand_dims(mask,-1))





                       
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

