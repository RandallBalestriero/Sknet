#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .. import Tensor



def MSE(target,prediction):
    N = np.float32(np.prod(target.shape.as_list()))
    return SSE(target,prediction)/N

def SSE(target,prediction):
    return tf.nn.l2_loss(target-prediction)*2

def AUC(target,prediction):
    return tf.metrics.auc(target,prediction)

def crossentropy_logits(p,q,weights=None,p_sparse=True):
    """Cross entropy loss given that :math:`p` is sparse and
    :math:`q` is the log-probability.

    The formal definition given that :math:`p` is now an
    index (of the Dirac) s.a. :math:`p\in \{1,\dots,D\}`
    and :math:`q` is unormalized (log-proba)
    is given by (for discrete variables, p sparse) 

    .. math::
        \mathcal{L}(p,q)=-q_{p}+\log(\sum_{d=1}^D \exp(q_d))
    .. math::
        \mathcal{L}(p,q)=-q_{p}+LogSumExp(q)
    .. math::
        \mathcal{L}(p,q)=-q_{p}+LogSumExp(q-\max_{d}q_d)

    or by (non p sparse)

    .. math::
        \mathcal{L}(p,q)=-\sum_{d=1}^Dp_{d}q_{d}+\log(\sum_{d=1}^D \exp(q_d))
    .. math::
        \mathcal{L}(p,q)=-\sum_{d=1}^Dp_{d}q_{d}+LogSumExp(q)
    .. math::
        \mathcal{L}(p,q)=-\sum_{d=1}^Dp_{d}q_{d}+LogSumExp(q-\max_{d}q_d)


    with :math:`p` the class index and :math:`q` the predicted one
    (output of the network). This class takes two non sparse
    vectors which should be nonnegative and sum to one.
    """
    if p_sparse:
        indices = tf.stack([tf.range(p.shape[0],dtype=tf.int32),p],1)
        linear_ = tf.gather_nd(q,indices)
    else:
        linear_ = tf.reduce_sum(p*q,1)
    # LogSumExp with max removal
    q_max      = tf.stop_gradient(tf.reduce_max(q,1,keepdims=True))
    logsumexp  = tf.log(tf.reduce_sum(tf.exp(q-q_max),1))+q_max[:,0]
    if weights is not None:
        return tf.reduce_mean(weights*(-linear_+logsumexp))
    else:
        return tf.reduce_mean(-linear_+logsumexp)
 







def accuracy(labels,predictions):
    """Accuracy

    The formal definition given that :math:`p` is now an
    index (of the Dirac) s.a. :math:`p\in \{1,\dots,D\}`
    and :math:`q` is unormalized (log-proba)
    is given by (for discrete variables) 

    .. math::
        \mathcal{L}(p,q)=-q_{p}+\log(\sum_{d=1}^D \exp(q_d))
    .. math::
        \mathcal{L}(p,q)=-q_{p}+LogSumExp(q)
    .. math::
        \mathcal{L}(p,q)=-q_{p}+LogSumExp(q-\max_{d}q_d)

    with :math:`p` the class index and :math:`q` the predicted one
    (output of the network). This class takes two non sparse
    vectors which should be nonnegative and sum to one.

    """
    equals =tf.equal(labels,tf.argmax(predictions,1,output_type=tf.int32))
    return tf.reduce_mean(tf.cast(equals,tf.float32))





