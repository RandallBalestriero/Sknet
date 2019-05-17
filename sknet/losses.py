#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .base import Tensor


class Loss(Tensor):
    def __init__(self, value):
        super().__init__(value)

class StreamingLoss(Tensor):
    def __init__(self, value, update, reset_op):
        super().__init__(value)
        self.reset_variables_op = reset_op
        self.update = update





def MSE(target,prediction):
    return tf.losses.mean_squared_error(target,prediction)
    N = np.float32(np.prod(target.shape.as_list()[1:]))
    return SSE(target,prediction)/N

def StreamingMSE(target, prediction, scope_name='mse'):
    with tf.variable_scope(scope_name) as scope:
        name = scope.original_name_scope
        mse = tf.metrics.mean_squared_error(target, prediction)
    variables = tf.local_variables(name)+tf.global_variables(name)
    reset_op = tf.variables_initializer(variables)
    return StreamingLoss(mse[0], mse[1], reset_op)


def SSE(target,prediction):
    N = np.float32(2./target.shape.as_list()[0])
    return tf.nn.l2_loss(target-prediction)*N

def StreamingAUC(target,prediction,scope_name='auc'):
    with tf.variable_scope(scope_name) as scope:
        name = scope.original_name_scope
        auc = tf.metrics.auc(target,prediction)
    variables = tf.local_variables(name)+tf.global_variables(name)
    reset_op = tf.variables_initializer(variables)
    return StreamingLoss(auc[0], auc[1], reset_op)


def StreamingAccuracy(target, prediction, scope_name='accuracy'):
    with tf.variable_scope(scope_name) as scope:
        name = scope.original_name_scope
        if len(prediction.shape.as_list())==2:
            prediction = tf.argmax(prediction,1,output_type=tf.int32)
        accu = tf.metrics.accuracy(target, prediction)
    variables = tf.local_variables(name)+tf.global_variables(name)
    reset_op = tf.variables_initializer(variables)
    return StreamingLoss(accu[0], accu[1], reset_op)


def StreamingMean(tensor, scope_name='mean'):
    with tf.variable_scope(scope_name) as scope:
        name = scope.original_name_scope
        amean = tf.metrics.mean(tensor)
    variables = tf.local_variables(name)+tf.global_variables(name)
    reset_op = tf.variables_initializer(variables)
    return StreamingLoss(amean[0], amean[1], reset_op)








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





