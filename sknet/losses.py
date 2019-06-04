#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from .base import StreamingTensor


def squared_error(target, prediction, option='mean', aggregate_func=None):
    """Implements the mean squared error between two tensors ```a, b```. 
    For each observation ```n``` the output is given by 
    ```output[n]=mean((a[n]-b[n])**2)``` if the given option is mean, or 
    ```output[n]=sum((a[n]-b[n])**2)``` is the given option is sum, ... 
    The output is thus a vector of
    length given by the first dimenson of ```a``` and/or ```b``` (in case
    of shape broadcasting). Optionally, this vector can be aggregated into
    a scalar (or other) if an aggregating function is provided.

    Parameters:
    -----------

    target: tensor or array
        tensor ```a```

    prediction: tensor or array
        tensor ```b```

    option: str
        the option for the squarred error aggregation. It can be one of
        ```"mean", "sum", "max", "min"```

    aggregate_func: func (optinal, default None)
        aggregating function taking as input the vector ```output```. 
        The output of this functin (if provided) is the output of this loss.
        The aggregate functin should take as input a tensor. Hence use 
        :class:`tf.reduce_mean` and not :class:`numpy.mean`.

    Returns:
    --------

    output: Tensor
        either the vector of mean squared error for each observatin ```n``` 
        or (if an aggreagating function is provided) the output of the given
        ```aggregate_func```.
    
    """
    assert option in ["mean", "sum", "max", "min"]

    if option == "mean":
        func = tf.reduce_mean
    elif option == "max":
        func = tf.reduce_max
    elif option == "sum":
        func = tf.reduce_sum
    elif option == "min":
        func = tf.reduce_min

    axis = list(range(1, max(len(target.shape.as_list()), 
                             len(prediction.shape.as_list()))))
    output = func(tf.square(target - prediction), axis)

    if aggregate_func is not None:
        return aggregate_func(output)
    return output

       
def accuracy(targets, predictions):
    assert len(prediction.shape.as_list()) < 3
    if len(prediction.shape.as_list()) == 2:
        prediction = tf.argmax(prediction, 1, output_type=tf.int32)
    accu = tf.reduce_mean(tf.cast(tf.equal(target, prediction), tf.float32))
    return accu


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





def streaming_mean(tensor):
    moving_sum = tf.Variable(tf.zeros_like(tensor), trainable=False,
                             name='moving_sum')
    moving_count = tf.Variable(tf.zeros(1, dtype=tf.float32), trainable=False,
                               name='moving_count')
    current_mean = moving_sum/moving_count
    updates = tf.group(tf.assign_add(moving_sum, tensor), 
                       tf.assign_add(moving_count, tensor.shape[0]))
    return StreamingTensor(current_mean, updates, [moving_sum, moving_count])

def streaming_sum(tensor):
    moving_sum = tf.Variable(tf.zeros_like(tensor), trainable=False,
                             name='moving_sum')
    updates = tf.assign_add(moving_sum, tensor)
    return StreamingTensor(current_mean, updates, moving_sum)

def streaming_auc(target, prediction):
    with tf.variable_scope('auc') as scope:
        name = scope.original_name_scope
        auc = tf.metrics.auc(target, prediction, num_thresholds=1000)
    variables = tf.local_variables(name)+tf.global_variables(name)
    return StreamingTensor(auc[0], auc[1], variables)
