#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf



class sparse_cross_entropy_logits:
    """Cross entropy loss given that :math:`p` is sparse and
    :math:`q` is the log-probability.

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
    def __init__(self,p=None,q=None,weights=None):
        self.name = 'sparse_cross_entropy_logits'
        if p is None and q is None:
            print("error")
            exit()
        elif p is None:
            self._p = tf.placeholder(tf.int32,shape=(q.shape[0],))
        if q is None:
            print('error')
            exit()
        self._q = q

        indices = tf.stack([tf.range(self.p.shape[0],dtype=tf.int32),
                            self.p],1)
        # we extract the max to rmeove it from q to ensure stability
        # we also stop the gradient to allow unbiased learning
        q_max   = tf.stop_gradient(tf.reduce_max(q,1,keepdims=True))
        linear_ = tf.gather_nd(self.q,indices)
        logsumexp  = tf.log(tf.reduce_sum(tf.exp(q-q_max),1))
        if weights is not None:
            self._loss = tf.reduce_sum(weights*(-linear_+logsumexp))
        else:
            self._loss = tf.reduce_sum(-linear_+logsumexp)

        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(
                    self.q,1,output_type=tf.int32),self.p),tf.float32))


    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy




class sparse_cross_entropy:
    """Cross entropy loss given that :math:`p` is sparse.

    The formal definition given that :math:`p` is now an
    index (of the Dirac) s.a. :math:`p\in \{1,\dots,D\}`
    is given by (for discrete variables)
    :math:`\mathcal{L}(p,q)=-\log(q_{p})`. it is
    commonly used for classification task with :math:`p` the class 
    index and :math:`q` the predicted one
    (output of the network). This class takes two non sparse
    vectors which should be nonnegative and sum to one.
    """
    def __init__(self,p=None,q=None, eps=1e-8):
        self.eps = eps
        self.name = 'oneclass_cross_entropy'
        if p is None and q is None:
            print("error")
            exit()
        elif p is None:
            self._p = tf.placeholder(q.dtype,shape=q.shape)
        elif q is None:
            self._q = tf.placeholder(p.dtype,shape=p.shape)

        indices = tf.stack([tf.range(self.p.shape[0],dtype=tf.int32),
                            self.p],1)
        self._loss = -tf.reduce_sum(tf.gather_nd(tf.log(self.q+self.eps),
                            tf.transpose(indices)))

    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    @property
    def loss(self):
        return self._loss




class cross_entropy:
    """Cross entropy loss given two probability distributions.

    The formal definition is given by (for discrete variables)
    :math:`\mathcal{L}(p,q)=-\sum_{d=1}^D p_d \log(q_d)`. it is
    commonly used for classification task with :math:`p` the true
    probability of class belonging and :math:`q` the predicted one
    (output of the network). This class takes two non sparse
    vectors which should be nonnegative and sum to one.
    """
    def __init__(self,p=None,q=None, eps=1e-8):
        self.eps = eps
        self.name = 'cross_entropy'
        if p is None and q is None:
            print("error")
            exit()
        elif p is None:
            self._p = tf.placeholder(q.dtype,shape=q.shape)
        elif q is None:
            self._q = tf.placeholder(p.dtype,shape=p.shape)

        self._loss = -tf.reduce_sum(self.p*tf.log(self.q+self.eps))

    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    @property
    def loss(self):
        return self._loss






def classification(layer,**kwargs):
    return tf.reduce_mean(tf.losses.softmax_cross_entropy(layer.observation,layer,**kwargs))

def accuracy(layer,**kwargs):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer.observation,1),tf.argmax(layer,1)),tf.float32))

        


