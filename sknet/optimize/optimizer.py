#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from . import schedule as sch
from .. import EMA,Variable,ONE_INT32

class Adam:
    def __init__(self, loss_or_grads, learning_rate, beta1=0.9,
                    beta2=0.999, epsilon=1e-8, params=None):
        with tf.variable_scope("adam"):
            # Parameters
            # if no parameters are given then we get them and
            # compute the gradients
            if params is None:
                params    = tf.trainable_variables()
                # ensure that loss_or_grads is really a tf value
                assert not hasattr(loss_or_grads,'__len__')
            # set up grad or loss
            if hasattr(loss_or_grads,'__len__'):
                gradients = loss_or_grads
            else:
                gradients = tf.gradients(loss_or_grads,params)

            # Perform Adam
            self.step = Variable(-ONE_INT32, trainable=False, name='step')
            step = tf.assign_add(self.step, ONE_INT32)
            # get the learning rate
            if not np.isscalar(learning_rate):
                learning_rate = learning_rate(step)
            else:
                learning_rate = tf.constant(learning_rate)

            eps     = tf.constant(epsilon)
            b1,b2   = tf.constant(beta1), tf.constant(beta2)
            updates = [step]

            for param, grad in zip(params,gradients):
                _, m_op = EMA(grad, b1, step)
                _, v_op = EMA(tf.square(grad), b2, step)
                update  = learning_rate*m_op/(tf.sqrt(v_op)+eps)
                updates.append(tf.assign_sub(param, update))
                updates.append(m_op)
                updates.append(v_op)
            self.updates = updates


