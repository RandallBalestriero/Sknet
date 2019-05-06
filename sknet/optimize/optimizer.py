#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from . import schedule as sch
from .. import EMA,Variable

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
            t = Variable(np.int32(0), trainable=False, name='t')
            self.params = [t]

            # get the learning rate
            if not np.isscalar(learning_rate):
                learning_rate = learning_rate(t)
            else:
                learning_rate = tf.constant(learning_rate)

            eps     = tf.constant(epsilon)
            updates = [tf.assign_add(t, 1)]

            for param, grad in zip(params,gradients):
                ema_m,m_op = EMA(grad,beta1,t)
                ema_v,v_op = EMA(tf.square(grad),beta2,t)
                step = learning_rate*m_op/(tf.sqrt(v_op)+eps)
                updates.append(tf.assign_sub(param, step))
                updates.append(m_op)
                updates.append(v_op)
            self.updates = updates


