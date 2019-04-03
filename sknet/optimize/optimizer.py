#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class Adam:
    def __init__(self,beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
    def minimize(self,loss_or_grads,params=None):
        #
        self.learning_rate = tf.placeholder(tf.float32)
        if params is None:
            params = tf.trainable_variables()
            gradients = tf.gradients(loss_or_grads, params)
        else:
            if callable(loss_or_grads):
                gradients = tf.gradients(loss_or_grads,params)
            else:
                gradients = loss_or_grads

        t_prev  = tf.Variable(np.float32(0), trainable=False, name='t_Adam')
        updates = list()

        one = np.float32(1)

        t   = tf.assign_add(t_prev, one)
        a_t = self.learning_rate*tf.sqrt(1-tf.pow(self.beta2,t))/(1-tf.pow(self.beta1, t))

        for param, g_t in zip(params, gradients):
            m_prev = tf.Variable(np.zeros(param.get_shape().as_list(),
                    dtype='float32'), name='m_prev', trainable=False)
            v_prev = tf.Variable(np.zeros(param.get_shape().as_list(),
                    dtype='float32'), name='m_prev', trainable=False)

            m_t = self.beta1 * m_prev + (one - self.beta1) * g_t
            v_t = self.beta2 * v_prev + (one - self.beta2) * tf.square(g_t)
            step = a_t * m_t / (tf.sqrt(v_t) + self.eps)
            updates.append(tf.assign(m_prev, m_t))
            updates.append(tf.assign(v_prev, v_t))
            updates.append(tf.assign_sub(param, step))

        updates.append(t)

        return tf.group(updates)


