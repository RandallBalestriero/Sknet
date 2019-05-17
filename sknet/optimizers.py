#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
#import .schedules as sch
from .base import EMA, ONE_INT32

class NesterovMomentum:
    def __init__(self, loss_or_grads, learning_rate, momentum=0.9, params=None):
        with tf.variable_scope("NesterovMomentumOptimizer") as scope:
            self.name = scope.original_name_scope
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

            self.step = tf.Variable(-ONE_INT32, trainable=False, name='step')
            step = tf.assign_add(self.step, ONE_INT32)
            # get the learning rate
            if not np.isscalar(learning_rate):
                learning_rate = learning_rate(step)
            else:
                learning_rate = tf.constant(learning_rate)

            updates = list()
            for param, grad in zip(params,gradients):
                lrg  = learning_rate*grad
                velocity_var = tf.Variable(tf.zeros_like(param),
                                     name='velocity',trainable=False)
                velocity = tf.assign(velocity_var,momentum*velocity_var-lrg)
                update   =momentum*velocity-lrg
                updates.append(tf.assign_add(param, update))
                updates.append(velocity)
            self.updates = updates
        self.reset_variables_op = list()
        for var in tf.global_variables(self.name):
            self.reset_variables_op.append(tf.assign(var,var.initial_value))

class Adam:
    def __init__(self, loss_or_grads, params, learning_rate, beta1=0.9,
                    beta2=0.999, epsilon=1e-6):
        with tf.variable_scope("AdamOptimizer") as scope:
            self.name = scope.original_name_scope
            # set up grad or loss
            if hasattr(loss_or_grads,'__len__'):
                gradients = loss_or_grads
                assert len(gradients)==len(params)
            else:
                gradients = tf.gradients(loss_or_grads,params)

            # Perform Adam
            self.step = tf.Variable(-ONE_INT32, trainable=False, name='step')
            step = tf.assign_add(self.step, ONE_INT32)
            # get the learning rate
            if not np.isscalar(learning_rate):
                learning_rate = learning_rate(step)
            else:
                learning_rate = tf.constant(learning_rate)

            eps = tf.constant(epsilon)
            beta1 = tf.constant(beta1)
            beta2 = tf.constant(beta2)
            self.updates = [step]

            for param, grad in zip(params,gradients):
                _, m_op = EMA(grad, beta1, step)
                _, v_op = EMA(tf.square(grad), beta2, step)
                update = learning_rate*m_op/(tf.sqrt(v_op)+eps)
                self.updates += [tf.assign_sub(param, update), m_op, v_op]
        self.reset_variables_op = list()
        for var in tf.global_variables(self.name):
            self.reset_variables_op.append(tf.assign(var,var.initial_value))

