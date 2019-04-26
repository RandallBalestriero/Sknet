#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from . import schedule as sch

class Adam:
    def __init__(self,loss_or_grads,learning_rate,beta1=0.9,
                    beta2=0.999, epsilon=1e-8,params=None):
        with tf.variable_scope("adam"):
            self.description=''
            # Parameters
            if params is None:
                params    = tf.trainable_variables()
                gradients = tf.gradients(loss_or_grads, params)

            # Set up grad or loss
            if hasattr(loss_or_grads,'__len__'):
                gradients = loss_or_grads
            else:
                gradients = tf.gradients(loss_or_grads,params)

            # Perform Adam
            t_prev  = tf.Variable(np.int32(0), trainable=False, name='t_Adam')

            t = tf.assign_add(t_prev, 1)
            if not np.isscalar(learning_rate):
                learning_rate = learning_rate(t)
            else:
                learning_rate = tf.constant(learning_rate)

            tfloat = tf.cast(t,tf.float32)
            a_t    = tf.sqrt(1-tf.pow(beta2,tfloat))/(1-tf.pow(beta1, tfloat))

            ema_m   = tf.train.ExponentialMovingAverage(beta1,t_prev)
            ema_v   = tf.train.ExponentialMovingAverage(beta2,t_prev)
            epsilon = tf.constant(epsilon)

            updates = list()
            for param, grad in zip(params,gradients):
                grad_square = tf.square(grad)
                op_m = ema_m.apply([grad])
                op_v = ema_v.apply([grad_square])
                with tf.control_dependencies([op_m,op_v]):
                    std  = tf.sqrt(ema_v.average(grad_square)) + epsilon
                    step = learning_rate * a_t * ema_m.average(grad)/std
                    updates.append(tf.assign_sub(param, step))

            updates.append(t)
            self.updates = updates


