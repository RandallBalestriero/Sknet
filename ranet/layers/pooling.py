#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class PoolLayer:
    def __init__(self, incoming, windows, strides=1, pool_type='MAX', padding='VALID',data_format='NCHW'):
        self.output = tf.nn.pool(incoming.output, windows, pool_type, padding=padding, 
                                    strides=strides, data_format=data_format)
        self.VQ     = None #TO CHANGE !
        self.output_shape=self.output.get_shape().as_list()


class GlobalPoolLayer:
    def __init__(self,incoming,data_format='NCHW',pool_type='AVG'):
        if data_format=='NCHW':
            if pool_type=='AVG':
                self.output = tf.reduce_mean(incoming.output,[2,3],keep_dims=True)
            else:
                self.output = tf.reduce_max(incoming.output,[2,3],keep_dims=True)
            self.output_shape = [incoming.output_shape[0],incoming.output_shape[1],1,1]
            self.VQ = None
        else:
            if pool_type=='AVG':
                self.output = tf.reduce_mean(incoming.output,[1,2],keep_dims=True)
            else:
                self.output = tf.reduce_max(incoming.output,[1,2],keep_dims=True)
            self.output_shape = [incoming.output_shape[0],1,1,incoming.output_shape[3]]
            self.VQ = None #TO CHANGE !













