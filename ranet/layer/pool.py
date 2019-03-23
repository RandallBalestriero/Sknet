#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


# ToDo: PoolLayer: optimize s.t. if it is just channel pooling you dont reshape


class PoolLayer:
    def __init__(self, incoming, windows, strides=None, pool_type='MAX', padding='VALID'):
        """

        incoming: a ranet.layer

        windows : integer vector of length 2 (spatial pooling)
                  or 
                  integer vector of length 3 with channel dimension value set to 1 (spatial pooling)
                  or
                  integer vector of length 3 (spatial and/or channel pooling)
                  
        strides : None or integer vector same length as windows

        pool_type: str, 'MAX' or 'AVG'

        padding : str, 'VALID' or 'SAME'

        data_format: str, 'NCHW' or 'NHWC'

        Example of use:

        # (3,3) max pooling with (3,3) strides
        # All ther below are equivalent
        PoolLayer(previous_layer, windows=(3,3), strides=(3,3),data_format='NCHW')
        PoolLayer(previous_layer, windows=(3,3),data_format='NCHW')
        PoolLayer(previous_layer, windows=(1,3,3), strides=(1,3,3),data_format='NCHW')
        PoolLayer(previous_layer, windows=(1,3,3),data_format='NCHW')


        """
        self.data_format = incoming.data_format
        if strides is not None:
            assert(len(strides)==len(windows))
        else:
            strides = windows
        if len(windows)==2:
            # This is the standard spatial pooling case
            self.output = tf.nn.pool(incoming.output,strides=strides,
                    pool_type=pool_type, padding=padding, data_format=self.data_format)
        else:
            # The length of windows is 3, but does not always imply channel pooling
            # Check if channel dimension pooling is >1, if yes, performs the pooling
            if (self.data_format=='NCHW' and windows[0]>1) or (self.data_format=='NHWC' and windows[2]>1):
                self.output = tf.nn.pool(tf.expand_dims(incoming.output,-1),window_shape=windows,strides=strides, 
                    pooling_type=pool_type, padding=padding, data_format=self.data_format)[...,0]
            else:
                # No channel pooling case, remove the '1' of the channel dimension of windows and
                # perform standard pooling
                if self.data_format=='NCHW':
                    windows = windows[1:]
                    strides = strides[1:]
                else:
                    windows = windows[:-1]
                    strides = strides[:-1]
                self.output = tf.nn.pool(incoming.output,strides=strides,window_shape = windows,
                    pooling_type=pool_type, padding=padding, data_format=self.data_format)
        # Set-up the the VQ
        if pool_type=='AVG':
            self.VQ = None
        else:
            self.VQ = tf.gradients(self.output,incoming.output,tf.ones_like(self.output))[0]
        self.output_shape=self.output.get_shape().as_list()


class GlobalSpatialPoolLayer:
    def __init__(self,incoming, pool_type='AVG'):
        """

        incoming: a ranet.layer

        pool_type: str, 'MAX' or 'AVG'

        data_format: str, 'NCHW' or 'NHWC'

        Pool over all the spatial dimension (according to data_format) and 
        to the pooling type (from pool_type)

        """
        self.data_format = incoming.data_format

        if self.data_format=='NCHW':
            axis = [2,3]
        else:
            axis = [1,2]
        if pool_type=='AVG':
            self.output = tf.reduce_mean(incoming.output,axis=axis,keep_dims=True)
            self.VQ = None
        else:
            self.output = tf.reduce_max(incoming.output,axis=axis,keep_dims=True)
            self.VQ = tf.gradients(self.output,incoming.output,tf.ones_like(self.output))[0]
        self.output_shape = [incoming.output_shape[0],1,1,incoming.output_shape[3]]







