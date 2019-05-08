#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class SplineFilter1D:
    def __init__(self, J, Q, K, trainable_knots=False, trainable_scale=False,
                   trainable_filters=False, init='gabor', complex=False, hilbert=False,
                   m=None, p=None, knots=None, start=0, end=-1):
        self.J = J
        self.K = K
        self.Q = Q
        self.trainable_filters = trainable_filters
        self.trainable_scale   = trainable_scale
        self.trainable_knots   = trainable_knots
        self.init    = init
        self.complex = complex
        self.hilbert = hilbert
        self.m,self.p= m, p
    def __call__(self,*args,**kwargs):

        # scales
        scales = 2**(tf.range(self.J,delta=1./self.Q,dtype=tf.float32))
        scales = tf.Variable(scales.astype('float32'),
                            trainable=self.trainable_scales, name='scales')
        sorted_scales = tf.contrib.framework.sort(scales)
        indices = np.arange(0,J,1./Q)
 
        # knots
        start = (self.K//2)
        grid  = tf.lin_space(np.float32(-start),np.float32(start), self.K)
        knots = tf.Variable(grid, trainable=self.trainable_knots, name='knots')
        all_knots = tf.einsum('i,j->ij',scales,knots)
 
        # initialize m and p
        if init=='gabor':
            window = np.hamming(self.K)
            m = (np.cos(np.arange(self.K) * np.pi)*window).astype('float32')
            p = np.zeros(self.K,dtype='float32')
            if complex and not hilbert:
                m_imag = np.zeros(self.K)
                p_imag = np.cos(np.arange(self.K) * np.pi)*window
                m = np.stack([m,m_imag]).astype('float32')
                p = np.stack([p,p_imag]).astype('float32')
        elif init=='random':
            m = np.random.randn(self.K,dtype='float32')
            p = np.random.randn(self.K,dtype='float32')
            if complex and not hilbert:
                m_imag = np.random.randn(self.K,dtype='float32')
                p_imag = np.random.randn(self.K,dtype='float32')
                m = np.stack([m,m_imag]).astype('float32')
                p = np.stack([p,p_imag]).astype('float32')

        if self.m is None:
            m = tf.Variable(m, trainable=self.trainable_filters, name='m')
        else:
            m = self.m
        if self.p is None:
            p = tf.Variable(p, trainable=self.trainable_filters, name='p')
        else:
            p = self.p

        # Boundary Conditions and centering
        mask    = np.ones(self.K, dtype=np.float32)
        mask[0], mask[-1] = 0, 0
        m_null  = m - tf.reduce_mean(m[...,1:-1], axis=-1, keepdims=True)

        # Filter sampling
        # add an extra octave if learnable scales (to go to lower frequency)
        # Define the integer time grid (time sampling)
        length    = int(self.K*2**(self._scales[end-1]
                                        +int(self.trainable_scales)))
        time_grid = tf.linspace(np.float32(-(length//2)),
                                    np.float32(length//2), length)

        # ------ FILTER-BANK
        if self.hilbert:
            filters_real = utils.hermite_interp(time_grid,
                        self.all_knots[start:end], self.m, self.p, True)
            filters_fft = tf.spectral.rfft(filters)
            filters = tf.ifft(tf.concat([filters_fft,
                                            tf.zeros_like(filters_fft)],1))
        else:
            filters = utils.hermite_interp(time_grid, self.all_knots[start:end],
                                            self.m, self.p,False)
            filters = tf.complex(filters[0],filters[1])


