#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf



class BatchIterator:
    """
    Parameters
    ----------

    sampling : str
        Can be ``"continuous"``, ``"random"``, ``"random_all"``

    """
    def __init__(self,sampling="continuous"):
        self.sampling = sampling
    def set_batch_size(self,bs):
        self.batch_size = bs
        self.i    = tf.Variable(np.zeros(bs).astype('int64'),trainable=False,
                                    name='batchiterator')
        self.i_   = tf.placeholder(tf.int64,shape=(bs,),
                                    name='batchiterator_holder')
        self.i_op = tf.assign(self.i,self.i_)
    def set_N(self,N):
        self.N = N
        self.reset()
    def reset(self):
        self.batch_counter = 0
        if self.sampling=="continuous":
            self.i_values = np.asarray(range(self.N)).astype('int64')
        elif self.sampling=="random":
            self.i_values = np.random.choice(self.N,(self.N,))
        else:
            self.i_values = np.random.permutation(self.N)

    def next(self,session):
        if session is None:
            session=tf.get_default_session()
        try:
            batch_indices = range(self.batch_counter*self.batch_size,
                                    (self.batch_counter+1)*self.batch_size)
            self.batch_counter += 1
            session.run(self.i_op,
                        feed_dict={self.i_:self.i_values[batch_indices]})
            return True
        except:
            self.reset()
            return False


