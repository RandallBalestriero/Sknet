#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class BatchIterator(dict):
    """
    Parameters
    ----------

    options : dict
        A dictionnary describing for each set of a :class:`sknet.Dataset`
        object how to read (sample) batches from it and the batch_size.
        The options are  ``"continuous"``, ``"random"``, ``"random_all"``.
        For example::

            iterator = BatchIterator({'train_set':'random',
                                     'test_set':continuous})
            # returns the array of indices that are looped through
            iterator['train_set']
            # returns the tf.Variable holding the current batch indices
            iterator.train_set


        For specific applications, such as semi-supervised learning, it might
        be useful to simultaneously extract patches from two different sets as
        a single batch. If those two dataset have same number of samples,
        it is straightforward to combine them. But if their length differ or
        if needs be to have a random (but same) batch sampling, then do as
        follows::

            iterator = BatchIterator({'sup_train_set,
                                     unsup_train_set':'random',
                                     'test_set':continuous})
            # returns the array of indices that are looped through
            iterator['train_set']
            # returns the tf.Variable holding the current batch indices
            iterator.train_set


    """
    def __init__(self, batch_size, options):

        self.options = options
        self.N = dict()
        self.N_BATCH = dict()
        self.batch_size = batch_size
        self.batch_counter = dict()
        self.sets = list(options.keys())
        self.indices_assign_op = dict()

        # extract the sets and then check for joint ones and separate them
        print('BatchIterator initialized with ')
        with tf.variable_scope("iterator"):
            for s, v in options.items():
                print('\t\t{}:{}'.format(s, v))
                indices = np.zeros(batch_size, dtype='int64')
                self.__dict__[s] = tf.Variable(indices, trainable=False,
                                               name='indices')
                self.__dict__[s+'_'] = tf.placeholder(tf.int64,
                                                      shape=(batch_size,),
                                                      name='indices')
                self.indices_assign_op[s] = tf.assign(self.__dict__[s],
                                                      self.__dict__[s+'_'])

            # first set up some variable to keep track of which
            # set is used
            self.current_set = tf.Variable(np.int64(0), trainable=False,
                                           name='current')
            self._current_set = tf.placeholder(tf.int64, name='current')
            self.assign_set = tf.assign(self.current_set, self._current_set)
            self.set2int = dict()
            for i, name in enumerate(options.keys()):
                self.set2int[name] = i

    def set_N(self, values=None, dataset=None):
        """set the total length of each set. This method keeps updating
        an internal dictionnary and thus can be called multiple times to add
        or update the lengths. This method should be called for all sets
        prior using the iterator to get the batches.

        Parameters
        ----------

        values : dict
            a dictionnary mapping set names (as string) to their sizes (int).
            For example::

                iterator.set_N({'train_set':50000, 'test_set':10000)

            If a joint dataset was given then the set length should be
            specified individually for each of them.

        dataset : sknet.Dataset
            optionally an already built dataset can be passed and use
            to get all the lengths.

        """
        if values is not None and dataset is None:
            self.N.update(values)
            for s, k in values.items():
                self.N_BATCH[s] = k//self.batch_size
                self.reset(s)
        elif values is None and dataset is not None:
            for s in self.sets:
                self.N.update({s: dataset.N(s)})
                self.N_BATCH.update({s: dataset.N(s)//self.batch_size})
                self.reset(s)
        else:
            print('error')
            exit()

    def set_set(self, name, session):
        session.run(self.assign_set,
                    feed_dict={self._current_set: self.set2int[name]})

    def reset(self, s):
        """reset the indices to loop through for a specific set

        Parameters
        ----------

        s : str
            one of the set, the one to have its indices reset

        """
        self.batch_counter.update({s: -1})
        if self.options[s] == "continuous":
            self[s] = np.asarray(range(self.N[s])).astype('int64')
        elif self.options[s] == "random":
            self[s] = np.random.choice(self.N[s], (self.N[s],))
        else:
            self[s] = np.random.permutation(self.N[s])

    def next(self, s, session=None):
        if session is None:
            session = tf.get_default_session()
        if self.N_BATCH[s] == (self.batch_counter[s]+1):
            self.reset(s)
            return False
        self.batch_counter[s] += 1
        batch_indices = range((self.batch_counter[s])*self.batch_size,
                              (self.batch_counter[s]+1)*self.batch_size)
        session.run(self.indices_assign_op[s],
                    feed_dict={self.__dict__[s+'_']: self[s][batch_indices]})
        return True










