#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from ..utils import case


class Dataset(dict):
    def __init__(self, **args):
        super().__init__()
        self.__dict__.update(args)

    def cast(self, varn, dtype):
        # get sets in which this var exists
        sets = self.sets_(varn)
        for s in sets:
            self[varn+'/'+s]=self[varn+'/'+s].astype(dtype)

    @property
    def init_dict(self):
        """return the list of couple with the tensorflow
        variable and the dataset variable to be used to initialized the
        tf one, this has to be run when initializing the graph,
        the TF variables are created before hand by create_placeholders"""
        alls = [(self[v+'/'+s+'/placeholder'], self[v+'/'+s])
                         for v in self.variables for s in self.sets_(v)]
        return dict(alls)

    def create_placeholders(self, iterator, device="/cpu:0"):
        # Many settings are put in int64 for GPU compatibility with tf
        iterator.set_N(dataset=self)
        sets = self.sets
        self.iterator = iterator
        with tf.device(device):
            with tf.variable_scope("dataset"):

                # create the tensorflow placeholders and variables that
                # will hold the values of the sets and variables as part of
                # the computational graph. Each variable/set is associated
                # to a tf.Variable and a tf.placeholder (to initialize the
                # variable)
                for varn in list(self.keys()):
                    # ensure that there is not already a member with this name
                    # as a method of the class (dict)
                    assert(varn not in self.__dict__)
                    if type(self[varn]) == 'int32':
                        type_ = 'int64'
                    else:
                        type_ = str(self[varn].dtype)
                    # create all the placeholders and variables for each data
                    name1 = varn+'/placeholder'
                    self[name1] = tf.placeholder(type_, shape=self[varn].
                                                 shape, name=varn)
                    name2 = varn+'/Variable'
                    self[name2] = tf.Variable(self[name1], trainable=False)

                # now initialize the batch version of each variable/set.
                # to do so, we use the iterator indices to extract each batch
                for v in self.variables:
                    pairs = list()
                    for s in self.sets_(v):
                        name = v+'/'+s+'/Variable'
                        indices = tf.mod(self.iterator.__dict__[s],
                                         self[v+'/'+s].shape[0])
                        if self[v+'/'+s].dtype == 'int32':
                            batch = tf.gather(tf.cast(self[name], tf.float32),
                                              indices)
                            batch = tf.cast(batch, tf.int32)
                        else:
                            batch = tf.gather(self[name], indices)
                        pairs.append(tf.placeholder_with_default(batch,
                                                                 batch.shape))
                    self.__dict__[v] = case(self.iterator.current_set, pairs)

    def split_variable(self, var, new_var, ratio, stratify=None, seed=None):
        assert new_var not in self.variables
        sets=self.sets
        if stratify is not None:
            exit()
            assert(len(y) > 1)
            train_indices = list()
            valid_indices = list()
            for c in set(y):
                c_indices = np.where(y==c)[0]
                np.random.shuffle(c_indices)
                limit = int(len(c_indices)*test_ratio)
                train_indices.append(c_indices[limit:])
                valid_indices.append(c_indices[:limit])
            train_indices = np.concatenate(train_indices, 0)
            valid_indices = np.concatenate(valid_indices, 0)
            #
            self["valid_set"]=[s[valid_indices] if isinstance(s, np.ndarray)
                        else [s[i] for i in valid_indices] 
                        for s in self["train_set"]]
            #
            self["train_set"]=[s[train_indices] if isinstance(s, np.ndarray)
                        else [s[i] for i in train_indices]
                        for s in self["train_set"]]
        else:
            indices = np.random.RandomState(seed).permutation(self.N(set_))
            if ratio < 1:
                N = int(self.N(set_)*ratio)
            else:
                N = ratio
            for s in sets:
                name = var+'/'+set_
                new_name = new_var+'/'+new_set_
                self[name], self[new_name]=\
                               self[name][indices[N:]],self[name][indices[:N]]



    def split_set(self, set_, new_set_, ratio, stratify=None, seed=None):
        assert new_set_ not in self.sets
        variables=self.variables
        if stratify is not None:
            exit()
            assert(len(y) > 1)
            train_indices = list()
            valid_indices = list()
            for c in set(y):
                c_indices = np.where(y==c)[0]
                np.random.shuffle(c_indices)
                limit = int(len(c_indices)*test_ratio)
                train_indices.append(c_indices[limit:])
                valid_indices.append(c_indices[:limit])
            train_indices = np.concatenate(train_indices, 0)
            valid_indices = np.concatenate(valid_indices, 0)
            #
            self["valid_set"]=[s[valid_indices] if isinstance(s, np.ndarray)
                        else [s[i] for i in valid_indices] 
                        for s in self["train_set"]]
            #
            self["train_set"]=[s[train_indices] if isinstance(s, np.ndarray)
                        else [s[i] for i in train_indices]
                        for s in self["train_set"]]
        else:
            indices = np.random.RandomState(seed).permutation(self.N(set_))
            if ratio < 1:
                N = int(self.N(set_)*ratio)
            else:
                N = ratio
            for var in self.variables:
                name = var+'/'+set_
                new_name = var+'/'+new_set_
                self[name], self[new_name]=\
                               self[name][indices[N:]],self[name][indices[:N]]

    @property
    def sets(self):
        return sorted(list(np.unique([k.split('/')[1] for k in self])))

    def sets_(self, varn):
        return sorted(list(np.unique([k.split('/')[1]
			          for k in self if varn in k])))

    @property
    def variables(self):
        return sorted(list(np.unique([k.split('/')[0] for k in self])))

    def N(self, context):
        """returns the length of a given set. If multiple variables
        belong to the given set, the largest length is returned
        """
        length = 0
        for var in self:
            if context in var and len(var.split('/')) == 2:
                length = np.maximum(length, len(self[var]))
        return length

    def N_BATCH(self, context):
        return self.N(context)//self.batch_size

    def shape(self, var):
        set_ = self.sets[0]
        return self[var+'/'+set_].shape

    def datum_shape(self, var):
        return self.shape(var)[1:]

