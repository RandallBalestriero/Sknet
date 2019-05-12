#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import tensorflow as tf

from ..utils import case


class Dataset(dict):
    def __init__(self,**args):
        super().__init__()
        self.__dict__.update(args)
    def next(self,session=None):
        return self.iterators[self.current_set_string].next(session=session)
    def set_set(self,name,session):
        self.current_set_string = name
        session.run(self.assign_set,
                    feed_dict={self.current_set_:self.set2int[name]})
    @property
    def init_dict(self):
        """return the list of couple with the tensorflow
        variable and the dataset variable to be used to initialized the
        tf one, this has to be run when initializing the graph,
        the TF variables are created before hand by create_placeholders"""
        alls= [(self[v+'/'+s+'/placeholder'],self[v+'/'+s]) 
                         for v in self.variables for s in self.sets_(v)]
        return dict(alls)
    def create_placeholders(self,batch_size,iterators_dict,device="/cpu:0"):
        # Many settings are put in int64 for GPU compatibility with tf
        sets            = self.sets
        for var in self:
            if np.shape(self[var])[0]<batch_size:
                print('Error, batch size larger than some variables')
                exit()
        self.set2int    = dict([(b,np.int64(a)) for a,b in enumerate(sets)])
        self.iterators  = iterators_dict
        with tf.device(device):
            with tf.variable_scope("dataset") as scope:
                # First set up some variable to keep track of which set is used
                self.current_set  = tf.Variable(np.int64(0),trainable=False,
                                            name='current_set')
                self.current_set_ = tf.placeholder(tf.int64,
                                            name='current_set')
                self.assign_set = tf.assign(self.current_set,self.current_set_)
                # Initialize iterators with dataset size and batch size
                for context in sets:
                    self.iterators[context].set_N(self.N(context))
                    self.iterators[context].set_batch_size(batch_size)
                # Now create all the variables
                self.tf_variables = dict()
                # Loop over all internal variables
                for varn in list(self.keys()):
                    # ensure that there is not already a member with this name
                    # as a method of the class (dict)
                    assert(varn not in self.__dict__)
                    if type(self[varn])=='int32':
                        type_ = 'int64'
                    else:
                        type_ = str(self[varn].dtype)
                    # create all the placeholders and variables for each data
                    name1 = varn+'/placeholder'
                    self[name1] = tf.placeholder(type_, shape=self[varn].shape,
                                                                  name = varn)
                    name2 = varn+'/Variable'
                    self[name2] =tf.Variable(self[name1], trainable=False)
                for v in self.variables:
                    pairs = list()
                    for s in self.sets_(v):
                        name = v+'/'+s+'/Variable'
                        if self[v+'/'+s].dtype=='int32':
                            batch = tf.cast(tf.gather(tf.cast(
                                                 self[name],tf.float32), 
                                                 self.iterators[s].i),tf.int32)
                        else:
                            batch = tf.gather(self[name],self.iterators[s].i)
                        pairs.append(tf.placeholder_with_default(batch,
                                                                batch.shape))
                    self.__dict__[v] = case(self.current_set,pairs)

    def split_set(self,set_,new_set_,ratio, stratify = None, seed=None):
        if new_set_ in self.sets:
            error
        variables = self.variables
        if stratify is not None:
            error
            assert(len(y)>1)
            train_indices = list()
            valid_indices = list()
            for c in set(y):
                c_indices = np.where(y==c)[0]
                np.random.shuffle(c_indices)
                limit = int(len(c_indices)*test_ratio)
                train_indices.append(c_indices[limit:])
                valid_indices.append(c_indices[:limit])
            train_indices = np.concatenate(train_indices,0)
            valid_indices = np.concatenate(valid_indices,0)
            #
            self["valid_set"]=[s[valid_indices] if isinstance(s,np.ndarray)
                        else [s[i] for i in valid_indices] 
                        for s in self["train_set"]]
            #
            self["train_set"]=[s[train_indices] if isinstance(s,np.ndarray)
                        else [s[i] for i in train_indices]
                        for s in self["train_set"]]
        else:
            indices = np.random.RandomState(seed).permutation(self.N(set_))
            N = int(self.N(set_)*ratio)
            for var in self.variables:
                self[var+'/'+set_],self[var+'/'+new_set_]=\
                           self[var+'/'+set_][indices[N:]],\
                           self[var+'/'+set_][indices[:N]]

    @property
    def sets(self):
        return sorted(list(np.unique([k.split('/')[1] for k in self])))
    def sets_(self,varn):
        return sorted(list(np.unique([k.split('/')[1] 
				for k in self if varn in k])))
    @property
    def variables(self):
        return sorted(list(np.unique([k.split('/')[0] for k in self])))

    def N(self,context):
        for var in self:
            if context in var and len(var.split('/'))==2:
                return len(self[var])
    def shape(self,var):
        set_ = self.sets[0]
        return self[var+'/'+set_].shape
    def datum_shape(self,var):
        return self.shape(var)[1:]

