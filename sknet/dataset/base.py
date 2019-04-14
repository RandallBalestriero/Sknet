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
        self.datum_shape = {}
        self.dtype = {}
    def add_variable(self,dict_):
        for item in dict_.items():
            self[item[0]]=item[1][0]
            self.datum_shape[item[0]]=item[1][1]
            self.dtype[item[0]]=item[1][2]
        variables = self.variables
        for set_ in self.sets:
            lengths = [len(self[v][set_]) for v in variables]
            assert(len(set(lengths))<=1)
            self.__dict__["n_"+set_]=lengths[0]
    def next(self,session=None):
        return self.iterators[self.current_set_string].next(session=session)
    def set_set(self,name,session):
        self.current_set_string = name
        session.run(self.assign_set,feed_dict={self.current_set_:self.set2int[name]})

    def init_dict(self):
        init = []
        for var in self.variables:
            for s in self.sets:
                init.append((self.tf_variables[var][s][0],self[var][s]))
        return init
    def create_placeholders(self,batch_size,iterators_dict,device="/cpu:0"):
        # Many settings are put in int64 for GPU compatibility with tf
        sets = self.sets
        self.set2int = dict([(b,np.int32(a)) for a,b in enumerate(sets)])
        self.iterators  = iterators_dict
        with tf.device(device):
            self.current_set  = tf.Variable(np.int64(0),trainable=False,
                                        name='current_set')
            self.current_set_ = tf.placeholder(tf.int64,
                                        name='current_set_ph')
            self.assign_set = tf.assign(self.current_set,self.current_set_)
            # Initialize iterators
            for context in sets:
                self.iterators[context].set_N(self.N(context))
                self.iterators[context].set_batch_size(batch_size)
            #
            self.tf_variables = dict()
            for var in self.variables:
                # ensure that there is not already a member with this name
                assert(var not in self.__dict__)
                self.tf_variables[var] = dict()
                pairs = list()
                for s in sets:
                    # cast also to int64 for GPU support
                    if self[var][s][0].dtype=='int32':
                        hold = tf.placeholder('int64',shape=self[var][s].shape,
                                        name = var+'_'+s+'_holder')
                        self.tf_variables[var][s]=(hold,
                                        tf.Variable(hold,trainable=False))
                        batch = tf.cast(tf.gather(tf.cast(
                                    self.tf_variables[var][s][1],tf.float32),
                                    self.iterators[s].i),tf.int32)
                    else:
                        hold = tf.placeholder(self[var][s][0].dtype,
                                shape=self[var][s].shape,name = var+s+'Holder')
                        self.tf_variables[var][s]=(hold,
                                        tf.Variable(hold,trainable=False))
                        batch = tf.gather(self.tf_variables[var][s][1],
                                    self.iterators[s].i)
                    pairs.append(batch)

                self.__dict__[var] = case(self.current_set,pairs)

    def preprocess(self,method,data,fitting_sets="train_set",
                        inplace=True, **kwargs):
        self.preprocessing = method(**kwargs)
        if inplace:
            self._preprocess_inplace(data=data,fitting_sets=fitting_sets)
        else:
            # loop over all the sets
            # Check if the data was already saved before from a prior
                # call or if it has to be saved now
            if '_saved_'+data in self[data]:
                saveit = False
            else:
                saveit = True
            for key in self[data].keys():
                # if we have to save the data, first time this method is
                # called without inplace
                if saveit:
                    self[data]["_save_"+key]=copy.deepcopy(self[data][key])
                # else, it was already saved up, we just have 
                # to put it back to original
                else:
                    self[data][key] = copy.deepcopy(self[data]["_save_"+key])
            # now perform the preprocess inplace
            self._preprocess_inplace(data=data,fitting_sets=fitting_sets)
    def _preprocess_inplace(self,data,fitting_sets="train_set"):
        # first accumulate all the data from each set 
        # to fit the preprocessor
        if type(fitting_sets)==str:
            self.preprocessing.fit(self[data][fitting_sets])
            for key in self[data].keys():
                self[data][key] = self.preprocessing.transform(self[data][key])
                print(key)
        else:
            # gather all the set data and stack them vertically
            all_data = np.concatenate([self[data][s] for s in fitting_sets],0)
            self.preprocessing.fit(all_data)
            for key in self[data].keys():
                self[data][key]=self.preprocessing.transform(self[data][key])
    def split_set(self,set_,new_set_,ratio, stratify = None):
        variables = self.variables
        if stratify is not None:
            assert(len(y)>1)
            train_indices = list()
            valid_indices = list()
            for c in set(y):
                c_indices = np.where(y==c)[0]
                np.random.shuffle(c_indices)
                limit = int(len(c_indices)*test_ratio)
                print(limit,len(c_indices))
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
            indices  = np.random.permutation(self.__dict__["n_"+set_])
            const    = int(self.__dict__["n_"+set_]*(1-ratio))
            for var in variables:
                self[var][new_set_]=self[var][set_][indices[const:]]
                self[var][set_]=self[var][set_][indices[:const]]
            # then we update the sizes
            self.add_variable({})

    @property
    def variables(self):
        return [k for k in self.keys() if type(k)==str]

    @property
    def sets(self):
        return [k for k in self[self.variables[0]].keys() if type(k)==str]

    def N(self,context):
        return len(self[self.variables[0]][context])

