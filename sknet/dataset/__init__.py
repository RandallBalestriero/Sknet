#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
from .. import DataArray

__all__ =["preprocess"]


class Dataset(dict):
    def __init__(self,**args):
        super().__init__()
        self.__dict__.update(args)
    def add_variable(self,dict_):
        self.update(dict_)
        variables = self.variables
        for set_ in self.sets:
            lengths = [len(self[v][set_]) for v in variables]
            assert(len(set(lengths))<=1)
            self.__dict__["n_"+set_]=lengths[0]
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



from .mnist import *
from .svhn import *
from .cifar10 import *
from .cifar100 import *
from .fashionmnist import *
from .freefield1010 import *
from .warblr import *
from .stl10 import *
from .custom import *
from .mini import *



from .preprocess import *
