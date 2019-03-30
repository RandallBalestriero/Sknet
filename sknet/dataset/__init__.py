#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ =["preprocess"]


class Dataset(dict):
    def __init__(self,dict_init):
        super().__init__(dict_init)
    def load(self):
        pass
    def preprocess(self,method,fit="train_set",transform=[], **method_kwargs):
        self.preprocessing = method(**method_kwargs)
        print("\tpreprocessing with"+self.preprocessing.name)
        t = time.time()
        if len(self[fit])>1:
            self[fit][0]=self.preprocessing.fit_transform(self[fit][0])
        else:
            self[fit]=self.preprocessing.fit_transform(self[fit])
        for other in transform:
            if len(self[fit])==2:
                self[transform][0]  = self.preprocessing.transform(self[other][0])
            else:
                self[transform]  = self.preprocessing.transform(self[other])
        print("\tDone in {:.2f} s.".format(time.time()-t))

from .mnist import *
from .svhn import *
from .cifar10 import *
from .cifar100 import *
from .fashionmnist import *
from .freefield1010 import *
from .warblr import *
from .stl10 import *
from .custom import *




from .preprocess import *
