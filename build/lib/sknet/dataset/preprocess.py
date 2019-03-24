#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time


class identity:
    def __init__(self,eps=0.0001):
        self.name = '-preprocessing(identity)'
    def fit(self,x,axis=[0],**kwargs):
        pass
    def fit_transform(self,x,**kwargs):
        return x
    def transform(self,x,**kwargs):
        return x


class standardize:
    def __init__(self,eps=0.0001):
        self.name = '-preprocessing(standardize,eps='+str(eps)+')'
        self.eps = eps
    def fit(self,x,axis=[0],**kwargs):
        print("Fitting standardize preprocessing")
        t=time.time()
        self.mean = x.mean(axis=tuple(axis),keepdims=True)
        self.std  = x.std(axis=tuple(axis),keepdims=True)+self.eps
        print("Fitting standardize preprocessing done in {0:.2f} s.".format(time.time()-t))
    def transform(self,x,**kwargs):
        return (x-self.mean)/self.std
    def fit_transform(self,x,**kwargs):
        self.fit(x)
        return self.transform(x)






class zca_whitening:
    def __init__(self,eps = 0.0001):
        self.name = '-preprocessing(zcawhitening,eps='+str(eps)+')'
        self.eps = eps
    def fit(self,x):
        print("Fitting zca_whitening preprocessing")
        t=time.time()
        flatx         = np.reshape(x, (x.shape[0], -1))
        self.mean     = flatx.mean(0,keepdims=True)
        self.S,self.U = _spectral_decomposition(flatx-self.mean,self.eps)
        print("Fitting zca_whitening preprocessing done in {0:.2f} s.".format(time.time()-t))
        return _zca_whitening(flatx,self.U,self.S).reshape(x.shape)
    def transform(self,x):
        flatx         = np.reshape(x, (x.shape[0], -1))-self.mean
        return _zca_whitening(flatx,self.U,self.S).reshape(x.shape)
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)





def _spectral_decomposition(flatx,eps):
    U, S, V= np.linalg.svd(flatx,full_matrices=False)
    S        = np.diag(1. / np.sqrt(S + eps))
    return S,V


def _zca_whitening(flatx, U, S):
    M     = np.dot(np.dot(U.T,S),U)
    return np.dot(M,flatx.T).T












