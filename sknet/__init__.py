#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import time

__all__ = [
        "dataset",
        "layer",
        "utils",
        "optimize",
        "network",
        "sampler"]


__version__ = 'alpha.1'



class DataArray(np.ndarray):
    def __new__(cls, input_array, partition=None, name='', info= ''):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj._partition = partition
        if name =='':
            obj._name = str(time.time())
        else:
            obj._name = name
        obj._info = info
        # Finally, we must return the newly created object:
        return obj
    def __getitem__(self,k):
        if type(k)==str:
            return super().__getitem__(self._partition[k])
        elif(type(k)==tuple):
            if(len(k)==2 and type(k[0])==str):
                return super().__getitem__(self.partition[k[0]][k[1]])
            else:
                return super().__getitem__(k)
        else:
            return super().__getitem__(k)

    # Binary operators
    def __add__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)+np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)+other,self.partition)
    def __radd__(self,other):
        return self+other
    def __sub__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)-np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)-other,self.partition)
    def __rsub__(self,other):
        return -1*(self-other)
    def __mul__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)*np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)*other,self.partition)
    def __rmul__(self,other):
        return self*other
    def __truediv__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)/np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)/other,self.partition)
    def __rtruediv__(self,other):
        return DataArray(other/np.asarray(self),self.partition)
    def __floordiv__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)//np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)//other,self.partition)
    def __rfloordiv__(self,other):
        return DataArray(other//np.asarray(self),self.partition)
    def __mod__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)%np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)%other,self.partition)
    def __rmod__(self,other):
        return DataArray(other%np.asarray(self),self.partition)
    def __pow__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)**np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)**other,self.partition)
    def __rpow__(self,other):
        return DataArray(other**np.asarray(self),self.partition)

    # Logical operators
    def __lt__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)<np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)<other,self.partition)
    def __rlt__(self,other):
        return self>=other
    def __gt__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)>np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)>other,self.partition)
    def __rgt__(self,other):
        return self<=other
    def __le__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)<=np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)<=other,self.partition)
    def __rle__(self,other):
        return self>other
    def __ge__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)>=np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)>=other,self.partition)
    def __rge__(self,other):
        return self<other
    def __eq__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)==np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)==other,self.partition)
    def __req__(self,other):
        return self==other
    def __ne__(self,other):
        if type(other)==DataArray:
            return DataArray(np.asarray(self)!= np.asarray(other),
                    {**self.partition,**other.partition})
        else:
            return DataArray(np.asarray(self)!=other,self.partition)
    def __rne__(self,other):
        return self!=other

    def set_partition(self,partition):
        self._partition = partition

    @property
    def partition(self):
        return self._partition

    @property
    def name(self):
        return self._name

    @property
    def info(self):
        return self._info



class Tensor(tf.Tensor):
    def __init__(self,tensor, observed=False, observation=None, 
                        teacher_forcing=None):
        self._observed = observed
        self._teacher_forcing = teacher_forcing
        if observed:
            if observation is None:
                self._observation = tf.placeholder_with_default(
                        tf.zeros(tensor.shape,dtype=tensor.dtype),
                        shape = tensor.shape,
                        name  = 'observation-'+tensor.name.replace(':','-'))
            else:
                self._observation = observation
            if teacher_forcing is None:
                self._teacher_forcing = tf.placeholder_with_default(False,
                        shape = (),
                        name  = 'teacherforcing-'+tensor.name.replace(':','-'))
            else:
                self._teacher_forcing = teacher_forcing
            output = tf.cond(self._teacher_forcing, lambda :self._observation, 
                                                    lambda :tensor)
        else:
            self._observation = None
            output = tensor
        self._op          = output._op
        self._value_index = output._value_index
        self._dtype       = dtypes.as_dtype(output._dtype)
        self._tf_output   = None
        self._shape_val   = self._c_api_shape()
        self._consumers   = output._consumers
        self._id          = output._id
    @property
    def observation(self):
        return self._observation
    
    @property
    def observed(self):
        return self._observed

    @property
    def teacher_forcing(self):
        return self._teacher_forcing


from . import *
