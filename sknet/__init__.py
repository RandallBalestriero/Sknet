#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import time
import re
import h5py

__all__ = [
        "dataset",
        "layer",
        "utils",
        "network",
        "optimize",
        "sampler"]


__version__ = 'alpha.1'





def get_tensor_dependencies(tensor):

    # If a tensor is passed in, get its op
    try:
        tensor_op = tensor.op
    except:
        tensor_op = tensor

    # Recursively analyze inputs
    dependencies = []
    for inp in tensor_op.inputs:
        new_d = get_tensor_dependencies(inp)
        non_repeated = [d for d in new_d if d not in dependencies]
        dependencies = [*dependencies, *non_repeated]

    # If we've reached the "end", return the op's name
    if tensor_op.type == 'Placeholder':
        dependencies = [tensor]

    # Return a list of tensor op names
    return dependencies




def to_file(value,filename,mode='w',compression_level=4):
    """dump the content of a worker, workergroupe, or
    list/tuple of them into h5 files
    if given with mode='w' then dump everything,
    if given with mode='a' then dump only the non present keys
    """
    # if given a single worker
    if type(value)==Worker:
        f = h5py.File(filename,mode)
        # if append mode
        if mode=='a':
            if value.name not in f.keys():
                f.create_group(value.name)
                f[value.name+'/description']=value.description
            written = f[value.name].keys()
            for i in range(len(value.data)):
                if str(i) in written:
                    continue
                else:
                    shape = np.shape(value.data[i])
                    if np.isscalar(value.data[i]):
                        f.create_dataset(value.name+'/'+str(i),shape)
                    else:
                        f.create_dataset(value.name+'/'+str(i),shape,
                            compression='gzip',
                            compression_opts=compression_level)
                    f[value.name+'/'+str(i)][...] = value.data[i]
        else:
            f.create_group(value.name)
            f[value.name+'/description']=value.description
            for i in range(len(value.data)):
                shape = np.shape(value.data[i])
                if np.isscalar(value.data[i]):
                    f.create_dataset(value.name+'/'+str(i),shape)
                else:
                    f.create_dataset(value.name+'/'+str(i),shape,
                            compression='gzip',
                            compression_opts=compression_level)
                f[value.name+'/'+str(i)][...] = value.data[i]
        f.close()
    # given a workergroup
    elif type(value)==WorkerGroup:
        if mode=='w':
            f = h5py.File(filename,mode)
            f.close()
        for worker in value.workers:
            to_file(worker,filename,'a',compression_level=compression_level)
    elif hasattr(value,'__len__'):
        if mode=='w':
            f = h5py.File(filename,mode)
            f.close()
        for item in value:
            to_file(item,filename,'a',compression_level=compression_level)


def from_file(filename):
    f = h5py.File(filename,'r')
    return f





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
    """Overloading the :py:class:`tf.Tensor`
    """
    def __init__(self,tensor_or_func=None, shape=None, dtype = None):
        if tensor_or_func is None:
            tensor_or_func = tf.zeros
        if callable(tensor_or_func):
            tensor_or_func = tensor_or_func(shape,dtype=dtype)
        if not hasattr(tensor_or_func,'_op'):
            tensor_or_func = tf.identity(tensor_or_func)

        self._op          = tensor_or_func._op
        self._value_index = tensor_or_func._value_index
        self._dtype       = dtypes.as_dtype(tensor_or_func._dtype)
        self._tf_output   = None
        self._shape_val   = self._c_api_shape()
        self._consumers   = tensor_or_func._consumers
        self._id          = tensor_or_func._id



# instruction
def Parser(instruction):
    """Parser to transform string into boolean variables.

    command2 is applied at the end of the batche

    command1 every time1 and command2 

    time1 = - batch (same as 1 batch)
            - (int) batch
            - epoch

    command1 = - (same as execute)
               - save (will gather the output into list)
               - print (will print the output)
               - any combination of the above linked with &

    command2 = same as command1 with extra possibility of giveing
               - average or maximum (a function to be applied)

    """
    splitting = instruction.split('and')
    part1     = splitting[0]
    command1, periodicity = part1.split('every')
    periodicity = periodicity.replace(' ','')
    if periodicity[0].isdigit():
        periodicity = re.findall(r'\d+', periodicity)[0]
    else:
        periodicity = 1
    if len(splitting)>1:
        command2 = splitting[1]
    else:
        command2 = ''
    if 'average' in command2:
        func = lambda x:np.mean(x,0)
        transform = True
    elif 'maximum' in command2:
        func = lambda x:np.max(x,0)
        transform = True
    else:
        func = lambda x:x
        transform = False
    # if we already save the results in batch pass, then
    # do not save after
    if 'save' in command1:
        command2.replace('save','')
    attr = dict([('batch_instr',command1),('epoch_instr',command2),
                ('periodicity',int(periodicity)),('transform',transform),
                ('func',func)])
    return attr





class Worker(object):
    def __init__(self,op_name,context,op, instruction, deterministic=None,repeat=1,description=''):
        self._dependencies = get_tensor_dependencies(op)
        self._name          = context+"/"+op_name
        self._repeat        = repeat
        self._description   = description
        self._deterministic = deterministic
        self._context       = context
        self._op           = op
        # to gather all the epoch/batch data
        self.data          = list()
        # to gather data (batch) at each epoch
        self.epoch_data    = list()
        # specific to the saving and transformation
        instr = Parser(instruction)
        self.__dict__.update(instr)
        self._concurrent = True
    @property
    def description(self):
        return self._description
    @property
    def deterministic(self):
        return self._deterministic
    @property
    def concurrent(self):
        return self._concurrent
    @property
    def dependencies(self):
        return self._dependencies
    @property
    def op(self):
        return self._op
    @property
    def repeat(self):
        return self._repeat
    @property 
    def name(self):
        return self._name
    @property
    def context(self):
        return self._context
    def get_op(self,batch_nb):
        if batch_nb%self.periodicity==0:
            return self.op
        else:
            return []
    def append(self,data,print_=True):
        if data==[]:
            return ''
        to_print = ''
        if 'save' in self.batch_instr or self.transform:
            self.epoch_data.append(data)
        if 'print' in self.batch_instr:
            to_print = self.name+':'+str(data)
            if print_:
                print(to_print)
        return to_print

    def epoch_done(self,print_ = True):
        to_print=''
        if len(self.epoch_data)>0:
            if hasattr(self.epoch_data[0],'__len__'):
                self.epoch_data = np.concatenate(self.epoch_data,0)
            else:
                self.epoch_data = np.asarray(self.epoch_data)
        # apply transform if needed
        # then print if needed
        # then save if needed
        if self.transform:
            transform = self.func(self.epoch_data)
            if 'print' in self.epoch_instr:
                to_print = self.name+':'+str(transform)
                if print_:
                    print(to_print)
            if 'save' in self.epoch_instr:
                self.data.append(transform)
        # save into the main data collection
        if 'save' in self.batch_instr:
            self.data.append(self.epoch_data)
        # reset epoch data
        self.epoch_data = list()
        return to_print
    def __add__(self,other):
        return WorkerGroup([self,other])

    def __radd__(self,other):
        return self.__add__(other)




class WorkerGroup(Worker):
    def __init__(self,workers,name=''):
        self._workers      = [workers[0]]
        self._dependencies = workers[0].dependencies
        self._op           = [worker.op for worker in workers]
        self._repeat       = workers[0].repeat
        self._context      = workers[0].context
        self._deterministic_list = [workers[0].deterministic]
        self.set_deterministic()
        self._name = workers[0].name
        for worker in workers[1:]:
            self.__add__(worker)
    def __add__(self,other):
        assert(self.repeat==other.repeat)
        assert(other.context==other.context)
        self._dependencies = list(set(self._dependencies+other.dependencies))
        if type(other)==WorkerGroup:
            self._deterministic_list += list(other.deterministic)
            self._workers            += other.workers
            self._op                 += other.op 
        else:
            self._deterministic_list += [other.deterministic]
            self._workers            += [other]
            self._op                 += [other.op]
        self.set_deterministic()
        self._name = str(tuple([w.name for w in self.workers])).replace(' ','').replace('\'','')
        return self
    def __radd__(self,other):
        return self.__add__(other)
    def set_deterministic(self):
        if len(set(self._deterministic_list))==1:
            self._concurrent    = True
            self._deterministic = self._deterministic_list[0]
        else:
            self._concurrent = False

    @property
    def deterministic_list(self):
        return self._deterministic_list
    @property
    def workers(self):
        return self._workers

    def get_op(self,batch_nb):
        ops = [worker.get_op(batch_nb) for worker in self.workers]
        return ops

    def append(self,data):
        to_print = ''
        for datum,worker in zip(data,self.workers):
            to_print+=worker.append(datum,print_=False)
        if len(to_print)>0:
            print(to_print)

    def epoch_done(self):
        to_print=''
        for worker in self.workers:
            to_print+=worker.epoch_done(print_=False)
        if len(to_print)>0:
            print(to_print)







class ObservedTensor(Tensor):
    """Tensor with dual behavior
    
    This tensor is doing 
    
    Parameters
    ----------
    
    tensor : tf.Tensor
        the tensor to equip with dual behavior
        
    """
    def __init__(self, tensor, observation=None, 
                        teacher_forcing=None):
        self._observed        = observed
        self._teacher_forcing = teacher_forcing

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
        super().__init__(output)


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

