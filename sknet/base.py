#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import time
import re
import h5py
import copy
from tensorflow.contrib.graph_editor import get_backward_walk_ops

ZERO_INT32 = tf.constant(np.int32(0))
ZERO_FLOAT32 = tf.constant(np.float32(0))
ONE_INT32 = tf.constant(np.int32(1))
ONE_FLOAT32 = tf.constant(np.float32(1))

def EMA(tensor, decay, step):
    ma_value = tf.Variable(tf.zeros_like(tensor), trainable=False, name='ma')
    ma = ma_value*decay+tensor*(1-decay)
    value = tf.cond(tf.greater(step,ZERO_INT32),lambda :ma, lambda :tensor)
    update = tf.assign(ma_value, value)
    return ma_value, update

def get_tensor_dependencies(tensor):
    dependencies = list()
    ops = list()
    for t in tensor:
        ops.append(get_backward_walk_ops(t,control_inputs=True,inclusive=False))
    ops = list(set([o for op in ops for o in op]))
    for op in ops:
        if op.type == 'Placeholder' and 'deterministic' not in op.name:
            dependencies.append(op.outputs[0])
    return dependencies

def get_layers(tensor):
    layers = list()
    ops = tf.contrib.graph_editor.get_backward_walk_ops(tensor,
                                                    control_inputs=True)
    for op in ops:
        if isinstance(op.outputs[0],layer.Layer):
            layers.append(op)
    return layers





class Queue(tuple):
    def __new__(cls,*args,filename=None):
        obj = super(Queue,cls).__new__(cls,*args)
        obj._filename = filename
        obj._file     = None
        obj.count     = 0
        return obj
    def close(self):
        self.count = 0
        if self._file is not None:
            self._file.close()
    def dump(self):
        """Method to be called to save data from workers and empty them
        """
        if self._filename is None:
            return
        self.count+=1
        if self._file is None:
            # create and open the file
            while 1:
                try:
                    self._file = h5py.File(self._filename, 'w',libver='latest')
                    break
                except:
                    print('Could not open file ',self._filename)
                    print('\tRetrying in 10 sec. ...')
            # init the arrays, get shape and init
            h5_dataset = list()
            for worker in self:
                h5_dataset.append([])
                for i,data in enumerate(worker.epoch_data):
                    maxshape = (None,)+data[0].shape
                    h5_dataset[-1].append(self._file.create_dataset(
                            worker.name+"/"+str(i), maxshape=maxshape,
                            compression='gzip', data=np.expand_dims(data[0],0)))
                worker.empty()
            self.h5_dataset      = h5_dataset
            self._file.swmr_mode = True
        else:
            for i,worker in enumerate(self):
                for j,data in enumerate(worker.epoch_data):
                    new_shape = (self.count,)+data[0].shape
                    self.h5_dataset[i][j].resize(new_shape)
                    self.h5_dataset[i][j][self.count-1]=data[0]
                    self.h5_dataset[i][j].flush()
                worker.empty()


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




class Worker(object):
    """processing unit that manages a single tensorflow op.
    A Worker allows to specify a tensorflow op to execute. Whether
    it is for monitoring or saving or printing, the user should favor
    the use of Workers to simplify the workflow. A worker contains the
    operator to use, the ations to take with it given by an instruction

    Parameters
    ----------

    op_name : str
        the name of the worker, used for saving and printing

    context : str
        the name of the set to apply the op on from the dataset. Typical example
        would be :py;data;`"train_set"` or :py:data:'"valid_set"'

    op : tf.Tensor or tf.operation
        the tensorflow variable of the worker that will be executed
        and monitored

    instruction : str
        a description of what how and when to interact with the op. We provide
        some typical examples ::

            instruction = "execute every batch"
            # when given another command such as print, execute is
            # always added by default
            instruction = "print every 30 batch"
            # one can give multiple commands to do at the same time, they
            # are linked with a & a in
            instruction = "print 7 save every 30 batch"
            # one can also use the instruction to specify a standard
            # operation method to do after the epoch such as done with
            # accuracy where it is computed on each batch and then averaged
            instruction = "execute every batch and average & print"
            # as an be seen, the commands to do after the epoch are
            # introduced via the and keyword. One can also do something like
            intruction = "execute every bath and save & print & average"
            # the order of the commands 9around the &) do not matter
            # finally, if asking to save and the per batch value AND
            # the epoch value, then the last one is disregarded as it
            # can be computed a posteriori from the batch ones, for example
            instruction = "print&save every 30 batch and save & print & average"
            # in this case the previous case will save only the batch values.

        The second set of commands (after the ``"and"`` must contain either
        ``"average'`` or ``"maximize"``.

        deterministic : bool
            the state of the network to execute the op in, for example
            it is common to set it to :data:`False` during training
            and :data:`True` during testing.

        description : str (optional)
            the description of the worker, used for saving,
            it is added to the h5 file.
    """
    def __init__(self, name, context, op, deterministic=False, period=1,
                                        verbose=0):
        self._op = [op] if type(op)!=list else op
        self.n_ops = len(self._op)

        # Verbose
        self._verbose= [verbose]*self.n_ops if type(verbose)!=list else verbose
        assert(len(self._verbose)==self.n_ops)

        # Period
        self._period = [period]*self.n_ops if type(period)!=list else period
        assert(len(self._period)==self.n_ops)

        self._dependencies = get_tensor_dependencies([op
                       for op in self._op if not isinstance(op,tf.Variable)])
        self._name = context+"/"+name
        self._deterministic = deterministic
        self._context = context
        self.empty()

    def empty(self):
        self.batch_data = [[] for i in range(len(self._op))]
        self.epoch_data = [[] for i in range(len(self._op))]

    @property
    def deterministic(self):
        return self._deterministic

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def op(self):
        return self._op

    @property
    def name(self):
        return self._name

    @property
    def context(self):
        return self._context

    def get_op(self,batch_nb):
        ops = list()
        for per, op in zip(self._period, self._op):
            if batch_nb%per==0:
                if isinstance(op, StreamingLoss):
                    ops.append(op.update)
                else:
                    ops.append(op)
            else:
                ops.append([])
        return ops

    def append(self,data):
        print_string =''
        for i,d in enumerate(data):
            if type(d)==list:
                if len(d)==0:
                    continue
            elif d is None:
                continue
            self.batch_data[i].append(d)
            if self._verbose[i]>1:
                print_string+='Op'+str(i)+':'+str(d)+', '
        if len(print_string)>0:
            print('\t\t'+self.name+':: '+print_string[:-2])

    def epoch_done(self):
        print_string = ''
        for i,data in enumerate(self.batch_data):
            if isinstance(self._op[i],StreamingLoss):
                data = data[-1]
            self.epoch_data[i].append(np.asarray(data))
            if self._verbose[i]==1 or self._verbose[i]==3:
                print_string+= 'Op'+str(i)+':'+str(self.epoch_data[i][-1])+', '
        self.batch_data = [[] for i in range(len(self._op))]
        if len(print_string)>0:
            print('\tEpoch Done:: '+print_string[:-2])
        reset_op = list()
        for op in self._op:
            if isinstance(op,StreamingLoss):
                reset_op.append(op.reset_variables_op)
        return reset_op


from .losses import StreamingLoss                                               


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



