#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import time
import re
import h5py
import copy

__all__ = [
        "dataset",
        "layers",
        "ops",
        "utils",
        "network",
        "optimize",
        "sampler"]


__version__ = 'alpha.1'





def get_tensor_dependencies(tensor):
    dependencies = list()
    ops = tf.contrib.graph_editor.get_backward_walk_ops(tensor,control_inputs=True)
    for op in ops:
        if op.type == 'Placeholder' and 'deterministic' not in op.name:
            dependencies.append(op.outputs[0])
    return dependencies

def get_layers(tensor):
    layers = list()
    ops = tf.contrib.graph_editor.get_backward_walk_ops(tensor,
                                                    control_inputs=True)
    for op in ops:
        print(op.outputs,isinstance(op.outputs[0],layer.Layer))
        if isinstance(op.outputs[0],layer.Layer):
            layers.append(op)
    return layers



def to_file(value,filename,mode='w',compression_level=4):
    """dump the content of a worker, workergroupe, or
    list/tuple of them into h5 files
    if given with mode='w' then dump everything,
    if given with mode='a' then dump only the non present keys
    """
    # if given a single worker
    if type(value)==Worker:
        f = h5py.File(filename,mode)
        try:
            f[value.name+'/description']=value.description
        except:
            1
            #already wrote it
        data = value.get_concatenate()
        name = value.name+'/'+str(value.start)
        if np.isscalar(data):
            f.create_dataset(name,shape)
        else:
            f.create_dataset(name,data.shape,compression='gzip',
                    compression_opts=compression_level)
        f[name][...] = data
        f.close()
    # given a workergroup
    elif type(value)==WorkerGroup:
        for worker in value.workers:
            to_file(worker,filename,mode,compression_level=compression_level)
    elif hasattr(value,'__len__'):
        for item in value:
            to_file(item,filename,mode,compression_level=compression_level)


def from_file(filename):
    f = h5py.File(filename,'r')
    return f



class Queue(tuple):
    def __new__(cls,*args,**kwargs):
        obj = super(Queue,cls).__new__(cls,*args,**kwargs)
#        self.filename = filename
        return obj
#    def __init__(self,*args,**kwargs):
#        self.first_time_writting = True
    def dump(self,filename,flush=False):
        """Method to be called to save data from workers and empty them
        """
        while True:
#            try:
            to_file(self,filename,mode='a')
#            except:
#                print('Can not open file',filename,'... retrying in 5 sec')
#                time.sleep(5)
#                continue
            break
        if flush:
            for worker in self:
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





class Variable(tf.Variable):
    """Analog to :class:`tf.Variable` used for initialization
    with additional inplace option. There are two behaviors for this class
        - the option :data:`trainable` is a boolean: this variable will be
          a :class:`tf.Variable` initialized with the given input or
          function, and with trainable flag set as the given one.
        - the option :data:`trainable` is :data:`None`: this variable will
          just be a :class:`tf.Tensor` with value given by the given input
          or function. This option is usefull when using tied weights
          between different layers, the tied layers should simply use the
          original layer weights in place, and not create fixed or
          independent parameters.

    Example of use ::

        # Let's demonstrate the behavior of this class which can then be
        # used with layers, ops or other objects.
        # By default, trainable is set to True. When called, the following 
        zero_init = sknet.Variable(tf.zeros)
        # Create one trainable variable initialized with 0s
        var1 = zero_init((32,32))
        # When setting trainable=False, the same occurs but returns
        # a fixed (non trainable variable)
        zero_init = sknet.Variable(tf.zeros,trainable=False)
        # the below is equivalent to tf.zeros((32,32))
        var1 = zero_init((32,32))
        # Finally, it is often required to use inplace the given
        # parameter (for example in the reconstruction part of an
        # autoencoder, we wish to use the encoder weights inplace,
        # And not as initialization of a :class:`tf.Variable`.
        # To do so, set trainable=None as
        w_t = sknet.Variable(tf.transpose(encoder.W),trainable=None)
        # this is equivalent to w_t = tf.transpose(encoder.W)

    Parameters
    ----------

    var_or_func : tf.Tensor or func
        The :class:`tf.Tensor` or :class:`np.ndarray` to use, or
        the function to use, if a function, it will be given
        the shape and should return a :class:`tf.Tensor` or
        :class:`np.ndarray`.

    trainable : bool or None
        If a boolean, then the variable will be a :class:`tf.Variable`
        with trainable attribute set with the given one. If :data:`None`
        then the variable will not be a :class:`tf.Variable but` directly
        the tensor. This should be used if the passed value or function
        should not be used as an initializer of a :class:`tf.Variable`
        but as the parameter itself (inplace).
    """
    def __new__(cls,*args,**kwargs):
        obj = tf.Variable.__new__(cls)
        if 'trainable' in kwargs:
            obj._trainable = kwargs['trainable']
        else:
            if len(args)>1:
                obj._trainable = args[0]
            else:
                obj._trainable = True
        return obj

    @property
    def trainable(self):
        return self._trainable






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
    def __init__(self,op_name,context,op, instruction, deterministic=False,
                    repeat=1,description='',sampling='continuous'):
        self._dependencies  = get_tensor_dependencies(op)
        self._op_name       = op_name
        self._name          = context+"/"+op_name
        self._repeat        = repeat
        self._description   = description
        self._instruction   = instruction
        self._deterministic = deterministic
        self._sampling      = sampling
        self._context       = context
        self._op            = op
        self._epoch         = 0
        self._start         = 0
        # to gather all the epoch/batch data
        self.data           = list()
        # to gather data (batch) at each epoch
        self.epoch_data     = list()
        # specific to the saving and transformation
        instr = Parser(instruction)
        self.__dict__.update(instr)
        self._concurrent = True
    def alter(self,**kwargs):
        init = {'op_name':self.op_name,
                'context':self.context,
                'op':self.op, 
                'instruction':self.instruction, 
                'deterministic':self.deterministic,
                'repeat':self.repeat,
                'description':self.description,
                'sampling':self.sampling}
        init.update(kwargs)
        obj = Worker(**init)
        return obj
    def empty(self):
        self._start         = self.epoch
        # to gather all the epoch/batch data
        self.data           = list()
        # to gather data (batch) at each epoch
        self.epoch_data     = list()
    @property
    def instruction(self):
        return self._instruction
    @property
    def sampling(self):
        return self._sampling
    @property
    def description(self):
        return self._description
    @property
    def op_name(self):
        return self._op_name
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
    def epoch(self):
        return self._epoch
    @property
    def start(self):
        return self._start
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
    def get_concatenate(self):
        try:
            return np.concatenate(self.data)
        except:
            return np.asarray(self.data)
    def epoch_done(self,print_ = True):
        to_print=''
        self._epoch +=1
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
    """Allow executing of multiple workers at once (similar to multi-threading).
    it is common to need to monitor or execute multiple workers during the 
    same process. Typical example would be to execute the weight update 
    (learning) while monitoring (printing or saving) the loss. those two 
    workers can thus be made parallel simply by using the 
    :py:class:`sknet.WorkerGroup` class. To obtain a :class:`WorkerGroup`
    one can do any of the followings ::

        worker1 = sknet.Worker(...)
        worker2 = sknet.Worker(...)
        workergroup = sknet.WorkerGroup([worker1,worker2])
        workergroup = worker1+worker2

    Similarly, once given a :class:`WorkerGroup` instance, it is possible to add
    workers (or evenwWorergroup) on the fly as follows ::
        
        extra_worker = sknet.Worker(...)
        workergroup = workergroup+extra_worker
        # or if adding an extra workergroup
        workergroup=workergroup+other_workergroup

    A typical example would be to do the following ::

        worker1 = sknet.Worker(op_name="minimizer",op=minimizer,
                    context="train_set",deterministic=False,
                    instruction="execute every batch")
        worker2 = sknet.Worker(op_name='loss',op=loss,context='train_set',
                    deterministic=False, instruction="print every 30 batch")
        train_worker = worker1+worker2
        # then train
        # ...

    Parameters
    ----------

    worker : list of sknet.Worker
        the list of workers that are to be run concurrently during an epoch

    """
    def __init__(self,workers,name=''):
        self._workers      = [workers[0]]
        self._dependencies = workers[0].dependencies
        self._sampling     = workers[0].sampling
        self._op           = [worker.op for worker in workers]
        self._repeat       = workers[0].repeat
        self._context      = workers[0].context
        self._deterministic_list = [workers[0].deterministic]
        self.set_deterministic()
        self._name = workers[0].name
        for worker in workers[1:]:
            self.__add__(worker)
    def empty(self):
        for worker in self.workers:
            worker.empty()
    def __add__(self,other):
#        assert(self.sampling==other.sampling)
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

