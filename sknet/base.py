#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import h5py
from tensorflow.contrib.graph_editor import get_backward_walk_ops

ZERO_INT32 = tf.constant(np.int32(0))
ZERO_FLOAT32 = tf.constant(np.float32(0))
ONE_INT32 = tf.constant(np.int32(1))
ONE_FLOAT32 = tf.constant(np.float32(1))

def EMA(tensor, decay, step):
    ma_value = tf.Variable(tf.zeros_like(tensor), trainable=False, name='ma')
    ma = ma_value*decay+tensor*(1-decay)
    value = tf.cond(tf.greater(step, ZERO_INT32), lambda :ma, lambda :tensor)
    update = tf.assign(ma_value, value)
    return ma_value, update

def get_tensor_dependencies(tensor):
    dependencies = list()
    ops = list()
    for t in tensor:
        ops.append(get_backward_walk_ops(t, control_inputs=True,
                   inclusive=False))
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
        obj._file = None
        obj.count = 0
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
                    self._file = h5py.File(self._filename, 'w', libver='latest')
                    break
                except:
                    print('Could not open file ', self._filename)
                    print('\tRetrying in 10 sec. ...')
            # init the arrays, get shape and init
            h5_dataset = list()
            for worker in self:
                h5_dataset.append(dict())
                for name, data in worker.epoch_data.items():
                    maxshape = (None,)+data[0].shape
                    savename = worker.context+"/"+name
                    h5_dataset[-1][name] = self._file.create_dataset(
                            savename,  maxshape=maxshape,
                            compression='gzip', data=np.expand_dims(data[0], 0))
                worker.empty()
            self.h5_dataset = h5_dataset
            self._file.swmr_mode = True
        else:
            for i, worker in enumerate(self):
                for name, data in worker.epoch_data.items():
                    new_shape = (self.count,)+data[0].shape
                    self.h5_dataset[i][name].resize(new_shape)
                    self.h5_dataset[i][name][self.count-1] = data[0]
                    self.h5_dataset[i][name].flush()
                worker.empty()


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


class StreamingTensor(Tensor):
    """base class for streaming tensors. A streaming tensor
    performs some type of aggregate (such as mean or max) over
    all the batches for each epoch. A typical example would be the
    :class:`sknet.losses.streaming_mean` which computes the average
    per batch,over all the batches. This can be used to compute the
    average accuracy on some batched dataset rather than getting all
    the values back in python and then computig the average.
    Streaming tensors work by creating an accumulator and keeping on
    updating it for each batch until completion of the epoch.
    If used without the :class:`sknet.Workplace` then one needs to be careful
    to execute the ``reset_variables_op``, to reset the
    accumulator value and any accompanying variables.

    Parameters:
    -----------

    value: tensor
        the current value of the running tensor, the value that this 
        tensor should hold

    update: op
        the update operation that should be run on each batch to keep
        updating the value until epoch completion

    variables: tensor or list of tensors
        all the :class:`tf.variables` used in the computation of the
        value, used to generate the reset op to set back all the needed
        variables at their initial value once an epoch has been completed.
    """
    def __init__(self, value, update, variables):
        super().__init__(value)
        self.reset_variables_op = tf.initializers.variables(variables)
        self.update = update



class Worker(object):
    """processing unit that manages a tensorflow ops during batch iterations.
    A Worker allows to specify a context, such as train set, test set, or any
    context present in a datset, and tensorflow ops to be executed, and if the
    deep net behavior should be deterministic or not (for example during 
    testing). The ops are given as kwargs of the function, the names used will
    be the same used for saving (think of :class:`np.savez`. For more control
    it is possible to give a couple where the second item is a function that
    takes for input a batch number and epoch number. This allow further
    control if one wants to only execute an op every X batch or only after the
    10th epoch and so on. Here is an example::

        # an op updating the dn weight
        update_ op = ...
        # an op measuring the accuracy on each batch
        accu = ...
        # an op measuring the average accuracy over all batches on the set
        avg_accu = sknet.losses.streaming_mean(accu)
        # a tensor which is a weight of the Dn that we wish to monitor
        # and we want to monitor it only at the first batch of epoch 0, 10, 15
        W = ....
        def fW(batch, epoch):
            if batch == 0 and epoch in [0,10,15]:
                return True
            return False

        # we now create the worker
        worker = Worker(context = 'train_set', deterministic=False,
                        update = update_op, accu=accu, avg_accu = avg_accu, 
                        W=(dnn.W,fW))

    
    with the above example (notice that the index starts at 0), the worker
    will execute on the train set (in parallel at each batch) all the given
    and for the weight monitoring, will only save the output if the conditioin
    if fulfilled. The update op, as obtained from a tensorflow operation
    returns None values. This will not be saved by the worker. Hence, at the
    end of an epoch. The data saved in ```worker.epoch_data``` will be::

        worker.epoch_data['update'] # an empty list
        worker.epoch_data['accu'] # array of shape (N_EPOCH, N_BATCH,1)
        worker.epoch_data['avg_accu'] # array of shape (N_EPOCH, 1), there are
                                      # no batch dimension when using 
                                      # streaming tensors as they compute 
                                      # aggregated stats. over all batches of 
                                      # each epoch
        worker.epoch_data['W'] # array of shape (N_EPOCH, N_BATCH,*np.shape(W))


    NOTE: if using a :class:`sknet.Queue` with a given filename, the results
    are saved into the .h5 file at each epoch, and the worker data are emptied
    for efficiency, in such cases, the same data as described above will be 
    present in the .h5 file and not in ```worker.epoch_data```.

    Parameters
    ----------

    context: str
        the name of the set to apply the ops on from the dataset. Typical example
        would be :py;data;`"train_set"` or :py:data:'"valid_set"'

    deterministic: bool
        the state of the network to execute the op in, for example
        it is common to set it to :data:`False` during training
        and :data:`True` during testing.

    ops: kwargs
        any desired op or tensor to be executed/monitored. It can be a
        op or a couple with the second item being a function with input
        ```batch, epoch``` that should return a boolean value telling if
        the op is to be executed at this moment.
    """
    def __init__(self, context, deterministic, **kwargs):
        self._ops = list()
        self._op_names = list()
        cpt = 0
        for key, value in kwargs.items():
            if type(value)==tuple:
                assert len(value)==2
                self._ops.append((cpt, value[0], value[1]))
            else:
                self._ops.append((cpt, value, lambda b, e:True))
            self._op_names.append(key)
            cpt += 1

        self.current_ops = [[] for i in range(len(self._ops))]
        self._dependencies = get_tensor_dependencies([op[1]
                       for op in self._ops if not isinstance(op, tf.Variable)])
        self._deterministic = deterministic
        self._context = context
        self.empty()

    def empty(self):
        self.batch_data = [[] for i in range(len(self._ops))]
        self.epoch_data = dict()
        for name in self._op_names:
            self.epoch_data[name] = []

    @property
    def deterministic(self):
        return self._deterministic

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def ops(self):
        return self._ops

    @property
    def context(self):
        return self._context

    def get_ops(self, batch, epoch):
        """given a batch index and the current epoch, return the list
        of ops to be executed

        Parameters:
        -----------

        batch: int
            the batch number in the current epoch

        epoch: int
            the current epoch

        Returns:
        --------

        ops: list of ops
            the ops to be executed at this batch
        """
        for index, op, func in self.ops:
            if func(batch, epoch):
                if isinstance(op, StreamingTensor):
                    self.current_ops[index] = op.update
                else:
                    self.current_ops[index] = op
            else:
                self.current_ops[index] = []
        return self.current_ops

    def append(self, data):
        """given the session output obtained by executing the given list of ops,
        append the data with the batch values"""
        for i, d in enumerate(data):
            if isinstance(self._ops[i], StreamingTensor):
                self.batch_data[i] = d
            elif d is None:
                continue
            elif type(d) == list:
                if len(d) == 0:
                    continue
            else:
                self.batch_data[i].append(d)

    def epoch_done(self):
        """method to be executed after completion of an epoch.
        It executes all the needed reset and savings"""
        reset_op = list()
        for i, data in enumerate(self.batch_data):
            if isinstance(self._ops[i], StreamingTensor):
                self.epoch_data[self._op_names[i]].append(data)
                reset_op.append(self._ops[i][1].reset_variables_op)
            else:
                self.epoch_data[self._op_names[i]].append(np.asarray(data))

        # reset values and ops
        self.batch_data = [[] for i in range(len(self._ops))]
        return reset_op



class ObservedTensor(Tensor):
    """Tensor with dual behavior

    This tensor is doing

    Parameters
    ----------

    tensor : tf.Tensor
        the tensor to equip with dual behavior

    """
    def __init__(self, tensor, observation=None, teacher_forcing=None):
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



