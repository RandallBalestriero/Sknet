#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
import h5py
import numpy as np
import tensorflow as tf
from ..base import StreamingTensor

# ToDo set the seed for the batches etc


class Workplace:
    """this class is the core of the active operation execution.
       the first time it is called (at initialization) the tf.Session is
        created and the initialization happens. This is why it is necessary
       to give the dataset (if one is used) because sknet.Dataset generates
       some special initialization cases."""
    def __init__(self, dataset=None):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(config=config)
        # Attributes
        self.dataset = dataset
        # initialize the variables
        ops = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        if dataset is not None:
            self.session.run(ops, feed_dict=dataset.init_dict)
        else:
            self.session.run(ops)
        self._file = None

    def init_file(self, filename):
        if self._file is not None:
            self._file.close()
        while 1:
            try:
                self._file = h5py.File(filename, 'w',
                                       libver='latest')
                break
            except:
                print('Could not open file ', self._filename)
                print('\tRetrying in 10 sec. ...')
                time.sleep(10)
        self._h5_dataset = dict()
        self._count = dict()
        for worker, name, op in tf.get_collection('vars_to_save'):
            savename = worker.name+"/"+name
            op_shape = tuple(op.shape.as_list())
            if not isinstance(op, StreamingTensor):
                maxshape = (None, None,)+tuple(op_shape)
                shape = (1, 1,)+op_shape
            else:
                maxshape = (None,)+tuple(op_shape)
                shape = (1,)+op_shape
            data = np.zeros(shape, dtype=op.dtype.as_numpy_dtype)
            dataset = self._file.create_dataset(savename, dtype=op.dtype.as_numpy_dtype,
                                                maxshape=maxshape,
                                                data=data,
                                                compression='gzip')
            self._h5_dataset[savename] = dataset
            self._count[savename] = 0
        self._file.swmr_mode = True

    def dump(self, worker):
        if self._file is None:
            return
        for name, data in worker.epoch_data.items():
            if type(data) == list:
                if len(data) == 0:
                    continue
            savename = worker.name+"/"+name
            self._count[savename] += 1
            new_shape = list(self._h5_dataset[savename].shape)
            new_shape = (self._count[savename],) + data[0].shape
            self._h5_dataset[savename].resize(new_shape)
            self._h5_dataset[savename][-1] = data[-1]
            self._h5_dataset[savename].flush()
        worker.empty()

    def close(self):
        """close the session"""
        self.session.close()

    def execute_op(self, op, feed_dict):
        """Executes the given ```op``` with the current session and the
        given ```feed_dict```.

        Parameters
        ----------

        op : (one or list of) tf.Tensor
            the op (or ops as a list of ops) to run for one epoch or
            a list of the form [(op1,periodicity, optional np op)]
            where op1 can again be a list.

        feed_dict : dict
            the dictionnary to be feed for executing the op(s).

        Returns
        -------

        output : (one or list of) tf.Tensor
            the output of executing the op input.

        """
        output = self.session.run(op, feed_dict=feed_dict)
        return output

    def _execute_worker(self, worker):
        """used for one by one worker
        """
        # update the feed_dict with deterministic behavior value
        feed = worker.feed_dict.copy()
        self.dataset.iterator.reset(worker.context)
        feed.update({self.dataset.iterator.set: worker.context})
        for i in tqdm(range(self.dataset.N_BATCH(worker.context)),
                      desc=worker.name):
            feed.update(self.dataset.iterator.next(worker.context))
            ops = worker.get_ops(batch=i)
            worker.append(self.execute_op(ops, feed_dict=feed))
        # signal the end of epoch and execute any needed reset op
        self.session.run(worker.epoch_done())
        self.dump(worker)

    def execute_worker(self, worker, repeat=1):
        """Perform an epoch (according to the set given by context).
        """
        repeat = range(1) if repeat == 1 else tqdm(range(repeat), desc='Epoch')
        if not hasattr(worker, '__len__'):
            worker = [worker]
        for _ in repeat:
            for w in worker:
                self._execute_worker(w)
            print('\n')
