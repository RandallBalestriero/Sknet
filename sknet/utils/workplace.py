#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import tensorflow as tf
import h5py
import numpy as np
import time
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
        self._h5_dataset = dict()

    def file(self, filename, workers):
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
        for worker in workers:
            for name, (op, _) in worker.named_ops.items():
                print(op.shape.as_list())
                savename = worker.name+"/"+name
                if not isinstance(op, StreamingTensor):
                    maxshape = (None, None, )+(None,) * len(op.shape.as_list())
                else:
                    maxshape = (None, )+(None,) * len(op.shape.as_list())
                shape = (1,) * len(maxshape)
                data = np.zeros(shape, dtype='float32')
                dataset = self._file.create_dataset(savename, dtype='float32',
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

        Returns
        -------

        output : (one or list of) tf.Tensor
            the output of executing the op input.

        """
        output = self.session.run(op, feed_dict=feed_dict)
        return output

    def _execute_worker(self, worker, deter_func, feed_dict):
        """used for one by one worker
        """
        # update the feed_dict with deterministic behavior value
        feed = feed_dict.copy()
        feed.update(deter_func(worker.deterministic))
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

    def execute_worker(self, worker, deter_func=lambda a: {},
                       feed_dict={}, repeat=1):
        """Perform an epoch (according to the set given by context).
        Execute and save the given ops and optionally apply a numpy function
        on them at the end of the epoch. THis is usefull for example to only
        return the average over the batches of the computed statistics, or the
        max....

        Example of use is ::

            # typical training setting
            op = [[minimizer_op,1],
                  [loss,30,np.mean]]
            pipeline.epoch(op=op,context='train_set',deterministic=False)

            # typical test setting, average over the batch accuracies
            op = [[accuracy,1,lambda x:np.mean(x,0)]]
            pipeline.epoch(op=op,context='test_set',deterministic=True)

            # Case where one is autonomous
            pipeline.epoch(op=[minimize,loss],context='train_set',
                            deterministic=False,
                            linkage={network[0]:X_train,loss.p:Y_train})


        Parameters
        ----------

        op : (one or list of) tf.Tensor or sknet.Op
            the op (or ops as a list of ops) to run for one epoch or
            a list of the form [(op1,periodicity, optional np op)]
            where op1 can again be a list. For example, during training,
            one might prefer to compute


        """
        if repeat == 1:
            repeat = range(1)
        else:
            repeat = tqdm(range(repeat), desc='Epoch :')
        for _ in repeat:
            if hasattr(worker, '__len__'):
                for w in worker:
                    self._execute_worker(w, deter_func, feed_dict)
            else:
                self._execute_worker(worker, deter_func, feed_dict)
