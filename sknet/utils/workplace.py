#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import h5py
#ToDo set the seed for the batches etc




class Workplace(object):
    def __init__(self, network, dataset=None):
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement     = True
        self.session                    = tf.Session(config=config)
        # Attributes
        self.dataset  = dataset
        self.network  = network
        # initialize the variables
        ops = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.session.run(ops,feed_dict=dataset.init_dict)

    def execute_op(self,op,feed_dict,deterministic=None):
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
        # if a deterministic behavior is given, set it
        if deterministic is not None:
            feed_dict.update(self.network.deter_dict(deterministic))

        output = self.session.run(op,feed_dict=feed_dict)
        return output
 
    def execute_worker(self,worker,feed_dict={}):
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
        # Loop over all the batches
        self.dataset.set_set(worker.context,session=self.session)
        feed_dict.update(self.network.deter_dict(worker.deterministic))
        batch_nb = 0
        while self.dataset.next(session=self.session):
            op = worker.get_op(batch_nb)
            worker.append(self.session.run(op,feed_dict=feed_dict))
            batch_nb+=1
        self.session.run(worker.epoch_done())

 
    def execute_queue(self,queue, repeat=1,feed_dict={}, close_file=True):
        """Apply multiple consecutive epochs of train test and valid

        Example of use ::

            train_ops = [[minimizer_op,1],
                         [loss,30]]
            test_ops  = [[accuracy,1,lambda x:np.mean(x,0)],
                         [loss,1,lambda x:np.mean(x,0)]]
            valid_ops = test_ops

            # will fit the model for 50 epochs and return the gathered op
            # outputs given the above definitions
            outputs = pipeline.fit(train_ops,valid_ops,test_ops,n_epochs=50)
            train_output, valid_output, test_output = outputs

        Parameters
        ----------

        repeat : int (default to 1)
            number of times to repeat the fit pattern

        contexts : list of str (optional, default None)
            the list describing the ops and the number of times
            to execute them. For example, suppose that during
            consturction, ops were added for context
            ``"train_set"`` and ``"test_set"`` and ``"valid_set"``,
            then one could do ::

                # the pipeline fit will perform on epoch for 
                # each and then going to the next, and this
                # :py:data:`repeat` times, which would lead to 
                # 1 epoch train_set -> 1 epoch valid_set
                # -> 1 epoch test_set ->1 epoch train_set
                # -> 1 epoch valid_set ...
                contexts = ("train_set","valid_set","test_set")
                # If one context needs to be execute more than 1
                # epoch prior going to the next, then use
                contexts = (("train_set",10),"valid_set","test_set")
                # this would make the pipeline do 10 epochs of train_set
                # prior doing 1 epoch of valid_set and 1 epoch of
                # test_set and then starting again with 10 epochs
                # of train_set. again, this process is reapeted 
                # :py:data:`repeat` times


        """
        for e in range(repeat):
            print("Repeat",e)
            for worker in queue:
                name     = worker.name
                print("\trunning Worker:",name)
                self.execute_worker(worker,feed_dict=feed_dict)
            queue.dump()
        if close_file:
            queue.close()


































