#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from .. import Worker

#ToDo set the seed for the batches etc




class Workplace(object):
    def __init__(self, network, dataset=None, linkage=None):
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement     = True
        self.session                    = tf.Session(config=config)
        # Attributes
        self.dataset  = dataset
        self.network= network
        self._linkage = linkage
        # initialize the variables
        self.session.run(tf.global_variables_initializer())

    def execute_op(self,op,feed_dict,batch_size=None,deterministic=None):
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
        N = max([len(value) for value in feed_dict.values()])
        # find out the batch_size, if not given, take it from the network, 
        # if given (case when the network was build with None 
        # as batch_size, then assert that
        # it is true (or at least that they both match)
        if batch_size is None and self.network.batch_size is None:
            batch_size = N
        elif batch_size is None and self.network.batch_size is not None:
            batch_size = self.network.batch_size
        else:
            print("error")
            exit()
        
        # if a deterministic behavior is given, set it
        if deterministic is not None:
            self.network.set_deterministic(deterministic,self.session)

        # Get number of data and batches
        N_BATCH = N//batch_size

        if N_BATCH>1:
            output = list()
            for i in range(N_BATCH):
                here  = range(i*batch_size,(i+1)*batch_size)
                feed_ = [(key,value[here]) for key,value in linkage.items()]
                output.append(self.session.run(op,feed_dict=feed_))
            # Concatenate the outputs over the batch axis
            if hasattr(op,'__len__'):
                output = np.concatenate([batch_out[i] 
                            for batch_out in output],0)
            else:
                output = np.concatenate(output,0)
        else:
            output = self.session.run(op,feed_dict=feed_dict)
        return output
 
    def execute_worker(self,worker):
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
        # find out the batch_size, if not given, take it from the network, 
        # if given (case when the network was build with None 
        # as batch_size, then assert that
        # it is true (or at least that they both match)
        batch_size = self.network.batch_size
        N = self.dataset.N(worker.context)
        N_BATCH = N//batch_size
        # get dependencies
        dependencies = worker.dependencies
        # get context
        context = worker.context
        if worker.concurrent:
            if worker.deterministic  is not None:
                self.network.set_deterministic(worker.deterministic,self.session)
        # Loop over all the batches
        for i in range(N_BATCH):
            # current indices
            here = range(i*batch_size,(i+1)*batch_size)
            # create the feed dict for the batch
            feed_dict = list()
            for var in dependencies:
                var_name = self.linkage[var.name]
                value = self.dataset[var_name][worker.context][here]
                feed_dict.append((var,value))
            # get the op(s)
            op = worker.get_op(i)
            # if we can run the ops concurently, then do it
            if worker.concurrent:
                output = self.session.run(op,feed_dict=dict(feed_dict))
            else:
                output = []
                for op_,deterministic in zip(op,worker.deterministic_list):
                    if op_==[]:
                        output.append([])
                        continue
                    if deterministic is not None:
                        self.network.set_deterministic(deterministic,self.session)
                    output.append(self.session.run(op_,feed_dict=dict(feed_dict)))
            worker.append(output)
        worker.epoch_done()
 
    def add_linkage(self,linkage):
        self._linkage = linkage
    @property
    def linkage(self):
        return self._linkage
    @property
    def workers(self):
        return self._workers
    def execute_queue(self,queue, repeat=1):
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
        for _ in range(repeat):
            print("Repeat",_)
            for worker in queue:
                n_epochs = worker.repeat
                name     = worker.name
                print("\trunning Worker:",name)
                for epoch in range(n_epochs):
                    if n_epochs>1:
                        print("\t  Epoch",epoch,'...')
                    self.execute_worker(worker)
        return queue






























class Trainer(object):
    def __init__(self, network, lr_schedule, optimizer = tf.train.AdamOptimizer, regression=False, 
            bn_training=True, display_name = ''):
        # Tensorflow Config
        tf.reset_default_graph()
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement     = True
        self.session                    = tf.Session(config=config)
        # Attributes
        self.display_name = display_name
        self.classes      = network.classes
        self.batch_size   = network.input_shape[0]
        self.regression   = regression
        self.bn_training  = bn_training
        self.lr_schedule  = lr_schedule
        with tf.device('/device:GPU:0'):
            self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
            self.x             = tf.placeholder(tf.float32, shape=network.input_shape,name='x')
            self.training      = tf.placeholder(tf.bool,name='phase')
            self.layers        = network.get_layers(self.x,training=self.training)
            self.prediction    = self.layers[-1].output
            self.number_of_params = 14#count_number_of_params()
            if(regression):
                self.y        = tf.placeholder(tf.float32, shape=[network.input_shape[0]],name='y')
                self.loss     = tf.reduce_sum(tf.nn.l2_loss(self.prediction-self.y)*2)/input_shape[0]
                self.accuracy = self.loss
            else:
                self.y        = tf.placeholder(tf.int32, shape=[network.input_shape[0]],name='y')
                self.loss     = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction,labels=self.y))/network.input_shape[0]
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y),tf.float32))
#            self.VQ        = [l.VQ for l in self.layers[1:]]
#            self.distances = tf.stack([get_distance(self.layers[:2+l]) for l in range(len(self.layers)-1)],1)
#            self.positive_radius = tf.stack([l.positive_radius for l in self.layers[1:]],1)
#            self.negative_radius = tf.stack([l.negative_radius for l in self.layers[1:]],1)
            optimizer        = optimizer(self.learning_rate)
            self.update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.apply_updates = optimizer.minimize(self.loss)
            self.session.run(tf.global_variables_initializer())
    def _fit_batch_norm(self,X):
        perm = np.random.permutation(X.shape[0])
        for i in range(n_train):
            here = perm[i*self.batch_size:(i+1)*self.batch_size]
            self.session.run(self.update_ops,feed_dict={self.x:X[here],self.training:True})
    def _fit(self,data,update_time=50):
        loss = []
        perm = np.random.permutation(data[0].shape[0])
        for i in range(self.n_train):
            here = perm[i*self.batch_size:(i+1)*self.batch_size]
            self.session.run(self.apply_updates,feed_dict={self.x:data[0][here],self.y:data[1][here],
                self.training:self.bn_training,self.learning_rate:np.float32(self.lr_schedule.lr)})
            if(i%update_time==0):
                loss.append(self.session.run(self.loss,
                    feed_dict={self.x:data[0][here],self.y:data[1][here],self.training:False}))
                print('\t\t{0:.1f}'.format(100*i/self.n_train),loss[-1])
        return loss
    def _fit_regression(self,data,update_time=50):
        loss = []
        perm = np.random.permutation(data[0].shape[0])
        for i in range(self.n_train):
            here = perm[i*self.batch_size:(i+1)*self.batch_size]
            self.session.run(self.apply_updates,feed_dict={self.x:data[0][here],self.y_:data[1][here],
                self.training:self.bn_training,self.learning_rate:np.float32(self.lr_schedule.lr)})
            if(i%update_time==0):
                loss.append(self.session.run(self.loss,
                    feed_dict={self.x:data[0][here],self.y:data[1][here],self.training:False}))
                print('\t\t{0:.1f}'.format(100*i/self.n_train),loss[-1])
        return loss
    def _test(self,data,n):
        acc = 0.0
        for j in range(n):
            acc+=self.session.run(self.accuracy,feed_dict={
                self.x:data[0][self.batch_size*j:self.batch_size*(j+1)],
                self.y:data[1][self.batch_size*j:self.batch_size*(j+1)],self.training:False})
        return acc/n
    def fit(self, train_set, valid_set, test_set, update_time = 50,
            n_epochs=5, batch_norm_epochs=0):
        # Set up the number of batches for each set
        self.n_train = train_set[0].shape[0]//self.batch_size
        self.n_test  = test_set[0].shape[0]//self.batch_size
        self.n_valid = valid_set[0].shape[0]//self.batch_size
        # Init list
        train_loss = []
        valid_accu = []
        test_accu  = []
        lr         = []
        # Perform training of batch-norm (optional, should be 0 in general)
        for i in range(batch_norm_epochs):
            self._fit_batch_norm(train_set[0])
        if(n_epochs==0): return 0,0
        for epoch in range(n_epochs):
            print(self.display_name)
            print("\tEpoch",epoch)
            if self.regression:
                train_loss.append(self._fit_regression(train_set, update_time=update_time))
            else:
                train_loss.append(self._fit(train_set, update_time=update_time))
            # NOW COMPUTE VALID SET ACCURACY
            valid_accu.append(self._test(valid_set,self.n_valid))
            # NOW COMPUTE TEST SET ACCURACY
            test_accu.append(self._test(test_set,self.n_test))
            self.lr_schedule.update(epoch=epoch,valid_accu=valid_accu)
            print('\tValid:',valid_accu[-1])
        return train_loss,valid_accu,test_accu,self.lr_schedule.lrs
    def predict(self,X):
        """
        Given an array return the prediction
        """
        n = X.shape[0]//self.batch_size
        preds = []
        for j in range(n):
            preds.append(self.session.run(self.prediction,
                feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.training:False}))
        return concatenate(preds,axis=0)
    def get_continuous_VQ(self,X):
        """
        given a collection of observations X, return the VQ for each of the layers
        in the form of a real value vector and a dictionnary mapping each real value to a VQ"""
        n            = X.shape[0]//self.batch_size
        VQs          = [[] for i in range(len(self.layers)-1)]
        dict_VQ2real = [dict() for i in range(len(self.layers)-1)]
        for j in range(n):
            vqs = self.session.run(self.VQ,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.training:False})
            for l,vq in enumerate(vqs):
                values,dict_VQ2real[l]=states2values(vq,dict_VQ2real[l])
                VQs[l].append(values)
        VQs = stack([concatenate(VQ,0) for VQ in VQs],1)
        return VQs
    def get_VQ(self,X):
        """
        given a collection of observations X, return the VQ (binary) of each layer"""
        n      = X.shape[0]//self.batch_size
        VQs    = []
        for j in range(n):
            VQs.append(self.session.run(self.VQ,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.training:False}))
        return VQs
    def get_radius(self,X):
        """
        given a collection of observations X, return the VQ (binary) of each layer"""
        n      = X.shape[0]//self.batch_size
        positive_radius = []
        negative_radius = []
        for j in range(n):
            a = self.session.run(self.distances,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.training:False})
            positive_radius.append(a)
            negative_radius.append(a)
        return concatenate(positive_radius,0),concatenate(negative_radius,0)















