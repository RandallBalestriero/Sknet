#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

#ToDo set the seed for the batches etc



class DummyTrainer(object):
    def __init__(self, network):
        # Tensorflow Config
        tf.reset_default_graph()
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement     = True
        self.session                    = tf.Session(config=config)
        # Attributes
        self.network = network
#        with tf.device('/device:GPU:0'):
        self.session.run(tf.global_variables_initializer())
        self.input  = self.network.input
        self.output = self.network.output 


class Pipeline(object):
    def __init__(self, network, dataset=None, external_loss=0, lr_schedule=None, 
                                optimizer=None, test_loss=None):
        # Tensorflow Config
#        tf.reset_default_graph()
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement     = True
        self.session                    = tf.Session(config=config)
        # Attributes
        self.network  = network
#        self.loss     = external_loss+self.network.loss
#        self.test_loss= test_loss
        self.dataset  = dataset
#        self.lr_schedule = lr_schedule
#        self.optimizer = optimizer
#        self.batch_size = self.network[0].shape.as_list()[0]

        # set up the network input-output shortcut
#        self.input  = self.network.input
#        self.output = self.network.output
#        self.infered_observed = self.network.infered_observed

        # Get the train op
#        self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        with tf.control_dependencies(self.update_op):
#            self.train_op = self.optimizer.minimize(self.loss)

        # initialize the variables
        self.session.run(tf.global_variables_initializer())

    def _epoch_fit_passive(self,items):
        """This method allows to run in training mode the network
        without updating the network according to self.train_op,
        this is to study the passive update rules
        """
        perm = np.random.permutation(len(items[0][1]))
        n = len(items[0][1])//self.batch_size
        self.network.set_deterministic(False,self.session)
        for i in range(n):
            here      = perm[i*self.batch_size:(i+1)*self.batch_size]
            if self.dataset is None:
                feed_dict = [(key,self.dataset["train_set"][value][here]) for key,value in items]
            else:
                feed_dict = [(key,value[here]) for key,value in items]
            self.session.run(self.update_op,feed_dict=dict(feed_dict))
    def _epoch_train(self,items=None,update_time=50,set_="train_set"):
        loss = []
        if items is not None:
            n = len(items[0][1])
        else:
            if self.items[0][1] is not None:
                n = len(self.dataset[self.items[0][1]][set_])
        perm = np.random.permutation(n)
        self.network.set_deterministic(False,self.session)
        for i in range(n//self.batch_size):
            here = perm[i*self.batch_size:(i+1)*self.batch_size]
            if items is None:
                feed_dict = [(key,self.dataset[value][set_][here]) for key,value in self.items]
            else:
                feed_dict = [(key,value[here]) for key,value in items]
            lr_dict   = [self.optimizer.learning_rate,np.float32(self.lr_schedule.lrs[-1])]
            self.session.run(self.train_op,feed_dict=dict(feed_dict+[lr_dict]))
            if(i%update_time==0):
                self.network.set_deterministic(True,self.session)
                loss.append(self.session.run(self.loss,feed_dict=dict(feed_dict)))
                self.network.set_deterministic(False,self.session)
                print('\t\t{0:.1f}%'.format(100*i/(n//self.batch_size)),loss[-1])
        return loss
    def epoch(self,op,linkage=None, context="test_set", deterministic=None, batch_size=None):
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

        Parameters
        ----------

        op : tf.Tensor or list of tf.Tensor or list of descriptor
            the op (or ops as a list of ops) to run for one epoch or 
            a list of the form [(op1,periodicity, optional np op)] where op1 can again
            be a list. For example, during training, one might prefer
            to compute some statistics only every X batch for example ::


        """
        # find out the batch_size, if not given, take it from the network, if given
        # (case when the network was build with None as batch_size, then assert that
        # it is true (or at least that they both match)
        if batch_size is None:
            assert(self.network.batch_size is not None)
            batch_size = self.network.batch_size
        elif self.network.batch_size is not None:
            assert(batch_size==self.network.batch_size)
        # if a deterministic behavior is given, set it
        if deterministic is not None:
            self.network.set_deterministic(deterministic,self.session)
        # compute the number of batches
        if linkage is not None:
            N = max([len(value) for value in linkage.values])
        else:
            N = self.network.dataset.N(context)
        N_BATCH = N//batch_size
        if hasattr(op,'__len__'):
            if hasattr(op[0],'__len__'):
                assert(type(op)==list)
                assert(type(op[0])==list)
                assert(not hasattr(op[0][0],'__len__'))
                output = [[] for _ in range(len(op))]
                multiple=True
                # ensure that an op is given or set it to identity
                for i in range(len(op)):
                    if len(op[i])==2:
                        op[i].append(lambda x:x)
        else:
            output = list()
            multiple=False
        for i in range(N_BATCH):
            here = range(i*batch_size,(i+1)*batch_size)
            if not multiple:
                if linkage is None:
                    output.append(self.execute(op,indices=here,context=context))
                else:
                    output.append(self.execute(op,linkage=linkage))
            else:
                for j in range(len(op)):
                    if (i%op[j][1])==0:
                        if linkage is None:
                            output[j].append(self.execute(op[j][0],indices=here,context=context))
                        else:
                            output[j].append(self.execute(op[j][0],linkage=linkage))
        if multiple:
            output = [op[i][-1](np.concatenate([batch_out for 
                            batch_out in output[i]],0)) if hasattr(output[i][0],'__len__') else op[i][-1](np.asarray([batch_out for
                            batch_out in output[i]])) for i in range(len(op))]
        elif hasattr(op,'__len__'):
            output = np.concatenate([batch_out[i] for batch_out in output],0)
        else:
            output = np.concatenate(output,0)
        return output
    def fit(self, train_ops, valid_ops, test_ops, n_epochs=5, passive_epochs=0):
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

        train_ops : operators

        """
        # Init list
        train_loss = []
        valid_loss = []
        test_loss  = []
        # Perform training of batch-norm (optional, should be 0 in general)
        for i in range(passive_epochs):
            self._epoch_fit_passive(items=items)
        for epoch in range(n_epochs):
            print("\tEpoch",epoch)
            # TRAIN SET
            train_loss.append(self.epoch(train_ops,context='train_set',deterministic=False))
            # VALID SET
            valid_loss.append(self.epoch(valid_ops,context='valid_set',deterministic=True))
            # TEST SET
            test_loss.append(self.epoch(test_ops,context='test_set',deterministic=True))
#            self.lr_schedule.update(epoch=epoch,valid_accu=valid_loss)
            print('\tValid:',valid_loss[-1])
            print('\tValid:',test_loss[-1])
        return train_loss,valid_loss,test_loss

    def execute(self,op,linkage=None,indices=None,context=None,additional_linkage=None):
        """Execute a given op given a linkage of inputs:variables
        or given indices and context additional linkage can be for 
        example the learning rate etc as in
        additional_linkage = {self.optimizer.learning_rate:np.float32(self.lr_schedule.lrs[-1])}
        
        Parameters
        ----------

        op : tf.Tensor or tf.Operation or list of tf.Tensor or tf.Operation
            the operation(s) and/or tensor(s) to execute

        context : str 
            the name of the set onto which extract the data via indices, it 
            should be a value present in dataset.sets

        indices : list or array of int
            the indices to extract the batch from the set given by context 
            the obtained data will thus be from the following 
            ``self.dataset[data][context][indices]`` where data is the name
            of the variable needed for each dependency of op

        """
        assert(type(additional_linkage)==dict or additional_linkage is None)
        if type(linkage)==dict:
            if additional_linkage is not None:
                return self.session.run(op,feed_dict=
                                        {**linkage,**additional_linkage})
            else:
                return self.session.run(op,feed_dict=linkage)
        # check if op is a list of op, in this case compute the non redundant
        # dependencies
        if hasattr(op,'__len__'):
            assert(not hasattr(p[0],'__len__'))
            dependencies = list(set(self.network.dependencies[op_] 
                                                            for op_ in op))
        else:
            dependencies = self.network.dependencies[op]

        feed_dict = [(var,self.network.get_input_for(var,indices,context)) 
                                for var in dependencies]
        if additional_linkage is not None:
            feed_dict = feed_dict+additional_linkage.items()
        return self.session.run(op,feed_dict=dict(feed_dict))




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















