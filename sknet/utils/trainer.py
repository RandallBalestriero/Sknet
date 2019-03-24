#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

#ToDo set the seed for the batches etc



class DummyTrainer(object):
    def __init__(self, model):
        # Tensorflow Config
        tf.reset_default_graph()
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement     = True
        self.session                    = tf.Session(config=config)
        # Attributes
        with tf.device('/device:GPU:0'):
            self.x             = tf.placeholder(tf.float32, shape=model.input_shape,name='x')
            self.training      = tf.placeholder(tf.bool,name='phase')
            self.layers        = model.get_layers(self.x,training=self.training)
            self.session.run(tf.global_variables_initializer())




















class Trainer(object):
    def __init__(self, model, lr_schedule, optimizer = tf.train.AdamOptimizer, regression=False, 
            bn_training=True, display_name = ''):
        # Tensorflow Config
        tf.reset_default_graph()
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement     = True
        self.session                    = tf.Session(config=config)
        # Attributes
        self.display_name = display_name
        self.classes      = model.classes
        self.batch_size   = model.input_shape[0]
        self.regression   = regression
        self.bn_training  = bn_training
        self.lr_schedule  = lr_schedule
        with tf.device('/device:GPU:0'):
            self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
            self.x             = tf.placeholder(tf.float32, shape=model.input_shape,name='x')
            self.training      = tf.placeholder(tf.bool,name='phase')
            self.layers        = model.get_layers(self.x,training=self.training)
            self.prediction    = self.layers[-1].output
            self.number_of_params = 14#count_number_of_params()
            if(regression):
                self.y        = tf.placeholder(tf.float32, shape=[model.input_shape[0]],name='y')
                self.loss     = tf.reduce_sum(tf.nn.l2_loss(self.prediction-self.y)*2)/input_shape[0]
                self.accuracy = self.loss
            else:
                self.y        = tf.placeholder(tf.int32, shape=[model.input_shape[0]],name='y')
                self.loss     = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction,labels=self.y))/model.input_shape[0]
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


 












