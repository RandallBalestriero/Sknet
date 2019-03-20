from utils_plot import *


import random



class Trainer(object):
    def __init__(self, model, lr=0.0001,
                optimizer = tf.train.AdamOptimizer, regression=False):
        # Tensorflow Config
        tf.reset_default_graph()
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement     = True
        self.session                    = tf.Session(config=config)
        # Attributes
        self.n_classes  = model.n_classes
        self.batch_size = model.input_shape[0]
        self.lr         = lr
#        self.update_batch_norm = update_batch_norm
        self.regression = regression
        with tf.device('/device:GPU:0'):
            self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
            optimizer          = optimizer(self.learning_rate)
            self.x             = tf.placeholder(tf.float32, shape=model.input_shape,name='x')
            self.training      = tf.placeholder(tf.bool,name='phase')
            self.prediction,self.layers = model_class.get_layers(self.x,training=self.training)
            self.number_of_params = count_number_of_params()
            if(regression):
                self.y_       = tf.placeholder(tf.float32, shape=[input_shape[0]],name='y')
                self.loss     = tf.reduce_sum(tf.nn.l2_loss(self.prediction-self.y_)*2)/input_shape[0]
                self.accuracy = self.loss
            else:
                self.y_       = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
                self.loss     = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction,labels=self.y_))/input_shape[0]
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
#            self.VQ        = [l.VQ for l in self.layers[1:]]
#            self.distances = tf.stack([get_distance(self.layers[:2+l]) for l in range(len(self.layers)-1)],1)
#            self.positive_radius = tf.stack([l.positive_radius for l in self.layers[1:]],1)
#            self.negative_radius = tf.stack([l.negative_radius for l in self.layers[1:]],1)
            self.update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.apply_updates = optimizer.minimize(self.loss)
            self.session.run(tf.global_variables_initializer())
    def _fit_batch_norm(self,X):
        perm = permutation(X.shape[0])
        n_train    = X.shape[0]//self.batch_size
        for i in range(n_train):
            here = perm[i*self.batch_size:(i+1)*self.batch_size]
            self.session.run(self.update_ops,feed_dict={self.x:X[here],self.training:True})
    def _fit(self,X,y,indices,update_time=50):
        self.e+=1
        n_train    = X.shape[0]//self.batch_size
        train_loss = []
        for i in range(n_train):
            if(self.batch_size<self.n_classes):
                here = [random.sample(k,1) for k in indices]
                here = [here[i] for i in permutation(self.n_classes)[:self.batch_size]]
            else:
                here = [random.sample(k,self.batch_size//self.n_classes) for k in indices]
                here = concatenate(here)
            self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.y_:y[here],self.training:self.update_batch_norm,self.learning_rate:float32(self.lr)})
            if(i%update_time==0):
                train_loss.append(self.session.run(self.loss,feed_dict={self.x:X[here],self.y_:y[here],self.training:False}))
            if(i%100 ==0):
                print(i,n_train,train_loss[-1])
        return train_loss
    def _fit_regression(self,X,y,update_time=50):
        self.e    += 1
        n_train    = X.shape[0]//self.batch_size
        train_loss = []
        perm       = permutation(X.shape[0])
        for i in range(n_train):
            here = perm[i*self.batch_size:(i+1)*self.batch_size]
            self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.y_:y[here],self.training:self.update_batch_norm,self.learning_rate:float32(self.lr)})
            if(i%update_time==0):
                train_loss.append(self.session.run(self.loss,feed_dict={self.x:X[here],self.y_:y[here],self.training:False}))
            if(i%100 ==0):
                print(i,n_train,train_loss[-1])
        return train_loss
    def fit(self,X,y,X_test,y_test,n_epochs=5,batch_norm_epochs=0):
        train_loss = []
        test_accu  = []
        self.e     = 0
        n_test     = X_test.shape[0]//self.batch_size
        if(self.regression==False): indices    = [list(find(y==k)) for k in range(self.n_classes)]
        for i in range(batch_norm_epochs):
            self._fit_batch_norm(X)
        if(n_epochs==0): return 0,0
        for i in range(n_epochs):
            if i==(3*n_epochs//4): self.lr/=3
            elif i==(5*n_epochs//6): self.lr/=3
            print("epoch",i)
            if(self.regression==False): train_loss.append(self._fit(X,y,indices))
            else:                  train_loss.append(self._fit_regression(X,y))
            # NOW COMPUTE TEST SET ACCURACY
            acc1 = 0.0
            for j in range(n_test):
                acc1+=self.session.run(self.accuracy,feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],
    				self.y_:y_test[self.batch_size*j:self.batch_size*(j+1)],self.training:False})
            test_accu.append(acc1/n_test)
            print('Test:',test_accu[-1])
        return concatenate(train_loss),test_accu
    def predict(self,X):
        n = X.shape[0]//self.batch_size
        preds = []
        for j in range(n):
            preds.append(self.session.run(self.prediction,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.training:False}))
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


 












