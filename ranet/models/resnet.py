#execfile('lasagne_tf.py')
#execfile('utils.py')
from layers import *
from utils import *
from utils_plot import *


import random

def onehot(n,k):
        z=zeros(n,dtype='float32')
        z[k]=1
        return z


class DNNClassifier(object):
    def __init__(self,input_shape,model_class,lr=0.0001,optimizer = tf.train.AdamOptimizer,regression=False,update_batch_norm=True):
        #setting = {base,pretrainlinear,}
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.n_classes=  model_class.n_classes
        config.log_device_placement=True
        self.session    = tf.Session(config=config)
        self.batch_size = input_shape[0]
        self.lr         = lr
        self.update_batch_norm = update_batch_norm
        self.regression = regression
        with tf.device('/device:GPU:0'):
            self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
            optimizer          = optimizer(self.learning_rate)
            self.x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
            self.training      = tf.placeholder(tf.bool,name='phase')
            self.prediction,self.layers = model_class.get_layers(self.x,input_shape,training=self.training)
            self.number_of_params = count_number_of_params()
            print(self.number_of_params)
            if(regression):
                self.y_       = tf.placeholder(tf.float32, shape=[input_shape[0]],name='y')
                self.loss     = tf.reduce_sum(tf.nn.l2_loss(self.prediction-self.y_)*2)/input_shape[0]
                self.accuracy = self.loss
            else:
                self.y_       = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
                self.loss     = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction,labels=self.y_))/input_shape[0]
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
            self.VQ     = [l.VQ for l in self.layers[1:]]
            self.distances = tf.stack([get_distance(self.layers[:2+l]) for l in range(len(self.layers)-1)],1)
#            self.positive_radius = tf.stack([l.positive_radius for l in self.layers[1:]],1)
#            self.negative_radius = tf.stack([l.negative_radius for l in self.layers[1:]],1)
            self.update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.apply_updates = optimizer.minimize(self.loss)
            self.session.run(tf.global_variables_initializer())
            print(count_number_of_params())
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


 



#class largeCNN:
#        def get_layers(self,input_variable,input_shape,test):
#                layers = [InputLayer(input_shape,input_variable)]
#                layers.append(Conv2DLayer(layers[-1],64,5,pad='same',test=test,bn=self.bn,first=True,centered=self.centered,ortho=self.ortho))
#		layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
#                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
#                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
#                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
#                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
#                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
##                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
#                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
#                layers.append(Conv2DLayer(layers[-1],192,3,pad='full',test=test,bn=self.bn,ortho=self.ortho))
#                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
#                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
#                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
#                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
#                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
#                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
#                layers.append(Conv2DLayer(layers[-1],192,1,test=test,centered=self.centered,ortho=self.ortho))
#                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
#                layers.append(GlobalPoolLayer(layers[-1]))
#                layers.append(DenseLayer(layers[-1],self.n_classes,training=test,bn=0,ortho=0))
#		self.layers = layers
#                return self.layers[-1].output,self.layers
#


class Model:
    def __init__(self,n_classes=10, nonlinearity=tf.nn.leaky_relu, batch_norm=True, K=4, L=3,trainable=True):
        self.n_classes    = n_classes
        self.trainable    = trainable
        self.nonlinearity = nonlinearity
        self.batch_norm   = batch_norm
	# Those variables are only used for toy examples
        self.K            = K
        self.L            = L



class MiniDense(Model):
    def get_layers(self,input_variable,input_shape,training):
        layers = [InputLayer(input_shape,input_variable)]
        for l in range(self.L):
            layers.append(DenseLayer(layers[-1],self.K,training=training, 
                batch_norm=self.batch_norm,nonlinearity=self.nonlinearity,trainable=self.trainable))
        layers.append(DenseLayer(layers[-1],self.n_classes,training=training,batch_norm=False,nonlinearity=tf.identity))
        return layers[-1].output,layers




class DenseCNN(Model):
    def get_layers(self,input_variable,input_shape,training):
        layers = [InputLayer(input_shape,input_variable)] #(? ? 1-3)
        layers.append(FirstConv2DLayer(layers[-1],filters_T=9,sampling_n=5,filter_shape=5,spline=False)) #(? ? 32)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) # (? ? 32)
        layers.append(Pool2DLayer(layers[-1],2)) #(? ? 32)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=17,channels_T=9,filter_shape=3,spline=False)) # -> n=5 -> (? ? 64)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(Pool2DLayer(layers[-1],2))#(? ? 64)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=33,channels_T=17,filter_shape=3,spline=False)) # -> n=5 -> (? ? 128)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(GlobalPoolLayer(layers[-1]))
        layers.append(DenseLayer(layers[-1],self.n_classes))
        return layers[-1].output,layers

class SplineDenseCNN(Model):
    def get_layers(self,input_variable,input_shape,training):
        layers = [InputLayer(input_shape,input_variable)] #(? ? 1-3)
        layers.append(FirstConv2DLayer(layers[-1],filters_T=9,sampling_n=5,filter_shape=5,spline=True)) #(? ? 32)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) # (? ? 32)
        layers.append(Pool2DLayer(layers[-1],2)) #(? ? 32)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=17,channels_T=9,filter_shape=3,spline=True)) # -> n=5 -> (? ? 64)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(Pool2DLayer(layers[-1],2))#(? ? 64)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=33,channels_T=17,filter_shape=3,spline=True)) # -> n=5 -> (? ? 128)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(GlobalPoolLayer(layers[-1]))
        layers.append(DenseLayer(layers[-1],self.n_classes))
        return layers[-1].output,layers


class DenseCNN2(Model):
    def get_layers(self,input_variable,input_shape,training):
        layers = [InputLayer(input_shape,input_variable)] #(? ? 1-3)
        layers.append(FirstConv2DLayer(layers[-1],filters_T=9,sampling_n=10,filter_shape=5,spline=False)) #(? ? 32)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) # (? ? 32)
        layers.append(Pool2DLayer(layers[-1],2)) #(? ? 32)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=17,channels_T=9,filter_shape=3,spline=False)) # -> n=5 -> (? ? 64)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(Pool2DLayer(layers[-1],2))#(? ? 64)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=33,channels_T=17,filter_shape=3,spline=False)) # -> n=5 -> (? ? 128)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(GlobalPoolLayer(layers[-1]))
        layers.append(DenseLayer(layers[-1],self.n_classes))
        return layers[-1].output,layers

class SplineDenseCNN2(Model):
    def get_layers(self,input_variable,input_shape,training):
        layers = [InputLayer(input_shape,input_variable)] #(? ? 1-3)
        layers.append(FirstConv2DLayer(layers[-1],filters_T=9,sampling_n=10,filter_shape=5,spline=True)) #(? ? 32)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) # (? ? 32)
        layers.append(Pool2DLayer(layers[-1],2)) #(? ? 32)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=17,channels_T=9,filter_shape=3,spline=True)) # -> n=5 -> (? ? 64)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(Pool2DLayer(layers[-1],2))#(? ? 64)
        #
        layers.append(Conv2DLayer(layers[-1],filters_T=33,channels_T=17,filter_shape=3,spline=True)) # -> n=5 -> (? ? 128)
        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,training=training)) #(? ? 64)
        layers.append(GlobalPoolLayer(layers[-1]))
        layers.append(DenseLayer(layers[-1],self.n_classes))
        return layers[-1].output,layers









############################



class largeCNN:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],64,5,pad='SAME',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(ConvLayer(layers[-1],96,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],96,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,1,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
		self.layers = layers
                return self.layers



class smallCNN:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],64,5,pad='SAME',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],128,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,1,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
		self.layers = layers
                return self.layers













class smallDENSE:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],1024,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],1024,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class largeDENSE:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],4096,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],4096,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

























class allCNN1:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],5,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (5 28 28) : 3920
                layers.append(ConvLayer(layers[-1],7,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (7 24 24) : 4032
                layers.append(ConvLayer(layers[-1],9,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (9 22 22) : 4356
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allCNN2:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],7,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (7 28 28)  : 5488
                layers.append(ConvLayer(layers[-1],11,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (11 24 24) : 6336
                layers.append(ConvLayer(layers[-1],15,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (15 22 22) : 7260
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allCNN3:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],12,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (12 28 28) : 9408
                layers.append(ConvLayer(layers[-1],16,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (16 24 24) : 9216
                layers.append(ConvLayer(layers[-1],20,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (20 22 22) : 9680
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allCNN4:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],20,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (20 28 28) : 15680
                layers.append(ConvLayer(layers[-1],32,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (32 24 24) : 18432
                layers.append(ConvLayer(layers[-1],48,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (48 20 20) : 19200
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 2048
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 256
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers





class allDENSE1:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],3920,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],4032,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],4356,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allDENSE2:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],5488,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],6336,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],7260,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allDENSE3:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],9408,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],9216,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],9680,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allDENSE4:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],15680,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],18432,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],19200,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers






class SpecialDense:
        def __init__(self,n_classes=10,constraint='dt',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.constraint  = constraint
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(SpecialDenseLayer(layers[-1],16,constraint=self.constraint,training=training,first=True))
#                layers.append(SpecialDenseLayer(layers[-1],64,constraint=self.constraint,training=training,first=False))
#                layers.append(DenseLayer(layers[-1],6,nonlinearity='relu',training=training))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=None,training=training))
		self.layers = layers
                return self.layers












