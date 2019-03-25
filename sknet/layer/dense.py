import tensorflow as tf
import tensorflow.contrib.layers as tfl
from .normalize import batchnormalization as bn
import numpy as np
from . import Layer


class Dense(Layer):
    def __init__(self, incoming, units, nonlinearity_c = np.float32(0),
                training=None, batch_norm = False,
                init_W = tfl.xavier_initializer(uniform=True), name=''):
        """
        2D pooling layer
        Performs 2D mean- or max-pooling over the two trailing axes of a 4D input
        tensor. This is an alternative implementation which uses
        """
        super().__init__(incoming)
        # Set up the input, flatten if needed
        if len(self.in_shape)>2:
            self.flatten_input = True
            flat_dim           = np.prod(self.in_shape[1:])
        else:
            self.flatten_input = False
            flat_dim           = self.in_shape[1]
        self.nonlinearity_c = np.float32(nonlinearity_c)
        
        # Set-up Output Shape
        self.out_shape = (self.in_shape[0],units)
        self.batch_norm = batch_norm
        # Initialize Layer Parameters
        self.W = tf.Variable(init_W((flat_dim,units)),
                            name='denselayer_W_'+name,
                            trainable=True)
        # Init batch-norm or bias
        if batch_norm:
            self.bn  = bn((self.in_shape[0],flat_dim),axis=[1])
        else:
            self.b   = tf.Variable(tf.zeros((1,units)),
                            trainable=True, 
                            name='denselayer_b_'+name)
        # If input is a layer,
        if self.given_input:
            self.forward(incoming.output,training=training)
    def forward(self,input,training=None,**kwargs):
        if self.flatten_input:
            input = tf.layers.flatten(input)
        if self.batch_norm:
            Wx       = tf.matmul(input,self.W)
            self.scaling,self.bias = self.bn.get_A_b(Wx,training=training)
        else:            
            self.scaling = 1
            self.bias    = self.b
        self.S  = self.scaling*Wx+self.bias
        if self.nonlinearity_c==1:
            self.VQ     = None
            self.output = self.S 
        else:
            self.VQ     = tf.greater(self.S,0)
            VQ          = tf.cast(self.VQ,tf.float32)
            self.mask   = VQ+(1-VQ)*self.nonlinearity_c
            self.output = tf.nn.leaky_relu(self.S,alpha=self.nonlinearity_c)
        return self.output
#    def backward(self,output):
#        """output is of shape (batch,n_output)
#        return of this function is of shape [(batch,in_dim),(batch,1)]"""
#        # use tf.nn.conv2d_backprop_input for conv2D
#        A = tf.reshape(tf.matmul(output*self.mask*scaling,self.W,transpose_b=True),incoming.out_shape)
#        B = tf.matmul(output*self.mask,self.b,transpose_b=True)
#        return A,B





class ConstraintDenseLayer:
    def __init__(self,incoming,n_output,constraint='none',training=None):
        # bias_option : {unconstrained,constrained,zero}
        if(len(incoming.out_shape)>2): reshape_input = tf.layers.flatten(incoming.output)
        else:                             reshape_input = incoming.output
        in_dim      = prod(incoming.out_shape[1:])
        self.gamma  = tf.Variable(ones(1,float32),trainable=False)
        gamma_update= tf.assign(self.gamma,tf.clip_by_value(tf.cond(training,lambda :self.gamma*1.005,lambda :self.gamma),0,60000))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,gamma_update)
        init_W      = tf.contrib.layers.xavier_initializer(uniform=True)
        if(constraint=='none'):
                self.W_     = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
                self.W      = self.W_
        elif(constraint=='dt'):
                self.W_     = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
                self.alpha  = tf.Variable(randn(1,n_output).astype('float32'),trainable=True)
                self.W      = self.alpha*tf.nn.softmax(tf.clip_by_value(self.gamma*self.W_,-20000,20000),axis=0)
        elif(constraint=='diag'):
                self.sign   = tf.Variable(randn(in_dim,n_output).astype('float32'),trainable=True)
                self.alpha  = tf.Variable((randn(1,n_output)/sqrt(n_output)).astype('float32'),trainable=True)
                self.W      = tf.nn.tanh(self.gamma*self.sign)*self.alpha
        self.out_shape = (incoming.out_shape[0],n_output)
        self.output       = tf.matmul(reshape_input,self.W)
        self.VQ = None










