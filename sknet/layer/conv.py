import tensorflow as tf
import tensorflow.contrib.layers as tfl
from .normalize import batchnormalization as bn
import numpy as np
from . import Layer

class Conv2D(Layer):
    """2D (spatial) convolutional layer.
    Layer to perform a 2D convolution onto a 4D input tensor

    :param incoming: input shape of incoming layer
    :type incoming: Layer or tuple of int
    :param filters: the shape of the filters in the form 
                    (#filters, height, width)
    :type filters: triple of int
    :param nonlinearity_c: coefficient of the nonlinearity,
                           0 for ReLU,-1 for absolute value,...
    :type nonlinearity_c: scalar

    """
    def __init__(self,incoming,filters,nonlinearity_c = np.float32(0),
                    training=None, batch_norm = True, 
                    init_W = tfl.xavier_initializer(uniform=True),
                    strides=1,pad='valid',mode='CONSTANT', name=''):
        super().__init__(incoming)
        
        # Set attributes
        self.batch_norm     = batch_norm
        self.nonlinearity_c = np.float32(nonlinearity_c)
        if np.isscalar(strides):
            self.strides = [strides,strides]
        else:
            self.strides = strides
            
        # Define the padding function
        if pad=='valid' or filter_shape==1:
            self.pad = lambda x:x
            p = [0,0]
        else:
            if pad=='same':
                assert(filters[1]%2==1 and filters[2]%2==1)
                p = [(filters[1]-1)//2,(filters[2]-1)//2]
            else:
                p = [filters[1]-1,filters[2]-1]
            if self.data_format=='NCHW':
                self.pad = lambda x: tf.pad(x,[[0,0],[0,0],
                                    [p[0],p[0]],[p[1],p[1]]],mode=mode)
            else:
                self.pad = lambda x: tf.pad(x,[[0,0],[p[0],p[0]],
                                        [p[1],p[1]],[0,0]],mode=mode)
                                        
        # Set up output shape
        if self.data_format=='NCHW':
            height = (self.in_shape[2]+p[0]-filters[1])//self.strides[0]+1
            width  = (self.in_shape[2]+p[0]-filters[1])//self.strides[0]+1
            self.out_shape = [self.in_shape[0], filters[0], height, width]
        else:
            height = (self.in_shape[1]+p[0]-filters[1])//self.strides[0]+1
            width  = (self.in_shape[2]+p[1]-filters[2])//self.strides[1]+1
            self.out_shape = [self.in_shape[0], height, width, filters[0]]
            
        # Compute shape of the W filter parameter
        if self.data_format=='NCHW':
            w_shape = (filters[1],filters[2],self.in_shape[1],filters[0])
        else:
            w_shape = (filters[1],filters[2],self.in_shape[3],filters[0])
            
        # Initialize parameters
        self.W = tf.Variable(init_W(w_shape), name='conv2dlayer_W_'+name, 
                            trainable=True)
        if batch_norm:
            if self.data_format=='NHWC':
                self.bn = bn(self.out_shape,axis=[0,1,2])
            else:
                self.bn = bn(self.out_shape,axis=[0,2,3])
        else:            
            self.b = tf.Variable(tf.zeros((1,1,1,n_filters)),
                                trainable=True,name='conv2dlayer_b_'+name)
         
        if self.given_input:
            self.forward(incoming.output,training=training)
    def forward(self,input,training=None,**kwargs):
        Wx = tf.nn.conv2d(self.pad(input),self.W,
                strides=[1,self.strides[0],self.strides[1],1],padding='VALID', 
                data_format=self.data_format)
        if(self.batch_norm):
                self.scaling,self.bias = self.bn.get_A_b(Wx,training=training)
        else:
                self.scaling,self.bias = 1,self.b
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
#        W_norm            = tf.sqrt(tf.reduce_sum(tf.square(self.W*self.scaling),[0,1,2]))
#        self.positive_radius = tf.reduce_min(tf.where(tf.greater(self.S,tf.zeros_like(self.S)),self.S,tf.reduce_max(self.S,keepdims=True)),[1,2,3])
#        self.negative_radius = tf.reduce_min(tf.where(tf.smaller(self.S,tf.zeros_like(self.S)),tf.abs(self.S),tf.reduce_max(tf.abs(self.S),keepdims=True)),[1,2,3])
#        print(self.output.get_shape().as_list())










