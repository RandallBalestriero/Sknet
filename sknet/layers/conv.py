import tensorflow as tf
import tensorflow.contrib.layers as tfl
from .normalize import BatchNorm as bn
import numpy as np
from . import Layer


from .. import Variable

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
    def __init__(self,incoming,filters,W = tfl.xavier_initializer(),
                    b = tf.zeros, strides=1, pad='valid',
                    mode='CONSTANT', name='', W_func = tf.identity,
                    b_func = tf.identity):
        with tf.variable_scope("Conv2D") as scope:
            self.scope_name = scope.original_name_scope
            self.mode = mode
            if np.isscalar(strides):
                self.strides = [strides,strides]
            else:
                self.strides = strides
                
            # Define the padding function
            if pad=='valid' or (filters[1]==1 and filters[2]==1):
                self.to_pad=False
            else:
                if pad=='same':
                    assert(filters[1]%2==1 and filters[2]%2==1)
                    self.p = [(filters[1]-1)//2,(filters[2]-1)//2]
                else:
                    self.p = [filters[1]-1,filters[2]-1]
                self.to_pad = True
                                            
            # Compute shape of the W filter parameter
            w_shape = (filters[1],filters[2],
                                incoming.shape.as_list()[1],filters[0])
            # Initialize W
            if type(W)!=Variable:
                W = Variable(W, name='conv2dlayer_W_'+name)
            self._W = W(w_shape)
            self.W  = W_func(self._W)
            # Initialize b
            if b is None:
                self._b = None
                self.b  = None
            else:
                if type(b)!=Variable:
                    b = Variable(b, name='conv2dlayer_b_'+name)
                self._b = b((1,filters[0],1,1))
                self.b  = b_func(self._b)
    
            super().__init__(incoming)

    def forward(self,input, *args,**kwargs):
        if self.to_pad:
            padded = tf.pad(input,[[0,0],[0,0],[self.p[0]]*2,
                                [self.p[1]]*2],mode=self.mode)
        else:
            padded = input
        Wx = tf.nn.conv2d(padded,self.W,
                strides=[1,self.strides[0],self.strides[1],1],padding='VALID',
                data_format="NCHW")
        if self.b is None:
            return Wx
        else:
            return Wx+self.b




