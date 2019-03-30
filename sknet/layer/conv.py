import tensorflow as tf
import tensorflow.contrib.layers as tfl
from .normalize import BatchNormalization as bn
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
    def __init__(self,incoming,filters,nonlinearity = np.float32(1),
                    training=None, batch_norm = True, 
                    init_W = tfl.xavier_initializer(uniform=True),
                    strides=1,pad='valid',mode='CONSTANT', name=''):
        super().__init__(incoming)
        
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
            height = (self.input_shape[2]+p[0]-filters[1])//self.strides[0]+1
            width  = (self.input_shape[2]+p[0]-filters[1])//self.strides[0]+1
            self.output_shape = [self.input_shape[0], filters[0], height, width]
        else:
            height = (self.input_shape[1]+p[0]-filters[1])//self.strides[0]+1
            width  = (self.input_shape[2]+p[1]-filters[2])//self.strides[1]+1
            self.output_shape = [self.input_shape[0], height, width, filters[0]]
            
        # Compute shape of the W filter parameter
        w_shape = (filters[1],filters[2],self.input_shape[1],filters[0])
        # Initialize parameters
        self.W = tf.Variable(init_W(w_shape), name='conv2dlayer_W_'+name, 
                            trainable=True)
        self.b = tf.Variable(tf.zeros((1,1,1,filters[0])),
                                trainable=True,name='conv2dlayer_b_'+name)

        # Set up the nonlinearity layer
        self.nonlinearity = ScalarActivation(self.input_shape,nonlinearity)

        # Set-up batch norm layer
        if batch_norm:
            if self.data_format=='NHWC':
                self.batch_norm = bn(self.input_shape,axis=[0,1,2],gamma=False)
            else:
                self.batch_norm = bn(self.input_shape,axis=[0,2,3],gamma=False)
        else:
            self.batch_norm = Identity(self.input_shape)
    def forward(self,input=None, training=None,**kwargs):
        if input is None:
            self.incoming.forward(training=training)
        Wx = tf.nn.conv2d(self.pad(input),self.W,
                strides=[1,self.strides[0],self.strides[1],1],padding='VALID', 
                data_format=self.data_format)
        self.output = self.nonlinearity.forward(
                        self.batch_norm.forward(Wx,training=training),
                        training=training)
        return self.output
#        W_norm            = tf.sqrt(tf.reduce_sum(tf.square(self.W*self.scaling),[0,1,2]))
#        self.positive_radius = tf.reduce_min(tf.where(tf.greater(self.S,tf.zeros_like(self.S)),self.S,tf.reduce_max(self.S,keepdims=True)),[1,2,3])
#        self.negative_radius = tf.reduce_min(tf.where(tf.smaller(self.S,tf.zeros_like(self.S)),tf.abs(self.S),tf.reduce_max(tf.abs(self.S),keepdims=True)),[1,2,3])
#        print(self.output.get_shape().as_list())










