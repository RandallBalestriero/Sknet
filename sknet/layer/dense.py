import tensorflow as tf
import tensorflow.contrib.layers as tfl
from .normalize import BatchNormalization as bn
from .special import Activation as sa
from .special import Identity
import numpy as np

from . import Layer

from ..utils import init_variable as init_var




class Dense(Layer):
    """Dense or fully connected layer.
    This layer implement a fully connected or a dense layer with or without
    nonlinearity, and batch-norm

    :param incoming: input shape or incoming :class:`Layer` instance
    :type incoming: tuple of int or Layer
    :param units: then umber of output units (or neurons)
    :type units: int
    :param nonlinearity_c: the coefficient for the nonlinearity, 
                           0 for ReLU, -1 for absolute value, ...
    :type nonlinearity_c: scalar
    :param deterministic: a dummy Tensorflow boolean stating if it is 
                     deterministic time or testing time
    :type deterministic: tf.bool
    :param batch_norm: using or not the batch-normalization
    :type batch_norm: bool
    :param init_W: initialization for the W weights
    :type init_W: initializer of tf.tensor or np.array
    :param init_b: initialization for the b weights
    :type init_b: initializer of tf.tensor or np.array
    :param name: name for the layer
    :type name: str

    """
    def __init__(self, incoming, units, nonlinearity = np.float32(1),
                deterministic=None,
                init_W = tfl.xavier_initializer(uniform=True), 
                init_b = tf.zeros,name='',**kwargs):
        # Set up the input, flatten if needed
        if len(incoming.shape.as_list())>2:
            self._flatten_input = True
            flat_dim  = np.prod(incoming.shape.as_list()[1:])
        else:
            self._flatten_input = False
            flat_dim  = incoming.shape.as_list()[1]

        # Initialize the layer variables
        self._W = init_var(init_W,(flat_dim,units),
                            name='dense_W_'+name,
                            trainable=True)
        self._b = init_var(init_b,(1,units),
                            trainable=True,
                            name='dense_b_'+name)

        self.W  = self._W
        self.b  = self._b

        super().__init__(incoming, deterministic=deterministic,**kwargs)

    def forward(self, input, deterministic=None, **kwargs):
        if self._flatten_input:
            input = tf.layers.flatten(input)
        return tf.matmul(input,self._W)+self._b






class Dense2:
    """Dense or fully connected layer.
    This layer implement a fully connected or a dense layer with or without
    nonlinearity, and batch-norm

    :param incoming: input shape or incoming :class:`Layer` instance
    :type incoming: tuple of int or Layer
    :param units: then umber of output units (or neurons)
    :type units: int
    :param nonlinearity_c: the coefficient for the nonlinearity, 
                           0 for ReLU, -1 for absolute value, ...
    :type nonlinearity_c: scalar
    :param deterministic: a dummy Tensorflow boolean stating if it is 
                     deterministic time or testing time
    :type deterministic: tf.bool
    :param batch_norm: using or not the batch-normalization
    :type batch_norm: bool
    :param init_W: initialization for the W weights
    :type init_W: initializer of tf.tensor or np.array
    :param init_b: initialization for the b weights
    :type init_b: initializer of tf.tensor or np.array
    :param name: name for the layer
    :type name: str

    """
    variables=["W","b","output"]
    def __init__(self, incoming, units, nonlinearity = np.float32(1),
                deterministic=None, batch_norm = False,
                init_W = tfl.xavier_initializer(uniform=True), 
                init_b = tf.zeros, observed=[],observation=[],
                teacher_forcing=[],name=''):
        if len(observed)>0:
            for obs in observed:
                assert(obs in Dense.variables)
        # Set up the input, flatten if needed
        if len(incoming.shape.as_list())>2:
            self._flatten_input = True
            flat_dim  = np.prod(incoming.shape.as_list()[1:])
        else:
            self._flatten_input = False
            flat_dim  = incoming.shape.as_list()[1]

        # Initialize the layer variables
        self._W = init_var(init_W,(flat_dim,units),
                            name='dense_W_'+name,
                            trainable=True)
        self._b = init_var(init_b,(1,units),
                            trainable=True,
                            name='dense_b_'+name)
        if "W" in observed:
            self.W = Tensor(self._W,observed=True,)

        super().__init__(incoming, deterministic=deterministic, 
                        observed=observed, observation=observation, 
                        teacher_forcing=teacher_forcing)

    def forward(self, input, deterministic=None, **kwargs):
        if self._flatten_input:
            input = tf.layers.flatten(input)
        return tf.matmul(input,self._W)+self._b




class ConstraintDenseLayer:
    def __init__(self,incoming,n_output,constraint='none',deterministic=None):
        # bias_option : {unconstrained,constrained,zero}
        if(len(incoming.output_shape)>2): reshape_input = tf.layers.flatten(incoming.output)
        else:                             reshape_input = incoming.output
        in_dim      = prod(incoming.output_shape[1:])
        self.gamma  = tf.Variable(ones(1,float32),trainable=False)
        gamma_update= tf.assign(self.gamma,tf.clip_by_value(tf.cond(deterministic,lambda :self.gamma*1.005,lambda :self.gamma),0,60000))
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
        self.output_shape = (incoming.output_shape[0],n_output)
        self.output       = tf.matmul(reshape_input,self.W)
        self.VQ = None










