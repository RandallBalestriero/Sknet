import tensorflow as tf
import numpy as np
from . import Layer


class Spectrogram(Layer):
    """spectrogram layer (time-frequency representation of a time serie)
    This layer implement a spectrogram computation to allow to include 
    it as part of a model when dealing with time serie data, it is also 
    crucial to allow backpropagation through it in case some previous 
    operation prior to this layer must be differentiated w.r.t. some
    quantities s.a. the loss.

    :param incoming: incoming layer
    :type incoming: Layer
    :param window: size of the specgram window
    :type window: int
    :param overlapp: overlapp of the window
    :type overlapp: int
    :param window_func: apodization function
    :type window_func: func
    """
    def __init__(self, incoming, window,overlapp,
                window_func=lambda x:np.ones_like(x)):
        super().__init__(incoming)
        # since we go from a 2d input to 3d input we need to
        # set up by hand the data_format for the next layer
        # we follow same order of channel versus spatial dims
        # we also get the time length

        if incoming.data_format=='NCT':
            self.data_format = 'NCHW'
            self.time_length = self.in_shape[2]
            self.n_channels  = self.in_shape[1]
            self.n_windows   = (time_length-window)//(window-overlapp)+1
            self.out_shape   = (self.in_shape[0],self.n_channels,window,n_windows)
        else:
            self.data_format = 'NHWC'
            self.time_length = self.in_shape[1]
            self.n_windows   = (time_length-window)//(window-overlapp)+1
            self.n_channels  = self.in_shape[2]
            self.out_shape   = (self.in_shape[0],window,n_windows,self.n_channels)
        # If input is a layer
        if self.given_input:
            self.forward(incoming.output,training=training)
    def forward(self,input,training=None,**kwargs):
        if self.data_format=='NCHW':
            input = tf.transpose(input,[0,2,1])

        patches = tf.reshape(tf.extract_image_patches(tf.expand_dims(input,1),
                    [1,1,self.window,1],[1,1,self.hop,1],[1,1,1,1]),
                    [self.in_shape[0],self.n_windows,self.window,self.n_channels])
        output  = tf.abs(tf.rfft(tf.transpose(patches,[0,1,3,2])))
        if self.data_format=='NCHW':
            self.output = tf.transpose(output,[0,2,3,1])
        else:
            self.output = tf.transpose(output,[0,3,1,2])
        self.VQ     = None # to change
        return self.output



class LambdaFunction(Layer):
    """applies a lambda function onto the layer input
    This layer allows to apply an arbitrary given function onto
    its input tensor allows to implement arbitrary operations.
    The fiven function must allow backpropagation through it
    if leanring with backpropagation is required, the function
    can alter the shape of the input but a func_shape must be provided
    which outputs the new shape given the tensor one.

    Example of use::

        input_shape = [10,1,32,32]
        # simple case of renormalization of all the values
        # by maximum value
        layer = LambdaFunction(input_shape,func= lambda x:x,sh/tf.reduce_max(x))
        # more complex case with shape alteration taking only the first
        # half of the second dimension
        def my_func(x,x_shape):
            return x[:,:x_shape[1]//2]
        def my_shape_func(x_shape):
            new_shape = x_shape
            new_shape[1]=new_shape[1]//2
            return new_shape

        layer = LambdaFunction(input_shape,func=my_func,
                            shape_func = my_shape_func)

    :param incoming: input shape of tensor
    :type incoming: shape or :class:`Layer` instance
    :param func: function to be applied taking as input the tensor
    """
    def __init__(self,incoming,func, shape_func = None):
        super().__init__(incoming)
        self.func = func
        if shape_func is None:
            self.shape_func = lambda x:x
        self.out_shape = self.shape_func(self.in_shape)
        if self.given_input:
            self.forward(input)
    def forward(self,input,**kwargs):
        self.output = self.func(input)
        return self.output
