import tensorflow as tf
from .normalization import batch_normalization


class InputLayer:
    def __init__(self,input_shape,x):
        self.output       = x
        self.output_shape = input_shape



class DenseLayer:
    def __init__(self, incoming, n_units, nonlinearity = tf.nn.relu,
                training=None, batch_norm = False,
                init_W = tf.contrib.layers.xavier_initializer(uniform=True)):
        # Set up the input, flatten if needed
        if(len(incoming.output_shape)>2):
            inputf = tf.layers.flatten(incoming.output)
            in_dim = prod(incoming.output_shape[1:])
        else:
            inputf = incoming.output
            in_dim = incoming.output_shape[1]
        # Output Shape
        self.output_shape = (incoming.output_shape[0],n_units)
        # Param Inits
        self.W            = tf.Variable(init_W((in_dim,n_units)),
                            name='W_dense',trainable=True)
        Wx                = tf.matmul(inputf,self.W)
        if(batch_norm):  
            self.scaling,self.bias = batch_normalization(Wx,axis=[0],training=training)
        else:            
            self.scaling = 1
            self.bias    = tf.Variable(tf.zeros((1,n_units)),trainable=True, name='denselayer_b')
        self.S            = self.scaling*Wx+self.bias
        self.VQ           = tf.greater(self.S,0)
        if(nonlinearity is tf.nn.relu):
            self.mask = tf.cast(self.VQ,tf.float32)
        elif(nonlinearity is tf.abs):
            self.mask = tf.cast(self.VQ,tf.float32)*2-1
        elif(nonlinearity is tf.nn.leaky_relu):
            self.mask = tf.cast(self.VQ,tf.float32)*0.8+0.2
        elif(nonlinearity is tf.identity):
            self.VQ   = None
        self.output   = self.S*self.mask
#    def backward(self,output):
#        """output is of shape (batch,n_output)
#        return of this function is of shape [(batch,in_dim),(batch,1)]"""
#        # use tf.nn.conv2d_backprop_input for conv2D
#        A = tf.reshape(tf.matmul(output*self.mask*scaling,self.W,transpose_b=True),incoming.output_shape)
#        B = tf.matmul(output*self.mask,self.b,transpose_b=True)
#        return A,B


class OutputLayer:
    def __init__(self, incoming, n_classes,
                init_W = tf.contrib.layers.xavier_initializer(uniform=True)):
        # Set up the input, flatten if needed
        if(len(incoming.output_shape)>2):
            inputf = tf.layers.flatten(incoming.output)
            in_dim = prod(incoming.output_shape[1:])
        else:
            inputf = incoming.output
            in_dim = incoming.output_shape[1]
        # Output Shape
        self.output_shape = (incoming.output_shape[0],n_classes)
        # Param Inits
        self.W            = tf.Variable(init_W((in_dim,n_classes)),
                            name='W_dense',trainable=True)
        Wx                = tf.matmul(inputf,self.W)
        self.bias         = tf.Variable(tf.zeros((1,n_classes)),
                                trainable=True, name='denselayer_b')
        self.S            = Wx+self.bias
        self.VQ           = tf.argmax(self.S,1)
        self.output       = self.S
#    def backward(self,output):
#        """output is of shape (batch,n_output)
#        return of this function is of shape [(batch,in_dim),(batch,1)]"""
#        # use tf.nn.conv2d_backprop_input for conv2D
#        A = tf.reshape(tf.matmul(output*self.mask*scaling,self.W,transpose_b=True),incoming.output_shape)
#        B = tf.matmul(output*self.mask,self.b,transpose_b=True)
#        return A,B




class ConstraintDenseLayer:
    def __init__(self,incoming,n_output,constraint='none',training=None):
        # bias_option : {unconstrained,constrained,zero}
        if(len(incoming.output_shape)>2): reshape_input = tf.layers.flatten(incoming.output)
        else:                             reshape_input = incoming.output
        in_dim      = prod(incoming.output_shape[1:])
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
        self.output_shape = (incoming.output_shape[0],n_output)
        self.output       = tf.matmul(reshape_input,self.W)
        self.VQ = None



class Conv2DLayer:
    def __init__(self,incoming,n_filters,filter_shape,nonlinearity = tf.nn.relu,
                    training=None, batch_norm = True, data_format='NCHW',
                    init_W = tf.contrib.layers.xavier_initializer(uniform=True),
                    stride=1,pad='valid',mode='CONSTANT'):
        if pad=='valid' or filter_shape==1:
            padded_input = incoming.output
        else:
            if pad=='same':
                assert(filter_shape%2 ==1)
                p = (filter_shape-1)/2
            else:
                p = filter_shape-1
            if data_format=='NCHW':
                padded_input = tf.pad(incoming.output,[[0,0],[0,0],[p,p],[p,p]],mode=mode)
            else:
                padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
        self.W      = tf.Variable(init_W((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),
                            name='W_conv2d',trainable=True)
        Wx                = tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],
                                padding='VALID', data_format=data_format)
        self.output_shape = Wx.get_shape().as_list()
        if(batch_norm):  
            self.scaling,self.bias = batch_normalization(Wx,[0,1,2],training=training,center=trainable,scale=trainable)
        else:            
            self.scaling = 1
            self.bias    = tf.Variable(tf.zeros((1,1,1,n_filters)),
                                trainable=True,name='convlayer_b')
        self.S            = self.scaling*Wx+self.bias
        self.VQ           = tf.greater(self.S,0)
        if(nonlinearity is tf.nn.relu):
            self.mask = tf.cast(self.VQ,tf.float32)
        elif(nonlinearity is tf.abs):
            self.mask = tf.cast(self.VQ,tf.float32)*2-1
        elif(nonlinearity is tf.nn.leaky_relu):
            self.mask = tf.cast(self.VQ,tf.float32)*0.8+0.2
        self.output   = self.S*self.mask

#        W_norm            = tf.sqrt(tf.reduce_sum(tf.square(self.W*self.scaling),[0,1,2]))
#        self.positive_radius = tf.reduce_min(tf.where(tf.greater(self.S,tf.zeros_like(self.S)),self.S,tf.reduce_max(self.S,keepdims=True)),[1,2,3])
#        self.negative_radius = tf.reduce_min(tf.where(tf.smaller(self.S,tf.zeros_like(self.S)),tf.abs(self.S),tf.reduce_max(tf.abs(self.S),keepdims=True)),[1,2,3])
#        print(self.output.get_shape().as_list())










