import tensorflow as tf
from . import Layer

def _batch_normalization(tensor,tensor_shape, axis,training,
                    beta_initializer = tf.zeros, gamma_initializer=tf.ones, 
                    center=True, scale=True, name='bn_layer', epsilon=1e-6,
                    decay=0.99):
    shape_          = [s if i not in axis else 1 for i,s in enumerate(input_shape)]
    beta            = tf.Variable(beta_initializer(shape_),trainable=center,name=name+'_beta')
    gamma           = tf.Variable(gamma_initializer(shape_),trainable=scale,name=name+'_gamma')
    moving_mean     = tf.Variable(tf.zeros(shape_),trainable=False,name=name+'_movingmean')
    moving_var      = tf.Variable(tf.ones(shape_),trainable=False,name=name+'_movingvar')
    cpt             = tf.Variable(tf.ones(1),trainable=False,name=name+'cpt')
    moments         = tf.nn.moments(tensor,axes=axis,keep_dims=True)
    coeff           = (cpt-1.)/cpt
    update_mean     = tf.assign(moving_mean,tf.cond(training,lambda :moving_mean*decay+moments[0]*(1-decay),lambda :moving_mean))
    update_var      = tf.assign(moving_var,tf.cond(training,lambda :moving_var*decay+moments[1]*(1-decay),lambda :moving_var))
    update_cpt      = tf.assign_add(cpt,tf.cond(training,lambda :tf.ones(1),lambda :tf.zeros(1)))
    update_ops      = tf.group(update_mean,update_var,update_cpt)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,update_ops)
    bias    = -(gamma*tf.cond(training,lambda :moments[0],lambda :moving_mean))/tf.sqrt(tf.cond(training,lambda :moments[1],lambda :moving_var)+epsilon)+beta
    scaling = gamma/tf.sqrt(tf.cond(training,lambda :moments[1],lambda :moving_var)+epsilon)
    return scaling,bias





class batchnormalization(Layer):
    def __init__(self,incoming,axis,training=None,
        beta_initializer = tf.zeros, gamma_initializer=tf.ones,
        name='', eps=1e-6, decay=0.99, trainable=True):
        super().__init__(incoming)
        
        # Set up the shape of the parameters
        shape_          = [s if i not in axis else 1 for i,s in enumerate(self.in_shape)]
        
        # Set attributes
        self.eps         = eps
        self.added_ops   = False
        self.VQ          = None
        self.axis        = axis
        self.decay       = decay
        
        # Initialize the learnable parameters
        self.beta        = tf.Variable(beta_initializer(shape_),
                               trainable=trainable,
                               name='bnlayer_beta'+name)
        self.gamma       = tf.Variable(gamma_initializer(shape_),
                               trainable=trainable,
                               name='bnlayer_gamma'+name)
                               
        # initialize the rolling statistics
        self.moving_mean = tf.Variable(tf.zeros(shape_), 
                                trainable=False,
                                name='bnlayer_movingmean'+name)
        self.moving_var  = tf.Variable(tf.ones(shape_),
                                trainable=False,
                                name='bnlayer_movingvar'+name)
        if self.given_input:
            self.forward(incoming,training=training)
    def forward(self,input,training):
        A,b = self.get_A_b(self,input,training)
        return A*input+b
    def get_A_b(self,input,training):
    
        # Compute moments of the inputs
        moments         = tf.nn.moments(input,axes=self.axis,keep_dims=True)
        
        # Update moments OPs, ensure that there is no replica
        if not self.added_ops:
            self.added_ops  = True
            updated_mean    = self.moving_mean*self.decay\
                                    +moments[0]*(1-self.decay)
            updated_var     = self.moving_var*self.decay\
                                    +moments[1]*(1-self.decay)
            update_mean     = tf.assign(self.moving_mean,tf.cond(training,
                                lambda :updated_mean,lambda :self.moving_mean))
            update_var      = tf.assign(self.moving_var,tf.cond(training,
                                lambda :updated_var,lambda :self.moving_var))
            update_ops      = tf.group(update_mean,update_var)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,update_ops)
            
        # Compute input renormalization
        _mean = tf.cond(training,lambda :moments[0],lambda :self.moving_mean)
        _var  = tf.cond(training,lambda :moments[1],lambda :self.moving_var)
        
        # bias
        b  = -(self.gamma*_mean)/tf.sqrt(_var+self.eps)+self.beta
        
        # slope
        A  = self.gamma/tf.sqrt(_var+self.eps)
        
        return A,b





