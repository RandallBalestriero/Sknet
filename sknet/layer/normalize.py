import tensorflow as tf
from . import Layer

def BatchNormalization:
    """applies batch-normalization onto the input
    Given some :py:data:`axis`, applies batch-normalization onto the 
    input on those axis. This applies udring training a moment based
    rneormalization, and during testing, the moving average of the
    moments encoutered during training. The moments are computed per batch

    :param incoming: input shape or incoming layer
    :type incoming: shape or :class:`Layer`
    :param axis: the axis to normalize
    :type axis: tuple or list of ints
    :param training: variable representing the state of the model, if
                     training or testing time
    :type training: tf.bool
    :param beta_initializer: initializer of the beta parameter
    :type beta_initializer: initializer or array
    :param gamma_initializer: initializer of the gamma parameter
    :type gamma_initializer: initializer or array
    :param name: name of the layer
    :type name: str
    :param epsilon: (optional) the epsilon constant to add 
                    to the renormalization
    :type epsilon: scalar
    :param decay: the decay to use for the exponential
                  moving average to compute the test time
                  moments form the training batches ones
    :type decay: scalar
    """
    def __init__(self,incoming,axis,training=None, beta_initializer = tf.zeros, 
                        gamma_initializer=tf.ones, name='bn_layer', 
                        epsilon=1e-6, decay=0.99)
    	super().__init__(incoming)

        self.eps   = epsilon
        self.decay = decay

    	# Infer the shape of the parameters, it is 1 for the axis that are
    	# being normalized over and the same as the input shape for the others
    	shape_        = [s if i not in axis else 1 for i,s in enumerate(self.in_shape)]
    	# Initialize all the variables
    	self.beta     = tf.Variable(beta_initializer(shape_),
					trainable=center,
					name=name+'_beta')
    	self.gamma    = tf.Variable(gamma_initializer(shape_),
					trainable=scale,
					name=name+'_gamma')
    	self.mov_mean = tf.Variable(tf.zeros(shape_),
					trainable=False,
					name=name+'_movingmean')
    	self.mov_var  = tf.Variable(tf.ones(shape_),
					trainable=False,
					name=name+'_movingvar')
        # we create a dummy variable to track if the forward method
        # to ensure that we do not duplicate the update ops
	self._first_pass = True
        if self.given_input:
            self.forward(incoming.output,training=training)

    def forward(self,input, training, **kwargs):
    	# Compute the update of the parameters
    	# first get the first and second order moments
    	moments   = tf.nn.moments(tensor,axes=axis,keep_dims=True)

    	# compute the update values
    	new_mean    = self.moving_mean*decay+moments[0]*(1-decay)
    	new_var     = self.moving_var*decay+moments[1]*(1-decay)

        if self._first_pass:
    	    # compute the update operations
    	    update_mean = tf.assign(moving_mean,tf.cond(training,
				lambda :new_mean, lambda :moving_mean))
    	    update_var  = tf.assign(moving_var,tf.cond(training,
				lambda :new_var, lambda :moving_var))
    	    update_ops  = tf.group(update_mean,update_var)
    	    # add them to the update ops collection
    	    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,update_ops)
            self._first_pass = False

        # compute the mean and var to use, depending is training or testing
        use_mean = tf.cond(training,lambda :moments[0],lambda :new_mean)
        use_std = tf.sqrt(tf.cond(training,lambda :moments[1],lambda :new_var)\
			+self.eps)
        self.output = self.gamma*(input-use_mean)/use_std+self.beta
        # we also compute those quantities that allow to rewrite the output as
        # A*input+b, this might be of use for research pruposes
        self.b = -(gamma*use_mean)/use_std+beta
        self.A = gamma/use_std
        return self.output





