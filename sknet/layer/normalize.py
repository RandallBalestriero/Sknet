import tensorflow as tf
from . import Layer
from ..utils import init_variable as init_var

class BatchNormalization:
    """applies batch-normalization onto the input
    Given some :py:data:`axis`, applies batch-normalization onto the 
    input on those axis. This applies udring training a moment based
    rneormalization, and during testing, the moving average of the
    moments encoutered during training. The moments are computed per batch

    To remove the learnable scaling or offset one can set 
    :py:data:`gamma=False` and/or :py:data:`beta=False`. 
    Even though one can let them as :py:data:`True` and give a constant
    value, this is no recommander as the implementation optimizes the
    computation otherwise.

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
                        epsilon=1e-6, decay=0.99, gamma=True, 
                        beta=True, **kwargs):
        super().__init__(incoming, **kwargs)

        # set attributes
        self.eps    = epsilon
        self.decay  = decay
        self._gamma = gamma
        self._beta  = beta
        self.initialized = False
        if np.isscalar(axis):
            self.axis = [axis]
        else:
            self.axis = axis

    	# Infer the shape of the parameters, it is 1 for the axis that are
    	# being normalized over and the same as the input shape for the others
        shape_        = [s if i not in self.axis else 1 for i,s in enumerate(self.in_shape)]

        # we create a dummy variable to track if the forward method
        # to ensure that we do not duplicate the update ops
        self._first_pass = True
        if self.given_input:
            self.initialized_variables()
            self.forward(incoming.output,dataerministic = deterministic)

    def initialize_variables(self):
        # Initialize all the variables
        if not self.initialized:
            if self._beta:
                self.beta = init_var(self.beta_initializer,shape_,
					trainable=center,
					name=name+'_beta')
            else:
                self.beta = tf.zeros(shape_)
            if self._gamma:
                self.gamma = init_var(self.gamma_initializer,shape_,
					trainable=scale,
					name=name+'_gamma')
            else:
                self.gamma = tf.ones(shape_)
            self.mov_mean = init_var(tf.zeros,shape_,
					trainable=False,
					name=name+'_movingmean')
            self.mov_var  = init_var(tf.ones,shape_,
					trainable=False,
					name=name+'_movingvar')
            self.initialized = True

    def forward(self,input, training, **kwargs):
    	# Compute the update of the parameters
    	# first get the first and second order moments
        moments   = tf.nn.moments(tensor,axes=self.axis,keep_dims=True)

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
        use_std  = tf.sqrt(tf.cond(training,lambda :moments[1],lambda :new_var)\
			+self.eps)
        
        # we also compute those quantities that allow to rewrite the output as
        # A*input+b, this might be of use for research pruposes
        # case of using beta and gamma
        if self._beta and self._gamma:
            self.output = self.gamma*(input-use_mean)/use_std+self.beta
            self.b = -(self.gamma*use_mean)/use_std+self.beta
            self.A = self.gamma/use_std
        # case of using beta and not gamma
        elif self_beta and not self._gamma:
            self.output = (input-use_mean)/use_std+self.beta
            self.b = -use_mean/use_std+self.beta
            self.A = 1/use_std
        # case of using gamma and not beta
        elif not self_beta and self._gamma:
            self.output = self.gamma*(input-use_mean)/use_std
            self.b = -(self.gamma*use_mean)/use_std
            self.A = self.gamma/use_std
        # case of using neither of gamma or beta
        else:
            self.output = (input-use_mean)/use_std
            self.b = -use_mean/use_std
            self.A = 1/use_std
        return self.output

